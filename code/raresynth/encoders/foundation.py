"""
Frozen foundation-model encoders for the remaining four modalities.

All encoders are used in inference mode with public released weights.  None is
fine-tuned, so the latent spaces are reproducible by anyone who downloads the
same checkpoints.

The pathology path uses stream-and-discard: download a slide, tile it, embed
the tiles, aggregate to a slide vector, delete the slide.  Holding 18 TB of
TCGA and GTEx SVS files on disk is unnecessary and is the reason projects of
this shape usually stall.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Transcriptomics -- Geneformer
# --------------------------------------------------------------------------

class GeneformerEncoder:
    """Rank-value encoding of bulk expression, then mean of last-4 [CLS].

    Geneformer was pretrained on single cells; applying it to bulk RNA-seq is
    an out-of-distribution use and should be stated as such in the Methods.
    The rank encoding is what makes it defensible -- Geneformer consumes gene
    rank order, not magnitude, and bulk rank order is a coherent quantity.
    A scVI/scGPT alternative is provided as an ablation in the paper because
    reviewers will ask whether the choice of RNA encoder drives the result.
    """

    def __init__(self, model_name="ctheodoris/Geneformer", n_tokens=2048,
                 device="cuda", median_dict=None):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.n_tokens = n_tokens
        self.median_dict = median_dict  # gene -> non-zero median for normalisation
        self.token_map = None           # gene ensembl id -> token id

    def load_token_map(self, path):
        import pickle

        with open(path, "rb") as fh:
            self.token_map = pickle.load(fh)
        return self

    def rank_encode(self, expr: dict):
        """expr: ensembl_id -> TPM. Returns a token id list of length n_tokens."""
        if self.median_dict:
            scored = {g: v / self.median_dict.get(g, 1.0)
                      for g, v in expr.items() if v > 0}
        else:
            scored = {g: v for g, v in expr.items() if v > 0}
        ordered = sorted(scored.items(), key=lambda kv: -kv[1])
        toks = [self.token_map[g] for g, _ in ordered
                if g in self.token_map][: self.n_tokens]
        return toks

    @torch.no_grad()
    def embed(self, token_batches, batch_size=16):
        out = []
        for i in range(0, len(token_batches), batch_size):
            chunk = token_batches[i : i + batch_size]
            L = max(len(c) for c in chunk)
            ids = torch.zeros(len(chunk), L, dtype=torch.long)
            mask = torch.zeros(len(chunk), L, dtype=torch.long)
            for j, c in enumerate(chunk):
                ids[j, : len(c)] = torch.tensor(c)
                mask[j, : len(c)] = 1
            o = self.model(input_ids=ids.to(self.device),
                           attention_mask=mask.to(self.device),
                           output_hidden_states=True)
            hs = o.hidden_states[-4:]
            # mean over the last four layers of the mask-weighted mean token
            m = mask.to(self.device).unsqueeze(-1).float()
            pooled = torch.stack([(h * m).sum(1) / m.sum(1) for h in hs]).mean(0)
            out.append(pooled.cpu())
        z = torch.cat(out)
        return (z / (z.norm(dim=-1, keepdim=True) + 1e-8)).numpy()


# --------------------------------------------------------------------------
# Histopathology -- UNI + ABMIL
# --------------------------------------------------------------------------

class GatedABMIL(nn.Module):
    """Gated attention pooling over tile embeddings (Ilse et al., 2018)."""

    def __init__(self, d_in=768, d_hidden=256, d_out=1024):
        # d_in default changed from the original 1024 (assumed UNI, 303M
        # params, gated access never came through) to 768, CTransPath's
        # real confirmed tile-embedding output dim -- the actual pathology
        # pipeline built and run in this project. d_out=1024 is unaffected
        # (ABMIL's own aggregation output width, independent of the input
        # tile embedding's width) and still matches ModalitySpec's "path".
        # If UNI access is ever granted and used as a secondary encoder,
        # instantiate with d_in=1024 explicitly for that run.
        super().__init__()
        self.V = nn.Linear(d_in, d_hidden)
        self.U = nn.Linear(d_in, d_hidden)
        self.w = nn.Linear(d_hidden, 1)
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, H, mask=None):
        """H: (B, K, d_in)."""
        a = self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H)))  # (B,K,1)
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(-1), float("-inf"))
        a = torch.softmax(a, dim=1)
        z = self.proj((a * H).sum(1))
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8), a.squeeze(-1)


def tile_slide(svs_path, tile_px=256, level_mpp=0.5, max_tiles=4000,
               bg_lum=220, blur_thresh=100.0, seed=0):
    """Otsu tissue mask, then grid tiling at ~20x with blur and background QC."""
    import cv2
    import openslide
    from skimage.filters import threshold_otsu

    slide = openslide.OpenSlide(str(svs_path))
    mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.5))
    level = 0
    scale = level_mpp / mpp
    size = int(round(tile_px * scale))

    thumb = np.array(slide.get_thumbnail((2048, 2048)).convert("L"))
    try:
        thr = threshold_otsu(thumb)
    except ValueError:
        slide.close()
        return []
    mask = thumb < thr
    W, H = slide.level_dimensions[level]
    fx, fy = thumb.shape[1] / W, thumb.shape[0] / H

    coords = []
    for y in range(0, H - size, size):
        for x in range(0, W - size, size):
            ty, tx = int(y * fy), int(x * fx)
            th, tw = max(1, int(size * fy)), max(1, int(size * fx))
            if mask[ty : ty + th, tx : tx + tw].mean() > 0.5:
                coords.append((x, y))

    rng = np.random.default_rng(seed)
    if len(coords) > max_tiles:
        coords = [coords[i] for i in rng.choice(len(coords), max_tiles, replace=False)]

    tiles = []
    for x, y in coords:
        img = np.array(slide.read_region((x, y), level, (size, size)).convert("RGB"))
        if size != tile_px:
            img = cv2.resize(img, (tile_px, tile_px), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if g.mean() > bg_lum:
            continue
        if cv2.Laplacian(g, cv2.CV_64F).var() < blur_thresh:
            continue
        tiles.append(img)
    slide.close()
    return tiles


class UNIEncoder:
    """UNI ViT-L/14 tile encoder. Requires HF access approval for the weights."""

    def __init__(self, device="cuda", model_name="hf-hub:MahmoodLab/UNI"):
        import timm
        from torchvision import transforms

        self.model = timm.create_model(
            model_name, pretrained=True, init_values=1e-5, dynamic_img_size=True
        ).to(device).eval()
        self.device = device
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def embed_tiles(self, tiles, batch_size=128, amp=True):
        out = []
        for i in range(0, len(tiles), batch_size):
            b = torch.stack([self.tf(t) for t in tiles[i : i + batch_size]])
            with torch.amp.autocast("cuda", enabled=amp and self.device == "cuda"):
                out.append(self.model(b.to(self.device)).float().cpu())
        return torch.cat(out) if out else torch.zeros(0, 1024)


class _ConvStem(nn.Module):
    """CTransPath's patch embedding: NOT a single conv (timm's default),
    but 2x [Conv2d(stride=2)-BatchNorm2d-ReLU] then a final 1x1 Conv2d.

    This exact structure was confirmed two ways before being used here:
    (1) the community HF mirror's README quotes the authors' own ctran.py
    lines 6-44 describing this structure; (2) more importantly, a LIVE
    state_dict-mismatch error from timm.create_model(..., pretrained=True)
    against this exact checkpoint revealed the real parameter keys
    directly (patch_embed.proj.{0,1,3,4,6} have parameters, {2,5} do not --
    matching Conv-BN-ReLU-Conv-BN-ReLU-Conv exactly), which is stronger
    evidence than the README alone. The community mirror's automatic
    hf-hub config did NOT correctly wire embed_layer=ConvStem into the
    model it built (it silently fell back to a standard single-conv patch
    embed), so this class is used with an EXPLICIT timm.create_model(...,
    embed_layer=_ConvStem, pretrained=False) call, with the real weights
    then loaded directly and separately -- not relying on timm's
    pretrained=True path for this repo, which is confirmed broken.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                norm_layer=None, flatten=True, **kwargs):
        super().__init__()
        from timm.layers import to_2tuple

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for _ in range(2):
            stem += [
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2,
                         padding=1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            ]
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            # NOTE: earlier drafted this as x.flatten(2).transpose(1, 2) --
            # a (B, N, C) flattened sequence, matching an older timm
            # PatchEmbed convention described in the community mirror's
            # README snippet. A live forward pass through the actual
            # installed timm SwinTransformer failed with a direct shape
            # error ("not enough values to unpack (expected 4, got 3)")
            # inside its internal block code, which expects patch_embed
            # output in SPATIAL (B, H, W, C) format, not a flattened
            # sequence -- this timm version's Swin implementation operates
            # on spatial grids internally. Permuting to channels-last
            # spatial format instead, confirmed correct by a full
            # end-to-end forward pass (see verification notes).
            x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        return self.norm(x)


def _remap_ctranspath_state_dict(state_dict, model):
    """Fix two confirmed real mismatches between the checkpoint (saved with
    an older timm Swin convention) and the currently installed timm's
    SwinTransformer:

    1. Downsample (patch-merging) index shift. The checkpoint attaches each
    stage's downsample module to the END of that stage (layers 0,1,2 have
    one, preparing input for the next stage; layer 3, the last stage, does
    not). This installed timm version attaches it to the START of the
    stage instead (layers 1,2,3 have one; layer 0 does not). Same three
    modules, shifted by one stage index. Confirmed by directly comparing
    the real checkpoint's reported shapes against this model's real
    state_dict shapes before writing this fix (see chat/PROGRESS.md):
    checkpoint layers.1 (768) matches model layers.2 (768); checkpoint
    layers.2 (1536) matches model layers.3 (1536).

    2. relative_position_index / attn_mask are non-persistent buffers in
    this timm version (recomputed fresh at model-construction time from
    window size / image size, deliberately excluded from state_dict()).
    The checkpoint includes stale copies from whatever version it was
    originally saved with. These must be DROPPED, not loaded -- even if a
    shape happened to match, loading a stale position-dependent buffer
    would be wrong, not just unnecessary.
    """
    import re

    real_keys = set(model.state_dict().keys())
    remapped = {}
    dropped = []
    shifted = 0

    for k, v in state_dict.items():
        if "relative_position_index" in k or "attn_mask" in k:
            dropped.append(k)
            continue
        m = re.match(r"^layers\.(\d+)\.downsample\.(.*)$", k)
        if m:
            new_k = f"layers.{int(m.group(1)) + 1}.downsample.{m.group(2)}"
            if new_k in real_keys:
                remapped[new_k] = v
                shifted += 1
                continue
            # if the shifted key isn't real either, fall through and let it
            # surface as a genuine mismatch below rather than silently drop it
        remapped[k] = v

    unexpected = set(remapped.keys()) - real_keys
    missing = real_keys - set(remapped.keys())
    print(f"  state_dict remap: {shifted} downsample key(s) shifted, "
         f"{len(dropped)} stale buffer key(s) dropped, "
         f"{len(unexpected)} still-unexpected, {len(missing)} still-missing")
    if unexpected or missing:
        print(f"    still-unexpected (first 5): {sorted(unexpected)[:5]}")
        print(f"    still-missing (first 5): {sorted(missing)[:5]}")
    return remapped


class CTransPathEncoder:
    """CTransPath tile encoder -- the primary pathology encoder for this
    project (see PROGRESS.md/MANUSCRIPT_NOTES.md for why: UNI requires
    gated HuggingFace access with an unpredictable approval timeline;
    CTransPath is open, trained natively on TCGA+PAIP, smaller (28M vs
    UNI's 303M params, faster at our scale), and UNI remains available as
    a secondary comparison if/when access is granted).

    The model is built EXPLICITLY with the correct _ConvStem embed_layer
    and loaded from raw weights directly (huggingface_hub.hf_hub_download +
    load_state_dict(strict=True)) rather than via
    timm.create_model(..., pretrained=True) -- that path was tried first
    and failed with a confirmed, informative state_dict mismatch: the
    community HF mirror's automatic config did not apply the custom
    ConvStem, so timm silently built a standard (wrong) patch embedding.
    strict=True on the direct-load path is deliberate and matches the
    original authors' own loading code exactly -- if the real checkpoint
    ever doesn't match _ConvStem's structure (e.g. the mirror changes),
    this will fail loudly instead of silently loading a wrong or partial
    model.

    Preprocessing confirmed from the authors' own get_features_CTransPath.py:
    resize to 224x224 (Swin's window-size=7 constraint requires this exact
    input size), then standard ImageNet normalization.
    """

    def __init__(self, device="cuda",
                hf_repo="1aurent/swin_tiny_patch4_window7_224.CTransPath",
                img_size=224):
        import timm
        from torchvision import transforms
        from huggingface_hub import hf_hub_download, list_repo_files

        # find the actual weights file rather than guessing a filename --
        # HF repos vary between pytorch_model.bin and model.safetensors
        repo_files = list_repo_files(hf_repo)
        weight_candidates = [f for f in repo_files
                            if f.endswith((".bin", ".safetensors"))
                            and "optimizer" not in f]
        if not weight_candidates:
            raise FileNotFoundError(
                f"no .bin/.safetensors weight file found in {hf_repo} -- "
                f"repo contents: {repo_files}"
            )
        weight_file = weight_candidates[0]
        print(f"  CTransPath: downloading {weight_file} from {hf_repo}")
        ckpt_path = hf_hub_download(hf_repo, weight_file)

        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            embed_layer=_ConvStem,
            pretrained=False,
            num_classes=0,
        )
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)  # authors' own checkpoints
                                                  # wrap the state dict in a
                                                  # "model" key; a raw
                                                  # state_dict has no such
                                                  # wrapper, handle both
        self.model.load_state_dict(
            _remap_ctranspath_state_dict(state_dict, self.model), strict=True
        )
        self.model = self.model.to(device).eval()
        self.device = device
        self.tf = transforms.Compose([
            transforms.ToPILImage(),  # tile_slide() returns HWC uint8 numpy
                                      # arrays; Resize expects PIL/tensor
                                      # input, not raw numpy -- confirmed by
                                      # a live TypeError before this fix.
                                      # This also matches the authors' own
                                      # preprocessing exactly (their script
                                      # used Image.open(...).convert('RGB')
                                      # before the same Resize/ToTensor/
                                      # Normalize chain).
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def embed_tiles(self, tiles, batch_size=256, amp=True):
        """Same interface as UNIEncoder.embed_tiles, so downstream code can
        swap between the two without changes. tiles: list of HxWx3 uint8
        RGB numpy arrays (exactly what tile_slide() produces).
        """
        out = []
        for i in range(0, len(tiles), batch_size):
            b = torch.stack([self.tf(t) for t in tiles[i : i + batch_size]])
            with torch.amp.autocast("cuda", enabled=amp and self.device == "cuda"):
                out.append(self.model(b.to(self.device)).float().cpu())
        return torch.cat(out) if out else torch.zeros(0, 768)


def encode_slide_stream(svs_url, uni, abmil, tmp_dir="/tmp/wsi", **tile_kw):
    """Download -> tile -> embed -> aggregate -> delete. Peak disk = one slide."""
    from .. data.sources import download

    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    local = tmp / Path(svs_url).name
    try:
        download(svs_url, local)
        tiles = tile_slide(local, **tile_kw)
        if not tiles:
            return None, 0
        H = uni.embed_tiles(tiles).unsqueeze(0).to(next(abmil.parameters()).device)
        with torch.no_grad():
            z, _ = abmil(H)
        return z.squeeze(0).cpu().numpy(), len(tiles)
    finally:
        if local.exists():
            os.unlink(local)
        gc.collect()


# --------------------------------------------------------------------------
# Radiology
# --------------------------------------------------------------------------

class RadiologyEncoder:
    """3D volume -> pooled CNN features -> PCA to 512.

    The PCA basis is fitted on TCIA normal-appearing and tumour-adjacent
    volumes rather than UK Biobank, which is not part of the open-access
    configuration.
    """

    def __init__(self, device="cuda", d_out=512):
        import monai
        from monai.networks.nets import resnet50

        self.net = resnet50(spatial_dims=3, n_input_channels=1,
                            feed_forward=False).to(device).eval()
        self.device = device
        self.d_out = d_out
        self.pca = None

    @torch.no_grad()
    def features(self, volumes, batch_size=4):
        out = []
        for i in range(0, len(volumes), batch_size):
            b = torch.as_tensor(np.stack(volumes[i : i + batch_size]),
                                dtype=torch.float32).unsqueeze(1)
            out.append(self.net(b.to(self.device)).flatten(1).cpu().numpy())
        return np.concatenate(out)

    def fit_pca(self, feats):
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=self.d_out, random_state=0).fit(feats)
        return self

    def transform(self, feats):
        z = self.pca.transform(feats)
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


# --------------------------------------------------------------------------
# Clinical text
# --------------------------------------------------------------------------

class ClinicalEncoder:
    """Bio_ClinicalBERT over a templated clinical summary.

    The weights are public and need no credentialing, so the EHR modality
    survives the open-access-only configuration.  What does not survive is
    MIMIC-IV as a healthy-baseline corpus; the baseline is instead the mean
    embedding of TCGA normal-adjacent cases and GTEx donor records.
    """

    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT",
                 device="cuda", max_len=512):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device, self.max_len = device, max_len

    @staticmethod
    def format_record(age=None, sex=None, hpo_terms=(), labs=None,
                      site=None, history=None):
        parts = []
        if age is not None or sex is not None:
            parts.append(f"Patient is a {age or 'unknown'}-year-old "
                         f"{sex or 'individual'}.")
        if site:
            parts.append(f"Primary site: {site}.")
        if hpo_terms:
            parts.append("Presenting features: " + "; ".join(hpo_terms) + ".")
        if labs:
            parts.append("Laboratory values: "
                         + ", ".join(f"{k} {v}" for k, v in labs.items()) + ".")
        if history:
            parts.append(f"History: {history}.")
        return " ".join(parts) if parts else "No clinical information recorded."

    @torch.no_grad()
    def embed(self, texts, batch_size=32):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i : i + batch_size], padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu())
        z = torch.cat(out)
        return (z / (z.norm(dim=-1, keepdim=True) + 1e-8)).numpy()
