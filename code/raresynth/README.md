# RareSynth

Mechanism-guided multimodal latent diffusion for coherent biomedical patient
representations.

This repository implements the redesigned method. Two things differ from the
original manuscript draft and both are load-bearing.

**The mechanism is a sampling-time force, not a training corpus.** In the draft,
a perturbation network and a cross-modal mapper generated 100,000 synthetic
profiles, and the diffusion model was then trained on those profiles. Its
distribution was therefore upper-bounded by the two MLPs that produced them,
and nothing in the evaluation distinguished it from sampling those MLPs
directly and adding noise. Here the diffusion model is trained on real
embeddings only (TCGA, GTEx, CPTAC), and disease specificity enters through a
differentiable mechanism-consistency energy injected into the reverse process.
The guidance scale is a continuous ablation axis from zero upward.

**The denoiser respects modality boundaries.** The draft's 1-D U-Net convolved
across the concatenated latent, treating coordinate index as a spatial axis and
mixing the tail of one modality with the head of the next. MoDiT projects each
modality to a shared width and denoises five tokens with self-attention. The
U-Net is retained in `baselines/simple.py` so the change is measured rather than
asserted.

---

## Layout

```
raresynth/
  model/
    dit.py          MoDiT: modality-token diffusion transformer, adaLN-Zero
    diffusion.py    cosine schedule, DDIM, CFG + mechanism guidance, EMA
    guidance.py     mechanism-consistency energy E(x; g, tissue)
    ppn.py          perturbation network, gene k-NN imputer, reduced-rank
                    cross-modal direction prior
  encoders/
    genomic.py      gene-annotation set transformer (replaces per-variant Enformer)
    foundation.py   Geneformer, UNI+ABMIL with slide streaming, MONAI, ClinicalBERT
  baselines/
    multimodal_vae.py   MVAE (PoE) and MoPoE
    simple.py           flat 1-D U-Net, Gaussian copula, independent diffusion
  eval/
    fidelity.py     FID, PRDC, C2ST with permutation null
    coherence.py    CMCS + shuffled-pair control, mechanism retrieval
    privacy.py      DCR, membership inference
    tasks.py        imputation, few-shot diagnosis, McNemar
  data/
    sources.py      open-access acquisition (GDC, GTEx, CPTAC, LINCS, Zenodo)
  train_dit.py
scripts/
  smoke_test.py     end-to-end run on simulated data with known dependence
```

## Verify before downloading anything

```bash
pip install torch numpy scipy scikit-learn
python scripts/smoke_test.py
```

The simulation builds pathology, radiology and genomic latents as known linear
functions of the transcriptomic latent plus noise. A correct implementation
gives CMCS near 1 for MoDiT and near 0 for the shuffled-modality control, MIA
AUC near 0.5, and no exact duplicates. If that ordering does not appear on
simulated data with ground-truth dependence, it will not appear on real data
either, and the problem is the code.

Two failure modes already found and fixed here, both of which look like a
diverging sampler and neither of which is:

- `alpha_bar` floored at 1e-8 makes DDIM's `x0 = (x - sqrt(1-ab)*eps)/sqrt(ab)`
  amplify the epsilon error by 1e4 at the first step. Floored at 1e-4, with a
  configurable `x0_clip`.
- EMA at decay 0.9999 has a 10,000-step averaging window. A short run leaves
  the shadow weights ~43% initialisation. `EMA` now ramps the decay.

## Data

Open-access only. No DUA, no dbGaP, no institutional affiliation required.

| Role | Source | Modalities |
|---|---|---|
| Train / internal val | TCGA (GDC open tier) | geno, rna, path, rad, ehr |
| Train / normal baseline | GTEx v8 (RNA + portal WSIs, 20x) | geno, rna, path |
| Radiology | TCIA TCGA-matched collections | rad |
| **External validation 1** | CPTAC | rna, path, ehr |
| **External validation 2** | Kremer et al., Zenodo 3887451 | rna |
| Mechanism supervision | LINCS L1000, DepMap | rna |
| Priors | ClinVar, gnomAD v4, OMIM, Orphanet, HPO | — |

External validation 2 is the one that tests the paper's weakest joint: whether
perturbation structure learned from tumour/normal cancer pairs transfers to
germline monogenic disease. It is real Mendelian disease, in a clinically
accessible tissue, with known causal genes, released as ready-to-use gene-level
count matrices.

```bash
python -m raresynth.data.sources                 # print the registry
python -c "from raresynth.data.sources import build_tcga_manifest; build_tcga_manifest()"
```

Slides are streamed, not stored: download, tile, embed, aggregate, delete. Peak
disk stays under 1 TB rather than the ~18 TB a naive full pull would need.

## Compute budget

| Stage | Hardware | Wall time |
|---|---|---|
| Gene annotation tensor, whole cohort | CPU, 32 cores | ~4 h |
| Geneformer over ~30k RNA samples | 1x A100 | ~6 h |
| UNI tile embedding, ~14k slides | 4x A100 | ~5 days |
| RadImageNet over ~2.5k volumes | 1x A100 | ~2 h |
| ClinicalBERT | 1x A100 | ~20 min |
| PPN training + held-out-gene eval | 1x A100 | ~2 h |
| MoDiT, 400 epochs | 4x A100 | ~10 h |
| All baselines | 4x A100 | ~14 h |
| Full ablation grid | 4x A100 | ~3 days |

The pathology embedding dominates and is embarrassingly parallel — split the
slide manifest across nodes.

```bash
torchrun --nproc_per_node=4 -m raresynth.train_dit \
    --data embeddings.npz --out runs/modit_base --epochs 400 --amp
```

## Experiment matrix

**Baselines.** CTGAN, TVAE, medGAN, TabDDPM, MVAE, MMVAE, MoPoE, TotalVI, scVI
or scDiff on the RNA slice, Gaussian copula, flat 1-D U-Net, independent
per-modality diffusion.

The copula is not a formality. On simulated data with known dependence it
outscored the MVAE on cross-modal coherence, which is unsurprising — a copula
preserves rank-correlation structure by construction, and that is exactly what
CMCS measures. If it holds on real embeddings, the paper cannot claim coherence
as its differentiator without beating it.

**Ablations.** Guidance off (s = 0); guidance-scale sweep; cross-modal prior
removed from the energy; modality tokens replaced by flat concatenation;
HPO conditioning removed; leave-one-modality-out; held-out-gene PPN;
leave-one-cancer-type-out for the direction prior; synthetic-sample scaling;
RNA encoder swapped (Geneformer to scVI) to show the result is not an artefact
of one frozen encoder.

**Metrics.** FID, PRDC and C2ST per modality and joint; CMCS with its shuffled
control; mechanism-retrieval top-k and MRR; imputation R², cosine and
retrieval@k against MVAE/MoPoE/TotalVI; few-shot diagnosis with bootstrap
intervals; DCR and membership-inference AUC.

## Splits

Donor-disjoint, not row-disjoint. TCGA and GTEx both contribute several samples
per donor, and a random row split leaks the same individual across train and
test, inflating every fidelity metric. `train_dit.py` expects a `donor_id`
field and the split must be built from it.

## Not included, deliberately

UDN, the 100,000 Genomes Project and UK Biobank appear in the original draft
but none can supply five-modality rare-disease cases to any investigator, and
the paper should not claim data it cannot obtain. The rare-disease claim is
instead carried by external validation 2 plus the mechanism-retrieval and
held-out-gene experiments.
