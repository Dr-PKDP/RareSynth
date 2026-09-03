# RareSynth — Progress Log

Running record of what has been done, what broke, and why decisions were
made the way they were. Update this file at the end of each work session
rather than relying on chat history — chat history is not part of the repo
and will not travel with it.

Consolidated 2026-09-01: older resolved-bug narratives condensed into the
summary table below. Full blow-by-blow detail for bugs #1-11 lives in chat
history if ever needed; what's preserved here is the durable lesson, not
the debugging play-by-play.

---

## Status as of 2026-09-01

**Data acquisition: COMPLETE.** All planned sources downloaded and
integrity-verified (~3.0 TB core + ~28 GB LINCS/DepMap + ~31.5 GB
annotation sources). See Data Inventory below.

**Encoders — ALL 4 OF 4 COMPLETE and live-verified against real data:**

| Encoder | Status | Coverage |
|---|---|---|
| RNA-seq (Geneformer) | DONE | TCGA 4,255/4,255 (100%), CPTAC 2,146/2,164 (99.2%), Pfib_423 423/423 (100%) |
| Genomic (set-transformer, 14 features) | DONE | TCGA 4,070/4,070 (100%), full external pipeline (gnomAD/ClinVar/GTEx/annotation sources) built and verified today |
| Clinical (ClinicalBERT) | DONE, TCGA only | 4,756/4,759 (99.9%) |
| Pathology (CTransPath, bag-of-tiles) | DONE | TCGA 2,000/2,000 (100%), zero anomalies of any kind |

**Assembly layer: COMPLETE.** All four modalities joined by case_id into
one manifest (4,865 TCGA cases, 36.0% with all four modalities, 0.9% with
none -- explained, not a bug) and a working, live-tested PyTorch Dataset.

**Real fidelity evaluation: TWO ROUNDS RUN, HONEST MIXED RESULT (see
below).** 200-epoch baseline showed C2ST ~0.99-1.00 and PRDC precision
~0.000 everywhere -- samples not yet close to real data. Found and fixed
a real LR-schedule gap (no decay, ever) and reran at 500 epochs: FID
improved for several modalities, but precision/C2ST did NOT meaningfully
improve, and pathology recall got WORSE. Recorded honestly as a genuine,
unresolved limitation at this stage rather than re-run toward a more
flattering number. Next real levers identified: training data scale,
guidance strength, DDIM step count.

**CFG-scale sweep + a second real evaluation bug fixed (see below).**
Increasing guidance scale substantially improved JOINT fidelity (FID
16.28->6.32) but exposed that per-modality precision was stuck at
EXACTLY 0.000 for a structural reason, not a model failure: the
per-modality feature space used unsupervised PCA, which was blind to the
specific direction CFG was actually improving. Fixed with per-modality
trained classifiers instead. Real result: pathology precision reaches
0.785 (a genuinely strong finding), RNA/EHR show small real precision,
and genomic's persistent 0.000 is now explained by weak real
tissue-signal in genomic data itself (classifier barely beats chance),
not a generation failure -- a different evaluation lens is likely needed
for that one modality specifically.

**Pathology, current decision point**: UNI (original plan) requires gated
HuggingFace access, request submitted, approval pending (timeline
unknown). Decided to proceed NOW with **CTransPath** instead — open,
no gating, trained natively on TCGA+PAIP (arguably better domain match
than UNI's private cohort anyway), 27.5M params (vs UNI's 303M, faster at
our scale), 768-dim output. UNI becomes an optional secondary comparison
if/when access comes through, which is a legitimate ablation rather than
just a fallback. ABMIL aggregation (the attention mechanism) is UNTRAINED
by design at this stage -- it must be learned jointly with MoDiT later,
not run now as if it were a frozen feature extractor; the pathology
pipeline output is therefore a per-case BAG of tile embeddings (variable
length, capped), not a single fixed vector like the other three
modalities. Tile extraction (tile_slide) verified working against a real
slide: 10,906 tissue tiles found on one representative slide in 119s;
344 cores / 1TB RAM available, full 6,209-slide cohort estimated well
under 2 hours wall time with reasonable parallelism.

**CTransPathEncoder: COMPLETE and fully live-verified** (2026-09-01) --
see "CTransPath loading" below for the three real bugs found and fixed
getting here. Confirmed on real tiles from a real TCGA slide: correct
(N, 768) output shape, zero NaNs, healthy embedding statistics (pairwise
cosine similarity mean 0.694 / std 0.109 / range 0.33-0.91 across 50 real
tissue tiles -- meaningfully differentiated, not collapsed to a point and
not pure noise; tight embedding norms 3.59+-0.13, consistent with a
properly calibrated pretrained encoder).

**Not yet started**: CPTAC pathology (DICOM format, different from TCGA's
.svs, not yet touched).

**Full-cohort pathology run: COMPLETE (2026-09-01).** 2,000/2,000 cases
(100%), zero crashed, zero no_tiles. Comprehensive post-run verification,
not just the run's own self-reported success count: 2,000/2,000 unique
case_ids, 0 id/filename mismatches, 0 files with any NaN, 2,000/2,000
unique embedding-content signatures (no duplicate content anywhere -- the
cleanest result of any modality in this project), tile count per case
min=406/max=8000/mean=6062/median=6354 (sensible spread, real biological
variation, cap genuinely binding on well-populated cases), and a PERFECT
cross-reference against case_index_tcga.csv's has_slide=True list: 0
missing, 0 extra.

**ALL FOUR ENCODERS ARE NOW COMPLETE.** RNA-seq, genomic, clinical,
pathology -- every one live-verified against real data, not just built.
Next real work: donor-disjoint assembly of all four modalities' per-case
embeddings into the final training tensor train_dit.py consumes (see Open
items), CPTAC pathology (DICOM, not started), and everything downstream
of the data pipeline -- PPN training, baseline training, MoDiT training
itself, ablations. See MANUSCRIPT_NOTES.md: none of the actual paper
Results exist yet, only a verified data/encoder pipeline to build them on.

---

## Genomic pipeline: the REAL scope, found and completed 2026-09-01

Genomic's earlier "COMPLETE" status (2026-08-31 entries above) was
overstated -- what was actually verified then was ONLY
build_gene_annotation_matrix's feature-extraction logic against mocked
inputs (clinvar=set(), gnomad_constraint={}). Three real external data
sources build_gene_annotation_matrix actually needs were never built:
ClinVar P/LP gene parsing, gnomAD constraint parsing (data not even
downloaded yet), GTEx tissue-median expression parsing. A fourth resource
GenomicSetEncoder needs (a STRING/GO gene embedding table) was never
acquired at all, only mentioned in a docstring. And there was no
full-cohort orchestration script (no run_genomic_encoder.py equivalent to
the other three encoders' run_*.py scripts) -- this was caught and fixed
today, all four pieces built and verified against real data:

  1. **gnomAD v4.1 constraint** (data/gnomad_constraint.py) -- downloaded
     fresh (95 MB, direct GCS bucket URL, confirmed 200 OK before
     committing to it). File is TRANSCRIPT-level (a gene like A1BG has 3
     rows, one using a RefSeq-style non-ENSG id that must be excluded).
     pLI is column 11 (lof_hc_lc.pLI); LOEUF has no column literally
     named "loeuf" -- it is column 23 (lof.oe_ci.upper), confirmed by
     definition, not a plausible-sounding guess. One row per gene selected
     via MANE Select, falling back to canonical. Verified: TP53 pLI=0.999
     (textbook value); BRCA1 pLI~0 is NOT a bug -- a well-documented
     property of pLI (population-level viability signal, not clinical
     importance) with BRCA1's real constraint better reflected in its
     LOEUF=0.885. 17,878 genes parsed.

  2. **ClinVar P/LP genes** (data/clinvar_parser.py) -- data was already
     downloaded (Phase 1). Real CLNSIG vocabulary confirmed messier than
     assumed: compound values joined by both "|" and "/", and
     "Conflicting_classifications_of_pathogenicity" contains the literal
     substring "pathogenic" (first 10 chars of "pathogenicity") so a
     naive substring check would wrongly include it -- filter policy
     (contains "pathogenic" AND NOT "conflicting", case-insensitive)
     verified against every real pattern found in the file before
     trusting it. GENEINFO can list multiple overlapping genes per
     variant ("|"-joined) -- all are counted, not just the first.
     Verified: TP53/BRCA1/BRCA2 all correctly found. 345,858 P/LP variants
     across 7,982 genes, 13.8s runtime.

  3. **GTEx tissue-median expression** (data/gtex_expression.py) -- data
     already downloaded (Phase 1). Confirmed real GCT format (version +
     dims header lines before the real header), confirmed SAMPID in the
     sample-attributes file matches GCT column headers directly (no id
     transform needed), confirmed SMTS (general tissue, e.g. "Blood") is
     the right granularity vs. SMTSD's much finer subdivisions, given
     TCGA's own coarse cancer-type tissue labels. 1,608 duplicate gene
     symbols in GTEx (real paralog/naming collisions) collapsed via max,
     not silently keeping one arbitrarily. Verified: ALB (albumin) shows
     25,200 TPM in Liver vs 1.4 in Blood; INS (insulin) shows 2,325.5 TPM
     in Pancreas vs 0.08 in Lung -- both textbook tissue-specificity
     positive controls, correct by orders of magnitude. 30 tissue types,
     1.64M (gene,tissue) entries, ~500s runtime (the slow one, dominated
     by the large TPM matrix).

  4. **Gene embedding table** (data/gene_vocabulary.py) -- HONEST
     SIMPLIFICATION, not hidden: GenomicSetEncoder was designed for a real
     STRING/GO knowledge-graph gene embedding that was never acquired (no
     source was ever even identified). Using a fixed-seed (reproducible)
     random unit-norm embedding instead, built over gnomAD's own
     17,878-gene vocabulary. The set-transformer's real information
     content comes from the 14 real annotation features, not this
     identity vector -- stated plainly here and in MANUSCRIPT_NOTES.md as
     a limitation to note in Methods, and is a straightforward swap for a
     real embedding later (same shape, same gene_index contract) without
     touching anything else.

  5. **run_genomic_encoder.py** -- the missing full-cohort orchestration,
     TCGA only (CPTAC deferred: its GDC "project" field is not
     tissue-specific like TCGA's, needs its own tissue-mapping logic not
     yet built, consistent with CPTAC pathology also being deferred).
     TCGA project code -> GTEx SMTS tissue name mapping (8 projects)
     confirmed against GTEx's real 30-tissue vocabulary before hardcoding
     -- all 8 present under simple expected names, no surprises. Storage:
     ONE combined .npz (not per-case like pathology), since
     build_gene_annotation_matrix always returns a FIXED (top_k, 14)
     shape (zero-padded), unlike pathology's genuinely variable tile
     bags -- a stacked array matches the RNA/clinical pattern and is
     sufficient. Path-resolution and project-filtering logic verified
     offline (including a case from an unmapped project correctly
     skipped, not crashed) before live testing.

     Live smoke test (20 real cases, all four annotation sources +
     external resources loaded together for the first time): 20/20
     succeeded, 0 parse failures, 0 NaN anywhere, gene indices correctly
     within vocabulary bounds, SIFT/PolyPhen direction-sensitive
     aggregation still correct on real aggregated data (both properly
     bounded to their real [0,1] ranges), n_valid_genes sensible
     (min=20, max=512 exactly -- one case hit the top_k cap, plausible
     for a high-mutation-burden tumor, mean=190).

  4,070 TCGA cases have a real MAF on disk and a mapped project --
  full-cohort run COMPLETE (2026-09-01): 4,070/4,070, 0 parse failures, 0
  NaN anywhere, perfect cross-reference against case_index_tcga.csv (0
  missing, 0 extra). n_valid_genes: min=0/max=512/mean=109/median=60,
  262 cases (6.4%) hit the top_k=512 cap.

  30 cases (0.74%) show n_valid_genes=0 -- investigated rather than
  assumed benign: located and inspected one real case's actual MAF file
  on disk (TCGA-06-0178), confirmed it has a header row and ZERO variant
  rows -- a genuinely empty MAF (known, if uncommon, real property of a
  small fraction of TCGA cases: low tumor purity or aggressive somatic
  calling filters can leave a case with no confident variant calls at
  all). The pipeline correctly produced a properly-shaped, all-zero-padded
  feature matrix for this case rather than crashing or fabricating data --
  correct behavior for genuinely empty input, not a bug.

**GENOMIC ENCODER FULLY COMPLETE, full TCGA cohort, live-verified at
every stage.**

---

## Assembly layer: COMPLETE (2026-09-01)

Two real fixes made before assembly, caught while thinking through exactly
how genomic/pathology's raw material would need to flow into MoDiT:

  - **ModalitySpec's "rna" dim was wrong.** The original spec (written
    before any real encoder existed) assumed 512; Geneformer's real,
    confirmed output is 768 (verified: every rna_*_full.npz shows shape
    (N, 768)). Fixed in model/dit.py.
  - **GatedABMIL's default d_in was wrong.** Defaulted to 1024 (assumed
    UNI, which was never actually used given the gated-access blocker).
    CTransPath's real confirmed tile-embedding output is 768. Fixed in
    encoders/foundation.py; d_out stays 1024 (ABMIL's own aggregation
    width, independent of input tile width). If UNI access is ever
    granted, instantiate with d_in=1024 explicitly for that comparison.

**build_case_manifest.py** joins all four modalities' REAL output (not
case_index.csv's predictions -- actual case_ids present in each
combined .npz / pathology directory) into one manifest, TCGA only
(CPTAC/Pfib_423 remain RNA-only external validation, not jointly
assembled -- their genomic/clinical/pathology were never built). Verified
against synthetic data with deliberately mismatched per-modality coverage
before running for real.

Real result on the full cohort: **4,865 cases**, has_rna 87.5%, has_clinical
97.8%, has_genomic 83.7%, has_pathology 41.1% (matches each encoder's own
independently-measured coverage exactly, cross-validating all of them
against each other). Modality-overlap distribution: 36.0% of cases have
ALL FOUR (1,753 cases), 45.7% have three, 11.4% have two, 5.9% have one,
0.9% (46 cases) have none. Investigated the zero-modality cases rather
than assuming: all 46 share one barcode pattern (TCGA-17-Z0**, all LUAD)
-- a coherent single-batch explanation (a submission series with no
molecular data of any kind ever deposited for open access), not a bug.

**dataset.py** (RareSynthTCGADataset) reads the manifest and serves real
per-case training examples: RNA/clinical returned as their real
precomputed fixed vectors; genomic/pathology returned as RAW material
(gene features + gene indices; a padded/masked tile bag) since
GenomicSetEncoder/GatedABMIL are untrained and must run as part of the
model's forward pass at train time, not as frozen preprocessing.
Radiology always zero-filled, always marked unavailable (explicit
placeholder -- no radiology encoder exists anywhere in this project yet).
Pathology bags are lazily loaded per-case (not preloaded -- ~2000 files up
to 8000 tiles each would be tens of GB in memory otherwise) and
subsampled/padded to a fixed max_tiles for batching; BOTH directions
(subsample when over the cap, pad+mask when under) verified separately
against synthetic data, including at the real observed minimum tile count
(406) to make sure padding doesn't silently corrupt or misalign the bag
against its mask.

Live end-to-end test: real DataLoader, batch_size=8, num_workers=2 (
exercises concurrent lazy pathology loading specifically, not just
single-threaded correctness) against the real 3,836-case train split.
Correct shapes on every field, genuine per-case availability heterogeneity
in the batch (not artificially uniform), zero NaN anywhere.

**NOT yet done, and this is the real next step, not administrative
bookkeeping: nothing trains on this data yet.** The Dataset correctly
PRODUCES raw per-case material; train_dit.py (or a rewrite of it) still
needs to: instantiate GenomicSetEncoder and GatedABMIL as trainable
modules with their parameters in the optimizer, run them inline each
forward pass on the raw geno/path material this Dataset returns to get
z_geno/z_path, concatenate with the real precomputed z_rna/z_ehr and the
always-zero z_rad into the full modality vector, then run MoDiT's
diffusion process on that. None of this training-loop wiring exists yet.

---

## MoDiT joint training: FIRST REAL RUN, WORKING (2026-09-01)

train_modit.py built, supersedes the original train_dit.py (written
speculatively before any real Dataset/encoder existed, wrong flat-tensor
assumption for geno/path).

Two real fixes found and fixed BEFORE the first live run, both confirmed
by direct experiment, not reasoned about in the abstract:

  1. **GatedABMIL masking-convention NaN risk, SEVERE.** GatedABMIL's
     mask parameter is True=hide (standard masked_fill convention) --
     the OPPOSITE of the Dataset's path_mask (True=real tile). A case
     with NO real pathology has path_mask entirely False, so the naive
     inverse is entirely True -- every attention position masked to
     -inf, softmax over an all -inf row is NaN. Given only ~41% of TCGA
     cases have real pathology, MOST training batches would have hit
     this and silently corrupted the loss via NaN propagation --
     confirmed live: an unfixed all-padding case's output genuinely is
     NaN. Fixed by forcing position 0 of the attention mask to always
     stay unmasked (confirmed live: eliminates the NaN entirely), then
     multiplying the resulting z_path by the real avail flag so the
     resulting fake-but-non-NaN output never leaks into training.
     GenomicSetEncoder was checked for the equivalent risk and confirmed
     SAFE without needing the same fix (it is called without an explicit
     pad_mask given the current Dataset design, so there is no internal
     masked-softmax step that could produce NaN) -- still zeroed via
     avail afterward for consistency.

  2. **Zero-gradient false alarm, fully resolved, not a bug.** The first
     end-to-end dry run showed gradient into BOTH GenomicSetEncoder and
     GatedABMIL exactly zero after one training step, which looked like
     a serious disconnected-graph bug. Traced systematically (spec.join
     confirmed clean via an isolated leaf-tensor test; the encoders'
     forward pass + avail-masking confirmed clean via an isolated
     pre-MoDiT test; the actual break isolated to MoDiT.forward() itself
     via a direct x.grad inspection) to the real, correct explanation:
     MoDiT's out_proj layers are DELIBERATELY zero-initialized (a
     standard diffusion-transformer stability technique), and a
     zero-weight Linear layer's gradient w.r.t. ITS OWN WEIGHTS is
     nonzero (so it does learn) while its gradient w.r.t. ITS INPUT is
     exactly zero AT THAT EXACT INITIALIZATION (since d(output)/d(input)
     = weight = 0). This is a one-step-only artifact, not a permanent
     structural bug -- confirmed live over 3 steps: out_proj's own
     weight+bias magnitude is exactly 0.0 before training, gradient into
     the model's input is exactly 0.0 at step 0, then genuinely nonzero
     (18,285 then 58,575 in the isolated test) at steps 1 and 2 once
     out_proj's own weights move away from zero. Re-ran the full dry run
     over multiple steps to confirm both GenomicSetEncoder and GatedABMIL
     correctly receive nonzero gradient from step 1 onward. A single-step
     gradient check would have wrongly condemned a CORRECT design here --
     worth remembering for any future debugging of a freshly-initialized
     zero-init architecture.

**First real live training run** (small model for speed: d_model=128,
depth=4, 9.2M total trainable params across MoDiT+GenomicSetEncoder+
GatedABMIL; 3 epochs, 239 real batches/epoch, full real 3,836-case train
split, max_tiles=500): loss 0.990 -> 0.748 -> 0.667, a genuine, monotonic,
~33% reduction over 3 epochs, zero NaN, ~13s/epoch. Checkpoint verified:
correctly saves all three trainable components' state dicts (79 MoDiT
tensors, 99 GenomicSetEncoder, 8 GatedABMIL) plus EMA shadow and the full
run config for reproducibility.

**Full-size run, real architecture, COMPLETE (2026-09-01)**: d_model=512,
depth=12, 73.6M total trainable params (MoDiT 67.7M + GenomicSetEncoder
4.6M + GatedABMIL 1.2M), 200 epochs, max_tiles=2000 (the real target
config, not the smoke test's reduced size), ~17s/epoch, ~53 minutes total
(measured, not estimated -- an initial extrapolation from the smoke
test's smaller config was deliberately NOT trusted given four
simultaneously-changing scale factors; a short real calibration run at
the target config was used instead, then this full run's own timing
confirmed it directly). Loss collapsed fast (0.990 -> ~0.01 by epoch
10-15) then plateaued around 0.005-0.008 for the remaining ~185 epochs
with periodic spikes (roughly every 20-25 epochs, most likely from the
30% random modality dropout injecting real per-epoch difficulty
variance, not confirmed as a bug) -- final loss 0.00643.

**Train-vs-validation check, the real test of memorization vs genuine
generalization** (eval_train_val_gap.py, new): the fast loss collapse and
73.6M params against only 3,836 training cases (~19,000 params/example)
was flagged as a real, plausible memorization risk BEFORE checking, not
dismissed. Evaluated both splits at the SAME fixed diffusion timestep
(controls for cross-split timestep-difficulty variance) with modality
dropout disabled (real avail mask only, for a fair comparison) on
479 real held-out validation cases the model never saw during training.
Result: **train loss 0.00396, val loss 0.00400** -- essentially
identical (~1% relative difference), val marginally higher as expected
for genuine generalization, NOT the large gap memorization would produce.
This is a real, clean, well-controlled piece of evidence that the model
learned a genuinely generalizable denoising function at this scale, not
just memorized its training examples -- worth a sentence in the paper's
implementation-verification framing (see MANUSCRIPT_NOTES.md), still not
a Results-table claim (this is loss on the training objective, not any
of the paper's actual evaluation metrics -- FID/CMCS/mechanism retrieval
etc. have not been computed on this or any model yet).

**This is the first real, working, end-to-end confirmation of the entire
pipeline built across this whole project** -- data acquisition through
joint multimodal training, genuinely producing a learning, genuinely
generalizing model on real TCGA data. Scale-up is done; baseline
comparisons, ablations, and the paper's actual evaluation metrics are the
next phase, not yet done.

---

## Real bug #12: RNA embeddings never normalized -- a genuine
## cross-modal scale mismatch, found while checking sampled-output quality

sample_modit.py built and verified offline; first real sampling run
against the trained checkpoint (500 unconditional samples) succeeded with
no NaN/Inf, but a closer check of the VALUE DISTRIBUTION (not just
absence-of-NaN) found a real problem: 99.80% of generated samples had at
least one coordinate pinned exactly at the x0_clip=+/-4.0 safety
boundary, and the 1st/99th percentiles of the entire flattened output sat
exactly at that boundary too -- the clip was actively truncating a
meaningful part of the generated distribution's tails, not rarely
catching genuine outliers.

Root cause, confirmed by directly measuring real training data (not
assumed): every custom-written encoder in this project (ClinicalEncoder,
GenomicSetEncoder, GatedABMIL) explicitly L2-normalizes its own output to
unit norm before returning -- but the RNA pipeline (run_rna_encoder.py)
uses Geneformer's own EmbExtractor.extract_embs() directly and saves its
output AS-IS, with no normalization step ever added, because it came
from a third-party library's own return convention rather than our own
encoder class where we controlled the return statement. Measured
directly: real clinical embeddings have mean row norm EXACTLY 1.00000 (as
designed); real RNA embeddings have mean row norm 6.58, with a real
per-coordinate max absolute value of 4.15 -- already exceeding the
diffusion process's x0_clip=4.0 bound in genuine TRAINING data, before
any generation was even involved.

This is a genuine architectural inconsistency, not cosmetic: with RNA at
~6.6x the scale of the other modalities, the SAME shared diffusion
process (same noise schedule, same timestep, same denoising network)
implicitly assumes comparable per-dimension scale across all five
modalities concatenated into x -- RNA's much larger natural magnitude
would dominate the combined representation's effective signal-to-noise
ratio at any given timestep, directly undermining the cross-modal
coherence the whole architecture is built around.

Fixed in RareSynthTCGADataset (data/dataset.py): RNA embeddings are now
L2-normalized to unit norm at load time, matching the convention every
other modality already follows. Verified against synthetic data
replicating the exact real scale imbalance (RNA norm ~85, clinical norm
exactly 1.0) before deploying: fix correctly brings RNA to unit norm,
leaves clinical unaffected, and preserves each row's DIRECTION exactly
(cosine similarity >0.9999 between raw and normalized RNA vectors) --
only the magnitude was corrected, not the actual encoded information.

Because the fix lives inside the Dataset class itself, every downstream
script (train_modit.py, sample_modit.py, eval_train_val_gap.py)
automatically inherits it with no separate code changes needed.

**The existing trained checkpoint (modit_full) was trained on the
UNNORMALIZED RNA data and is now stale** -- moved to
runs/modit_full_PRENORM_STALE for reference, not deleted, but should not
be used for any further evaluation. A fresh full training run with the
corrected data is the immediate next step, in progress as of this
writing. All downstream evaluation (fidelity metrics, baselines,
ablations) should wait for the corrected checkpoint rather than build on
the stale one.

---

## Real bug #13: absolute data scale mismatch -- the RNA fix (bug #12)
## was necessary but NOT sufficient, found by re-checking rather than
## assuming the first fix had resolved the problem

Retrained with the RNA-normalization fix (bug #12) and re-ran the SAME
sampling-quality diagnostic that caught bug #12 in the first place --
deliberately did not assume the fix had worked just because nothing
crashed. Result: the clipping problem was essentially UNCHANGED (2.29%
vs 2.15% of all values at the boundary, 98.4% vs 99.8% of samples
affected) -- confirming the RNA fix, while itself real and correct (every
custom encoder's own convention IS now consistently unit-norm), was not
the full explanation.

Root cause, confirmed by DIRECTLY MEASURING real training data's
concatenated-x statistics (not theorized): overall real x has std
0.02995 -- but the generated samples (post RNA-fix-only) had std 1.4524,
a ~48x mismatch. This is a SEPARATE, ABSOLUTE-scale problem, distinct
from the cross-modal RELATIVE imbalance bug #12 fixed: every modality's
encoder L2-normalizes to unit NORM per vector, which for a
several-hundred-to-thousand-dimensional vector gives a per-coordinate
std on the order of 1/sqrt(d) -- genuinely tiny (~0.02-0.04) -- while
standard diffusion model formulations (DDPM/DDIM, including this
project's own GaussianDiffusion) implicitly assume roughly UNIT-VARIANCE
data, since the forward noise process is designed to smoothly interpolate
between data-scale and standard-normal-noise-scale. With real data ~33x
smaller in scale than the diffusion process's implicit assumption, the
reverse (sampling) process had to learn a large, poorly-calibrated
compression from noise-scale down to true data-scale, overshooting and
hitting the x0_clip=4.0 safety bound repeatedly.

Fixed with compute_data_scale() (train_modit.py, new): measures the REAL
training data's actual std directly (not a guessed or literature-typical
constant) from several real batches at the start of training, computes
scale=1/real_std, and multiplies x by this factor before every diffusion
step during training. sample_modit.py divides generated samples by the
SAME factor to return them to the real embedding space. The factor is
saved into the checkpoint itself (data_scale key) rather than
independently hardcoded in multiple files -- eval_train_val_gap.py and
sample_modit.py both read it back, with a backward-compatible default of
1.0 for the two now-stale checkpoints that predate this fix (matching
what those runs actually did, rather than silently applying a rescaling
they were never trained with).

Verified the scale-computation mechanism precisely before deploying:
synthetic data with a known target std, confirmed computed_scale *
measured_std == 1.0 exactly (not approximately).

**Real result after retraining with BOTH fixes (bug #12 + bug #13)**:
train/val generalization re-confirmed on the corrected checkpoint (train
0.01017, val 0.01011 -- still essentially identical, no memorization).
data_scale computed automatically as 33.401, matching the earlier direct
measurement (1/0.02995 ~= 33.4) almost exactly -- a strong independent
consistency check that the automatic computation is working correctly,
not coincidentally close. Generated samples: **0.0000% of values at the
clip boundary, 0.00% of samples affected** (down from 2.29%/98.4%) --
overall std 0.0506 vs real data's 0.02995, a ~1.7x difference, a
completely normal and expected level of generation variance rather than
the previous 48x systematic distortion. The min/max (+/-0.1198) trace
back exactly to the x0_clip=4.0 boundary divided by data_scale=33.401,
confirming the clip is now only an occasional safety net on rare true
outliers among 1.79 million total generated values, not a pervasive
distribution-wide truncation.

**Two real, substantial, necessary-and-jointly-sufficient scale
corrections (bugs #12 and #13) were required before generated samples
were genuinely trustworthy** -- caught specifically because sampled
OUTPUT QUALITY was checked directly (value distributions, clipping
frequency, comparison against real data statistics) rather than settling
for "no NaN/crash" as sufficient evidence of correctness. This is exactly
the standard of scrutiny the eventual paper's evaluation needs to survive
review, and it is worth stating plainly in Methods that this verification
was done (see MANUSCRIPT_NOTES.md).

**Third checkpoint (runs/modit_full) is the first one considered
trustworthy for downstream evaluation.** The two earlier ones
(modit_full_PRENORM_STALE, and the RNA-fix-only intermediate run) remain
on disk for reference but should never be cited or built upon.

---

## Real fidelity evaluation, first ever run against a trained model (2026-09-02)

eval/fidelity.py's per_modality_report had a real, previously-unnoticed
gap between its own docstring and its actual code: the docstring says
FID is computed "in the penultimate feature space of an independently
trained tissue-of-origin classifier," but the code called
frechet_distance() directly on RAW, full-dimensional embeddings (up to
3584-dim for the joint case). Confirmed this is a severe problem before
building anything further, not assumed: two samples drawn from the
IDENTICAL distribution (n=500, d=3584) produced FID=4939 (should be ~0),
with the real covariance matrix's rank capped at n-1=499 instead of the
full 3584 dimensions -- numerically meaningless at this
dimensionality/sample-size ratio.

Built eval/tissue_classifier.py (a real MLP trained on real tissue
labels, penultimate layer width 64, chosen so n>>d for well-conditioned
covariance estimation) and eval_fidelity_real.py (the full real
evaluation pipeline: joint uses the classifier's penultimate features;
each individual modality slice uses PCA fit on real data -- simpler than
training 5 more classifiers, and does not assume every modality carries
equally strong tissue signal; rad is EXCLUDED entirely from the report,
since real radiology is always exactly zero by construction and there is
no genuine distribution to compare against). Verified the fix works
before trusting it: same-distribution FID with class-discriminative
synthetic data dropped from a raw-space 191.68 to a feature-space 4.87 (a
39x reduction), confirming the small-n-large-p problem is resolved.

**Real result, first trustworthy fidelity numbers this project has ever
produced** (trained tissue classifier: 100% real validation accuracy --
plausible, not suspicious, given how well-separated tissue-of-origin is
in real multi-omic cancer data; 479 real held-out val cases vs 479
generated samples):

| modality | FID | precision | recall | C2ST accuracy |
|---|---|---|---|---|
| joint | 18.67 | 0.159 | 0.906 | 0.991 |
| geno | 0.75 | 0.000 | 0.184 | 0.999 |
| rna | 1.05 | 0.000 | 0.138 | 1.000 |
| path | 1.18 | 0.000 | 0.610 | 1.000 |
| ehr | 2.34 | 0.000 | 0.013 | 1.000 |

**Honest reading**: a real classifier distinguishes real from generated
samples almost perfectly (C2ST ~0.99-1.00) everywhere, and PRDC precision
near 0.000 across every modality confirms generated samples mostly do NOT
land in real data's local density regions -- the model does not yet
closely match the real distribution. Recall is more mixed (joint 0.906
vs much lower per-modality), a mode-covering-but-imprecise pattern
(generated output spans a broad enough region to contain real data but is
diffuse rather than sharp). This is a genuine, informative limitation
finding, not a bug -- a first full training run of a novel 5-modality
architecture on ~3,800 real cases producing samples that do not yet
tightly match real data is a completely plausible, unremarkable outcome,
not evidence of a defect. Recorded honestly rather than hidden or
oversold; see MANUSCRIPT_NOTES.md for how this should be framed if it
persists into the paper.

---

## Real bug/gap #14: no learning-rate decay schedule, likely
## contributing to the plateau above

Checked before just running more epochs on top of the same schedule:
train_modit.py's LR schedule warms up linearly then holds the PEAK LR
CONSTANT for the rest of training -- no decay at all, confirmed by
reading the actual code (`args.lr * min(1.0, (step+1)/warmup)`, which
caps at exactly 1.0 and never decreases). This is a real gap against
standard practice (virtually every published diffusion transformer uses
a post-warmup decay schedule, precisely because a constant LR prevents
fine-grained late-training convergence), and plausibly explains why the
completed run's loss oscillated in a narrow band (0.026-0.031) for its
final ~150 epochs rather than continuing to settle, rather than having
genuinely converged.

Fixed with get_lr() (train_modit.py, new): linear warmup unchanged, then
COSINE decay from peak lr down to a configurable floor (default 2% of
peak) by the final training step. Verified numerically before deploying:
confirmed correct values at start (near zero), end of warmup (at peak),
midpoint (strictly between floor and peak), and final step (exactly at
the floor).

Deliberately the ONLY change made before the next training run (not
bundled with other hyperparameter changes) so any improvement in the
fidelity numbers above can be attributed to it specifically, not
confounded with several simultaneous changes -- a real methodological
discipline worth maintaining given how much this project's results will
need to withstand scrutiny.

**Next training run**: 500 epochs (up from 200), same architecture, this
LR schedule fix, in progress as of this writing. Re-run
eval_fidelity_real.py against the result and compare directly against
the table above once complete.

---

## 500-epoch run results: mixed, honestly reported, no clean win (2026-09-02)

Generalization re-confirmed first (train 0.00420, val 0.00422 -- still
essentially identical, no memorization at the longer duration either).

Real fidelity comparison against the 200-epoch (no-decay) baseline:

| modality | FID (200ep -> 500ep) | precision | recall (200ep -> 500ep) | C2ST (200ep -> 500ep) |
|---|---|---|---|---|
| joint | 18.67 -> 16.28 | 0.159 -> 0.136 (worse) | 0.906 -> 1.000 | 0.991 -> 0.983 |
| geno | 0.75 -> 0.73 | 0.000 -> 0.000 | 0.184 -> 0.184 (identical) | 0.999 -> 1.000 |
| rna | 1.05 -> 0.73 | 0.000 -> 0.000 | 0.138 -> 0.138 (identical) | 1.000 -> 0.999 |
| path | 1.18 -> 0.34 | 0.000 -> 0.000 | 0.610 -> 0.000 (worse) | 1.000 -> 0.998 |
| ehr | 2.34 -> 1.03 | 0.000 -> 0.000 | 0.013 -> 0.013 (identical) | 1.000 -> 1.000 |

**Honest reading, not spun toward a positive framing**: FID genuinely
improved for RNA, pathology, EHR, and modestly for joint -- real
distributional-distance progress consistent with the LR fix helping
convergence. But PRECISION remains exactly 0.000 across every modality
in BOTH runs, and C2ST accuracy remains 0.98-1.00 everywhere in both
runs -- the core sample-level fidelity problem (a real classifier can
still nearly perfectly distinguish real from generated samples) did NOT
resolve. Pathology's recall specifically got WORSE (0.610 -> 0.000), a
real regression, not noise favoring the fix.

**Flagged, not explained away**: geno/rna/ehr recall values are exactly
identical between the two runs to three decimal places (0.184, 0.138,
0.013) -- an unusual amount of coincidence for two genuinely different
checkpoints trained with different schedules. No root cause identified
yet. Does not change the overall honest conclusion (precision/C2ST both
independently say samples do not yet closely match real data either
way), but should be investigated -- possibly a PRDC recall
implementation detail that is less sensitive to this particular kind of
model change than expected, possibly something else -- before any of
these numbers are treated as final for the paper.

**Conclusion for this round**: the LR-decay fix was a real, isolated,
worthwhile change (distributional shape improved) but is not sufficient
on its own to close the sample-fidelity gap. Likely next levers, in
rough order of expected impact: training data scale (3,836 real cases is
small for a 67.7M-parameter MoDiT backbone -- CPTAC and Pfib_423 remain
RNA-only and are not part of this joint training set at all, see Open
items), classifier-free guidance strength at sampling time (not yet
tuned, currently using whatever default sample_modit.py's ddim_sample
applies), and DDIM step count (currently 200, worth checking whether
more steps meaningfully changes precision specifically). This is a
genuine, reportable limitation at this stage of the project, not a
failure of the underlying approach -- recorded honestly rather than
re-run repeatedly until a more flattering number appears.

---

## Real bug #15: PCA-based per-modality feature space was structurally
## blind to real, measured improvements -- found via a CFG-scale sweep

Per Pijush's direction ("we cannot have precision 0, solve that first"),
investigated the CFG scale as the first lever (cheap, no retraining
needed). Swept cfg_scale in {2, 7, 12, 20, 35} using a proxy diagnostic
(does an independent classifier agree a sample generated "as tissue X"
actually looks like tissue X): 0.360 -> 0.674 -> 0.750 -> 0.806 -> 0.836,
a real, substantial, monotonic improvement with clear diminishing
returns by cfg=35. Ran the full fidelity evaluation at cfg=35: JOINT FID
improved substantially (16.28 -> 6.32) and JOINT precision improved
(0.136 -> 0.163), confirming CFG scale was a genuine, real lever -- but
every INDIVIDUAL modality's precision remained EXACTLY 0.000, unchanged
from cfg=2.

This exposed a second, more serious problem: geno/rna/ehr recall came
back BIT-IDENTICAL to three decimal places across THREE separate sample
sets (200-epoch, 500-epoch cfg=2, 500-epoch cfg=35) -- not investigated
away as coincidence this time, chased down directly. Confirmed via a
direct measurement that the RAW GENERATED VALUES for these modalities
genuinely change substantially between cfg=2 and cfg=35 (RNA and EHR
showed the LARGEST relative change of any modality, ratio 0.9-1.4x their
own signal scale) -- ruling out "CFG doesn't affect these modalities" as
an explanation. Root cause: the per-modality feature space used PCA
(fit on real data), which is UNSUPERVISED and finds directions of
maximum variance with no awareness of tissue identity at all. Whatever
direction CFG was genuinely improving apparently fell outside the
handful of top-variance components PCA kept, making the metric
structurally blind to real, measured improvement -- not because nothing
was changing, but because PCA wasn't looking in the right place.

Fixed in eval_fidelity_real.py: a SEPARATE tissue classifier is now
trained per modality (not just for "joint"), so every modality is
evaluated in a feature space that is actually organized around what the
metric is trying to measure. Verified offline first with a positive
control (synthetic data with genuine per-modality tissue signal and fake
data close to real): precision/recall both correctly reached 1.000
across every modality, confirming the classifier-based approach CAN
detect real improvement where PCA could not.

**Real result on the real checkpoint (cfg=35 samples), a materially
different and more informative picture than PCA ever showed**:

| modality | classifier val acc | FID | precision | recall |
|---|---|---|---|---|
| joint | 1.000 | 6.31 | 0.163 | 0.747 |
| geno | 0.246 (barely above chance) | 31.84 | 0.000 | 1.000 |
| rna | 0.929 | 21.38 | 0.090 | 0.741 |
| path | 0.480 | 12.86 | **0.785** | 0.935 |
| ehr | 0.996 | 16.01 | 0.025 | 0.313 |

Pathology precision (0.785) is a real, substantial, positive result --
genuine signal PCA had been hiding. RNA and EHR now show small but
genuinely nonzero precision (0.090, 0.025), a meaningful change from
PCA's suspicious flat zero everywhere. Genomic remains at exactly 0.000,
but this is now explainable rather than mysterious: its classifier
barely beats chance (0.246 vs 0.125), meaning tissue-of-origin is
genuinely NOT a strong organizing signal in real somatic mutation
profiles to begin with -- biologically consistent with cancer driver
genes (TP53 etc.) mutating across many tissue types rather than being
tissue-specific the way expression is. Testing "does genomic fidelity
match real data's tissue clusters" may simply be the wrong lens for this
modality, independent of generator quality -- a different evaluation
axis (e.g. gene-annotation-feature distributional match, or known
driver-gene co-occurrence patterns) is likely more appropriate for
genomic specifically, a design question for MANUSCRIPT_NOTES.md rather
than something further guidance-scale tuning will fix.

---

## Thread oversubscription — parallel pathology launch, severe, caught before real damage

First launch of the 8-worker parallel pathology pipeline showed processes
consuming enormous CPU time (7 to 66 CPU-minutes accumulated in about 1
minute of wall-clock time per worker) while producing ZERO completed
cases. Root cause confirmed directly: `ps -eLf` showed a SINGLE worker
process had spawned 429 OS threads, and `uptime` showed a load average of
755 on a 344-core machine -- more than double the physical core count,
from the FIRST minute of an 8-hour run that would only have gotten worse
as encoding ramped up.

Cause: OpenMP/MKL/OpenBLAS/NumExpr (used internally by NumPy, and
critically OpenCV) default to spawning one thread PER SYSTEM CORE, PER
PROCESS, unless explicitly told otherwise. With 8 worker processes each
independently trying to claim all 344 cores for internal
BLAS/image-processing operations, they were not getting 8x throughput --
they were mostly contending with each other for the same physical cores,
while --n-shards was already providing the intended process-level
parallelism. This is a general risk for ANY future parallel launch on
this server, not specific to pathology -- worth remembering before
launching multi-worker jobs for any other stage of this project.

Fixed with two layers of defense, not just one:
  1. scripts/run_pathology_parallel.sh now computes a per-worker thread
     budget (system cores / n_workers, capped at 8 -- BLAS-style
     operations show negligible benefit and rising overhead well past 8
     threads for tile-sized data) and sets OMP_NUM_THREADS/MKL_NUM_THREADS/
     OPENBLAS_NUM_THREADS/NUMEXPR_NUM_THREADS accordingly for each worker.
  2. run_pathology_encoder.py ALSO explicitly calls cv2.setNumThreads()
     and torch.set_num_threads() directly in Python -- necessary because
     OpenCV maintains its own internal thread pool that does not reliably
     respect the environment variables alone, confirmed as a real,
     independent risk rather than assumed redundant with layer 1.

Real result after the fix, on the same server, same workload: load average
dropped from 755 to ~11 (settling further as the prior spike's 5/15-minute
averages drained), and genuine case throughput appeared immediately (16
real cases completed in the first 3 minutes, none in the entire failed
first attempt).

Also fixed in the same session, found while building this: the launcher's
stdout was fully buffered when redirected to a log file (Python's default
for non-TTY output), making an unattended run look completely silent even
while genuinely working -- fixed with `python -u`. And a real risk in the
sharding design itself: every worker was originally going to write to the
same `_manifest.json` filename and silently clobber each other's
diagnostic records -- fixed with a shard-specific filename
(`_manifest_shard<i>.json`) before this was ever run for real.

---

## CTransPath loading — three real bugs, all confirmed by live errors

Getting from "architecture and preprocessing confirmed from the authors'
own official code" to "actually loads and runs correctly" took three
separate, real mismatches, each surfaced by an actual error message on the
live server, never by assumption:

1. **Community HF mirror's automatic config silently built the wrong
   architecture.** `timm.create_model("hf-hub:1aurent/...", pretrained=True)`
   was supposed to apply CTransPath's custom convolutional patch-embedding
   stem (ConvStem) automatically, per the mirror's stated design -- it did
   not; timm silently fell back to a standard single-conv patch embedding,
   caught by a state_dict key mismatch (`patch_embed.proj.weight/bias`
   missing, `patch_embed.proj.0/1/3/4/6...` unexpected -- the checkpoint's
   real ConvStem keys). Fixed by building the model EXPLICITLY with
   `embed_layer=_ConvStem` and loading raw weights directly via
   `huggingface_hub.hf_hub_download` + `load_state_dict(strict=True)`,
   bypassing the mirror's broken automatic config entirely. The exact
   ConvStem structure was cross-verified two independent ways (a community
   README snippet, and the live error's own key names) before being coded.

2. **ConvStem output format convention mismatch.** Initially implemented
   as a flattened (B, N, C) sequence output (an older timm PatchEmbed
   convention, matching the community snippet). A live forward pass
   through the actual installed timm SwinTransformer failed inside its own
   block code expecting spatial (B, H, W, C) format instead -- this timm
   version's Swin implementation operates on spatial grids internally, a
   real version-dependent difference from what the snippet assumed. Fixed
   by permuting to channels-last spatial format; verified with a full
   untrained-weights forward pass producing the correct (B, 768) output
   before touching real weights at all.

3. **Downsample (patch-merging) stage-index convention shift.** The
   checkpoint attaches each stage's downsample module to the END of that
   stage (layers 0,1,2 have one; layer 3, the last stage, does not) --
   this installed timm version attaches it to the START of the stage
   instead (layers 1,2,3 have one; layer 0 does not). Same 3 modules,
   shifted by one stage index -- confirmed precisely by comparing the real
   checkpoint's reported shapes (768, 1536) against the real target
   model's own state_dict shapes at each layer index before writing the
   fix, not guessed. Also found and fixed alongside this: stale
   relative_position_index/attn_mask keys in the checkpoint that are
   non-persistent buffers in this timm version (recomputed fresh at
   construction time) and must be dropped, not loaded, even where a shape
   happens to match. `_remap_ctranspath_state_dict()` handles both; verified
   against a synthetic checkpoint built to replicate the exact real
   mismatch pattern, loaded against the REAL target model with
   `strict=True`, with an explicit check (correcting an aliasing mistake
   in the first version of that same test -- `model.state_dict()` returns
   live references, not copies, so a "before" snapshot must be cloned to
   mean anything) that weights genuinely changed value after loading.

A fourth, much simpler bug (transform ordering -- `Resize` was placed
before converting the raw numpy tile array to a PIL Image, which it
cannot accept directly) was caught and fixed in the same debugging
session, verified against a synthetic array in tile_slide()'s exact
output format before redeploying.

---

## Data inventory

| Phase | Contents | Location | Size | Status |
|---|---|---|---|---|
| 1 | ClinVar, HPO, Orphadata | `data/raw/priors/` | ~330 MB | done |
| 1 | GTEx v8 RNA-seq + annotations | `data/raw/gtex/` | 1.6 GB | done |
| 1 | Kremer et al. (superseded, see Pfib_423) | `data/raw/kremer/` | ~200 MB | superseded |
| 2/3 | TCGA RNA-seq, MAF, clinical | `data/raw/tcga/` | 20 GB, 14,338 files | done |
| 5 | TCGA slide subset, 2,000 cases stratified | `data/raw/tcga/slides/` | 2.6 TB, 6,209 slides | done |
| 6 | TCIA radiology matched to TCGA | `data/raw/tcia/` | 92 GB, 217,861 files, 2,907 series | done |
| 7 | CPTAC RNA/MAF/clinical (ext. val. 1) | `data/raw/cptac/` | 21 GB, 5,834 files | done |
| 7 | CPTAC imaging (5 collections, DICOM) | `data/raw/cptac_tcia/` | 225 GB, 5,713 series | done (unused so far) |
| — | LINCS L1000 (PPN mechanism supervision) | `data/raw/mechanism/lincs/` | 27 GB, 9 files | done, gzip-verified |
| — | DepMap 24Q4 Public (PPN mechanism supervision) | `data/raw/mechanism/depmap/` | 936 MB, 3 files | done, size-verified |
| — | Pfib_423 (ext. val. 2, replaces Kremer) | `data/raw/pfib423/` | ~480 MB, 423 samples | done |
| — | AlphaMissense hg38 | `data/raw/annotations/` | 643 MB + tabix index | done, allele-matching verified |
| — | CADD v1.7 GRCh38 (bigWig, position-max) | `data/raw/annotations/` | 11.4 GB | done, integrity-verified |
| — | phyloP100way hg38 (bigWig) | `data/raw/annotations/` | 9.9 GB | done, integrity-verified |
| — | GERP Ensembl Compara hg38 (bigWig) | `data/raw/annotations/` | 9.6 GB | done, integrity-verified (see bug #10) |

Manifests: `data/manifests/tcga_manifest.json`, `slide_subset_cases.txt`,
`tcia_manifest.tcia`, `cptac_manifest.json`, `cptac_case_list.txt`,
`cptac_tcia_manifest.tcia`, `case_index_{tcga,cptac,kremer}.csv`,
`case_index_pfib423.csv`.

Embeddings so far: `data/embeddings/rna_{tcga,cptac}_full/`,
`rna_pfib423_full.npz`, `clinical_tcga_full/`.

## TCGA/TCIA/CPTAC collections used

TCGA-BRCA, TCGA-LUAD, TCGA-KIRC, TCGA-GBM, TCGA-LGG, TCGA-COAD, TCGA-OV,
TCGA-STAD (TCGA-GBM/LGG have no matched TCIA radiology collection under
those names -- confirmed not a bug, no matched imaging archive exists for
those two).

CPTAC-2, CPTAC-3 (GDC). CPTAC-CCRCC, CPTAC-LUAD, CPTAC-UCEC, CPTAC-PDA,
CPTAC-LSCC, CPTAC-AML matched on TCIA (5 of 12 candidate names tried
actually exist).

---

## Real bugs found and fixed — summary table

Full narrative detail for each lives in chat history. What matters going
forward is the durable lesson (right column) — read it before touching the
listed file again.

| # | File | What broke | Lesson |
|---|---|---|---|
| 1 | gdc_pull.py | Manifest pagination silently capped at 2000/type | Always loop on a paginated API's own total field; a suspiciously round result count is a truncation signal, not luck |
| 2 | gdc_pull.py | Fixed-count batching failed deterministically on large files | Batch by byte budget, not item count, for heterogeneous file sizes; add bisection-on-failure as a general fallback |
| 3 | gdc_pull.py | GDC single-file responses aren't gzip (only multi-file are) | Detect actual format by magic bytes, never assume based on request shape |
| 4 | mechanism_pull.py | Invented a Figshare endpoint that doesn't exist | Never assume an API endpoint from convention; check docs or a live 404 tells you fast |
| 5 | case_index.py | Wrong NBIA nesting-depth assumption (radiology showed 0% despite data existing) | Search recursively for known identifiers rather than assuming a fixed directory depth |
| 6 | run_rna_encoder.py, rna_to_anndata.py | No identity tracking through tokenization — silent misalignment risk | Never trust row order through a pipeline with any internal reordering/multiprocessing; carry identity as data (custom_attr_name_dict + emb_label), verify by reading it back |
| 7 | run_rna_encoder.py | Over-rigid exact-count check crashed on a legitimate small quality exclusion | Distinguish "some inputs legitimately excluded" (log + proceed, cap the fraction) from "structural corruption" (duplicate/unexpected ids — always hard fail) |
| 8 | annotation_sources.py | 3 wrong guesses for one UCSC URL (SSL host, path case, filename) | A source's own README can be wrong; trust the live directory listing over descriptive text |
| 9 | case_index.py | build_kremer_index checked for columns (DISEASE/KNOWN_MUTATION) that never existed; real column is ICD_10 | Silent falsy defaults (has_clinical=False) hide identically whether the data is genuinely absent or the lookup is broken — always verify against a real file before trusting a "0%" |
| 10 | annotation_sources.py / annotation_lookup.py | GERP bigWig silently corrupted despite correct file size, successful open(), plausible global stats -- 94.5% of real interval queries failed | Existence/size is NOT sufficient evidence a binary/indexed file downloaded correctly; do a functional check (real queries, multiple locations) at download time |
| 11 | run_clinical_encoder.py | Picked essentially at random among 8 GDC "clinical" files per case (patient XML, malignancy-form XML, 5 unrelated .txt tables) by UUID sort order | When one "data type" bundles several distinct file kinds, filter by an actual distinguishing filename pattern, never by an arbitrary tiebreak like a UUID |

Also worth remembering: **chromosome naming is not universal** (UCSC/CADD
use "chr1", Ensembl/GERP uses bare "1" — `BigWigLookup` now auto-detects
per file) and **SIFT/PolyPhen run in opposite severity directions** (SIFT
lower=worse needs MIN aggregation, PolyPhen higher=worse needs MAX — this
is the single easiest place to introduce a silent, plausible-looking wrong
result in the genomic encoder).

---

## Case index (built 2026-08-31/09-01, verified against real data)

| Cohort | Cases | RNA | MAF | Clinical | Slide | Radiology |
|---|---|---|---|---|---|---|
| TCGA (train/val/test 3872/483/510) | 4,865 | 87% | 84% | 100%* | 41% | 6% (291) |
| CPTAC (external_val_1, held out) | 2,202 | 98% | 97% | 0%** | 0%** | 13% (296) |
| Pfib_423 (external_val_2, held out) | 423 | 100% | — | 100% (ICD_10 only) | — | — |

\* Index-level flag built from the GDC manifest listing, not verified
against actual downloaded files (unlike has_slide/has_radiology, which do
check the filesystem) — real on-disk success rate is 4,756/4,759 (99.9%),
see bug #11.
\*\* CPTAC clinical is PDC-hosted (not this BCR-XML format), CPTAC
pathology is DICOM on TCIA (not GDC .svs) — both out of scope so far, not
bugs.

Radiology coverage (6%/13%) is lower than raw series-match counts
suggested because many patients contributed several series each --
confirmed real via an independent filesystem count, not a bug.
**Implication for training**: full five-modality cases are a small
minority; modality dropout during MoDiT training is the majority-case
requirement, not an edge-case feature.

---

## Pathology — in progress (as of 2026-09-01)

Tile extraction (`encoders/foundation.py::tile_slide`) verified working
against a real TCGA slide: Otsu tissue detection, grid tiling at target
MPP via direct level-0 reads (confirmed the right choice — this slide's
pyramid jumps 1x->4x->16x->32x with no intermediate level, so
`get_best_level_for_downsample` correctly always picks level 0 for our
~2x target), background/blur QC. 10,906 real tissue tiles found on one
representative slide in 119.2s uncapped; the coded `max_tiles=4000`
default meaningfully binds on well-populated slides (not just a number
that never matters). MPP property confirmed present on a 10-slide sample
(not universally 40x native -- 3/10 sampled were natively 20x; MPP-based
targeting, already the design, correctly handles this without change).

`UNIEncoder` and `GatedABMIL` already exist in foundation.py from early in
the project but are UNTESTED against real data and, for ABMIL, untrained
by construction -- see the Status section above for the CTransPath
decision and the bag-of-embeddings output design.

Still to do: build the CTransPath tile encoder, the per-slide
tile->encode->bag pipeline (stream-and-discard, same principle as slide
download -- do not persist raw tiles to disk), the cohort-wide
orchestration script (same case_id-tracking discipline as every other
encoder), and CPTAC's DICOM path (openslide may not read these directly --
`wsidicom` likely needed, not yet checked).

---

## Environment

- Conda env `raresynth` at `/data/pduttapramanik/raresynth/miniconda3`
  (installed without sudo, reusing the fmcl_paper3 installer script)
- PyTorch 2.6.0+cu124, 2x NVIDIA H200 NVL (143 GB each), 344 CPU cores, 1TB RAM
- Server: `sbcphadlp004` (internal), reached via jump host `129.106.31.39`,
  destination `129.106.31.17`
- All project data under `/data/pduttapramanik/raresynth/` — never home dir
- HuggingFace: logged in with a token; UNI access requested (institutional
  email required, matching name/affiliation, or auto-rejected) but not yet
  approved as of this writing

---

## Open items (not yet done, roughly in expected order)

- [ ] Baseline comparisons and ablations against the trained MoDiT
      checkpoint (/data/pduttapramanik/raresynth/runs/modit_full) --
      training itself works and generalizes; nothing has been COMPARED
      against yet. EMA tracking extended to GenomicSetEncoder/GatedABMIL
      too (currently MoDiT-only, a reasonable scope for the first working
      version, noted as a gap)
- [ ] The paper's actual evaluation metrics (FID/PRDC/C2ST, CMCS, mechanism
      retrieval, privacy) have not been computed on any trained model yet
      -- the train/val loss check confirms generalization on the training
      objective, not any downstream evaluation metric
- [ ] CPTAC pathology (DICOM format — needs investigation, likely wsidicom)
- [ ] CPTAC/Pfib_423 genomic+clinical (both TCGA-only so far; CPTAC's
      project field needs its own tissue-mapping logic, not yet built)
- [ ] Baseline implementations: CTGAN, TVAE, medGAN, TabDDPM, MMVAE, TotalVI
      (MVAE/MoPoE/copula/flat-U-Net/independent-diffusion already built)
- [ ] PPN training against LINCS/DepMap (data downloaded, training script
      not yet built)
- [ ] Mechanism-guidance energy live testing (guidance.py written and unit
      tested against mocks only, never run against real trained components)
- [ ] Real STRING/GO gene embedding to replace the fixed-seed random
      placeholder in gene_vocabulary.py (stated limitation, not urgent --
      same gene_index contract, drop-in swap whenever pursued)

