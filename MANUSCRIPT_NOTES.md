# RareSynth — Manuscript Notes

What the PAPER needs to say, as distinct from PROGRESS.md (what the CODE
does). Read this before writing or revising any section. Update it whenever
a decision changes what a section will say, not just when code changes.

Target: Nature Communications (primary), npj Digital Medicine or Genome
Biology as fallbacks. 25-30 pages, ~4,500 words main text + extended
Methods, per Nature-journal formatting norms.

---

## The original draft's fatal problem (already fixed, keep it fixed)

The draft submitted for review had Sections 8 (Results) and 9
(Discussion) reporting specific numbers (FID 1.35, C2ST 0.53, 78.5%
pathway enrichment, "approaching clinical expert panel" accuracy) that
were NEVER COMPUTED — Tables 4-6 were empty placeholders in the same
document. This is fabrication, not a drafting shortcut, and both sections
were deleted at the start of this rebuild. They get rewritten ONLY from
real computed output, never before. Do not let a future editing pass
reintroduce placeholder numbers "to hold the argument's shape."

---

## Core redesign, and why (for Methods + a paragraph in Discussion)

**Original design (draft)**: A perturbation network (PPN) + cross-modal
mapper (CPM) generated 100,000 synthetic training profiles; the diffusion
model was trained ON THAT SYNTHETIC OUTPUT. Fatal flaw: the diffusion
model's distribution is then upper-bounded by PPN+CPM's — it cannot
express anything the two MLPs could not already produce directly, making
the diffusion stage arguably a redundant distillation step. A reviewer's
first question would be "what does the diffusion model add over sampling
PPN+CPM and adding noise?" and the original ablation table never asked it.

**New design**: MoDiT trains on REAL data only (TCGA + GTEx + CPTAC).
Rare-disease/mechanism specificity enters at SAMPLING TIME via a
differentiable mechanism-consistency energy (classifier-guidance style,
`model/guidance.py`), not via a fabricated training corpus. This removes
the circularity, and the guidance scale `s` (0 upward) is now a clean,
continuous ablation axis the reviewer will want to see.

**Architecture**: original design used a 1-D U-Net over the concatenated
[z_geno; z_rna; z_path; z_rad; z_ehr] vector — convolution treats the
coordinate index as spatial, which it is not (dimension 47 of a
Geneformer CLS embedding has no adjacency relationship to dimension 48),
and receptive fields cross modality boundaries as if they were
neighbouring pixels. Replaced with MoDiT: five modality tokens, shared
width, self-attention (`model/dit.py`). The old U-Net is KEPT as a named
baseline (`baselines/simple.py::FlatUNet1D`) so the architecture change is
measured, not just asserted.

**PPN/CPM**: retained, but demoted from a generative pathway to a
DIRECTION PRIOR inside the guidance energy. The CPM specifically was a
~1.5M-parameter 3-MLP stack trained on 50 paired tumor/normal
observations (30,000:1 parameter-to-sample ratio, statistically
indefensible as a generator) — replaced with a low-rank (r<=8) ridge
regression, selected by leave-one-cancer-type-out CV, used only to
constrain a DIRECTION not a full mapping, which matches what 50
observations can actually support.

**Genomic encoder**: original design assumed per-variant Enformer deltas
across ~19,200 genes x ~10,000 patients (10^8-10^9 forward passes, a
GPU-years-scale computation, not feasible). Replaced with a set-transformer
over a 14-dim per-gene annotation vector for the top-k most-perturbed
genes (`encoders/genomic.py`) -- tractable, and the encoder's own
attention weights are directly interpretable as "which genes drove this
patient's representation," a property Enformer deltas would not give for
free. Enformer is retained ONLY as a small supporting analysis for the
~200 curated disease genes (~400 forward passes, genuinely feasible),
used to derive the `geno` term of the mechanism-guidance direction, not as
the cohort-wide encoder.

---

## Data — what's real, what's claimed, and how to describe each

**Training cohort**: TCGA (8 cancer types: BRCA, LUAD, KIRC, GBM, LGG,
COAD, OV, STAD), GTEx v8 as the paired normal baseline. Real, open,
GDC/GTEx portal.

**External validation 1**: CPTAC (independent institutions, independent
processing pipeline from TCGA). ~2,200 cases, RNA+MAF from GDC, imaging
from TCIA (5 real matched collections).

**External validation 2**: "Pfib_423" — 423 real Mendelian/mitochondrial
disease patients (Yepez et al. 2022, Genome Medicine, Zenodo
4646823+4646827), fibroblast RNA-seq. THIS IS WHAT CARRIES THE
RARE-DISEASE CLAIM — it is real monogenic disease, in a clinically
accessible tissue, with known-or-suspected causal genes, and directly
tests whether mechanism structure learned from cancer tumor/normal pairs
transfers to germline monogenic disease (the paper's single biggest
inferential leap). Cite as: Yepez, V.A. et al. Genome Medicine (2022);
data at Zenodo DOIs 10.5281/zenodo.4646823 and 10.5281/zenodo.4646827.
NOTE: an earlier, smaller version of this (Kremer et al. 2017, 119
samples) was used through part of this project and is now SUPERSEDED —
if any earlier notes, code comments, or draft text reference "Kremer" as
the external validation cohort, that is the deprecated smaller version;
update to Pfib_423.

**Mechanism supervision**: LINCS L1000 (Level 5 consensus signatures,
GSE92742+GSE70138) + DepMap 24Q4 Public (CRISPR knockout effect +
baseline expression). Used to train the PPN (perturbation -> expression
shift), not as training data for the diffusion model itself.

**DROPPED from the original draft, do not reintroduce**: UDN, 100,000
Genomes Project, UK Biobank. None can supply five-modality rare-disease
cases to an outside investigator without an application process this
project cannot complete; claiming them would be claiming data the paper
did not actually use. If a methods reviewer asks why these obvious
resources are absent, the honest answer (accessibility, not oversight) is
defensible and should be stated plainly rather than hedged.

**Modality coverage is NOT five-of-five for most cases** — real, confirmed
numbers: full radiology coverage is only 6% (TCGA) / 13% (CPTAC) of
cases. This is not a limitation to apologize for; it is the reason
modality dropout during training is a design requirement, not a
robustness nicety, and the paper should frame it that way rather than
defensively.

---

## Novel evaluation contributions (Methods + emphasize in Introduction)

1. **Cross-Modal Coherence Score (CMCS)** — ratio of a modality-pair
   predictor's R^2 on synthetic pairs to its R^2 on real held-out pairs,
   with a shuffled-pair control reported alongside every CMCS. This is
   the paper's own evaluation contribution: FID/PRDC/C2ST all measure
   marginals or an undifferentiated joint and would NOT detect a generator
   that samples each modality correctly but pairs them at random — which
   is exactly the failure mode the paper's central claim is about.
   IMPORTANT CAVEAT confirmed on the smoke test: a Gaussian copula scored
   competitively on CMCS (0.76-0.79 vs MVAE's 0.48-0.57) because a copula
   preserves rank-correlation by construction. The copula MUST stay in
   every results table regardless of outcome — omitting it would be
   exactly the kind of gap a reviewer catches immediately.

2. **Mechanism retrieval** — given a sample generated under guidance
   toward gene g, can a retriever trained only on real data recover g
   from among all candidates? Stronger and more interpretable than
   pathway enrichment alone.

---

## Baselines — required for the results table

CTGAN, TVAE, medGAN, TabDDPM, Gaussian copula, flat 1-D U-Net (the
original draft's own architecture, now a baseline), independent
per-modality diffusion (no cross-modal coherence, the "null hypothesis"
for the paper's central claim), MVAE (PoE), MoPoE-VAE, TotalVI/scVI (RNA
slice). The MVAE/MoPoE family is the REAL competition for joint
multimodal generation — omitting it in favor of only unimodal
GANs/tabular synthesizers is named in this project's own earlier analysis
as "the most likely reason a methods reviewer rejects the paper."

Status: MVAE/MoPoE/copula/flat-U-Net/independent-diffusion implemented.
CTGAN/TVAE/medGAN/TabDDPM/TotalVI NOT yet implemented (see PROGRESS.md
open items).

---

## Ablations required

Guidance off (s=0, the real test of whether mechanism guidance adds
anything over the real-data prior alone); guidance-scale sweep;
cross-modal prior removed from the guidance energy; modality tokens vs.
flat concatenation (MoDiT vs. FlatUNet1D, same training data, isolates
the architecture change); HPO conditioning removed; leave-one-modality-out;
held-out-gene PPN evaluation (gene-disjoint split, not sample-disjoint —
tests generalization to a genuinely unseen gene); leave-one-cancer-type-out
for the guidance direction prior; synthetic-sample-count scaling; RNA
encoder choice (Geneformer vs. an alternative like scVI, to show the
result is not an artifact of one frozen encoder — not yet planned in
detail, worth a line if time allows).

---

## Privacy evaluation (required, cheap, do not skip)

Membership inference (distance-threshold attack, AUC target ~0.5) and
distance-to-closest-record, both implemented (`eval/privacy.py`). Any
synthetic clinical data paper without this draws an immediate reviewer
request; already built, needs a live run against real trained-model
output once available.

---

## What's genuinely done vs. what still needs live results

DONE, real numbers exist, ALL FOUR ENCODERS COMPLETE (as of 2026-09-01):
data acquisition (all sources), RNA-seq encoding (TCGA/CPTAC/Pfib_423),
genomic encoding (14 features, real MAF + 4 external annotation sources),
clinical encoding (TCGA), pathology encoding (TCGA, CTransPath +
bag-of-tile-embeddings, 2,000/2,000 cases, the cleanest verification
result of any modality -- zero anomalies of any kind on a comprehensive
post-run check). ALSO DONE: the assembly layer joining all four by
case_id (4,865 TCGA cases, 36.0% with all four modalities present) and a
working, live-tested PyTorch Dataset serving real batches. **ALSO DONE,
and this is the first genuine evidence the architecture works at all**:
a real joint training run of MoDiT + GenomicSetEncoder + GatedABMIL on
real TCGA data, loss decreasing monotonically (0.990 -> 0.748 -> 0.667
over 3 epochs), zero NaN, checkpoint verified correct. Two serious risks
were found and fixed getting here (see PROGRESS.md): a masking-convention
bug that would have produced NaN loss on most training batches (~41%
pathology coverage meant most batches would have hit an all-padding
pathology bag), and a fully-diagnosed false alarm around MoDiT's
deliberate zero-initialization (a temporary, one-step-only artifact, not
a bug -- worth remembering if this ever needs re-explaining).

STILL NOT RESULTS in the paper's sense: this training run used a small
model (d_model=128, depth=4, 9.2M total params) for 3 epochs specifically
to confirm the pipeline works end-to-end quickly, not to produce a model
worth evaluating. Scale-up (the real architecture size, full epoch
budget, full tile counts, baseline comparisons, ablations) has not
happened. Every number that goes into Tables 4-8 of the eventual
manuscript still does not exist. Do not conflate "the pipeline trains
correctly at small scale" with "the paper has results" when drafting --
this milestone proves FEASIBILITY, which is worth a sentence in Methods
about implementation verification, not a Results claim.

**UPDATE, same day**: the full-size run (real architecture, 73.6M params,
200 epochs, real max_tiles=2000) is also done, and — genuinely worth a
sentence in Methods, not just an internal check — a real train-vs-
validation comparison was run specifically because the training loss
collapsed fast enough (0.99 -> ~0.01 within 10-15 epochs) that
memorization was a real, plausible concern given 73.6M params against
only 3,836 training cases. Result: train loss 0.00396 vs held-out val
loss 0.00400 on 479 real cases the model never saw -- essentially
identical, val marginally higher as expected for genuine generalization,
not the large gap memorization would produce. This is real evidence the
architecture generalizes at this scale, worth citing in an implementation
-verification sentence ("the model was confirmed to generalize to
held-out cases before any downstream evaluation, train/val loss 0.00396
vs 0.00400") -- but it is loss on the diffusion training objective, NOT
any of the paper's actual evaluation metrics (FID/PRDC/C2ST, CMCS,
mechanism retrieval, privacy), none of which have been computed on any
trained model yet. Baselines and ablations have not been run against
this checkpoint either. Still no Results-table numbers exist.

**SECOND UPDATE, same day -- the checkpoint above is STALE, do not cite
its numbers.** Generating the first real synthetic samples surfaced a
genuine cross-modal scale bug: RNA embeddings (from Geneformer's own
EmbExtractor, used directly with no normalization step) sit at ~6.6x the
scale of every other modality (all of which are properly unit-norm by
design). This is a real architectural inconsistency worth a sentence in
Methods regardless of outcome ("RNA embeddings are L2-normalized to unit
norm, matching every other modality's encoder, before entering the
shared diffusion process -- an early version omitted this for RNA
specifically since it used a third-party library's raw output
convention, corrected before any reported results"). Fixed and a fresh
training run is in progress; the train/val generalization number above
(0.00396 vs 0.00400) was computed on the PRE-FIX model and should be
RE-CONFIRMED on the corrected checkpoint before citing, not assumed to
still hold (plausibly similar, but not confirmed -- verify, don't
assume, matching every other piece of this project so far). See
PROGRESS.md bug #12 for the full investigation.

**THIRD UPDATE, same day -- the RNA fix alone was NOT sufficient either;
a second, more significant scale bug found and fixed before anything is
trustworthy.** Re-checking generated-sample quality after the RNA fix
(not assuming the fix had resolved things) found the clipping problem
essentially unchanged. Root cause, confirmed by directly measuring real
data: every modality's L2-normalize-to-unit-NORM convention gives a real
per-coordinate std of only ~0.03, roughly 33x smaller than the
UNIT-VARIANCE data scale standard diffusion formulations (including this
project's own) implicitly assume. This is worth its own sentence in
Methods: "training data is rescaled by a factor computed from its own
measured standard deviation (not a fixed literature constant) to match
the diffusion process's standard unit-variance assumption; generated
samples are rescaled back by the inverse factor before evaluation." Fixed
via compute_data_scale() (measures real std directly, saved into the
checkpoint, applied consistently across training/sampling/evaluation).

**Final, trustworthy result (third checkpoint, both fixes applied)**:
train/val loss 0.01017 vs 0.01011 (generalization re-confirmed, not
assumed to carry over from the earlier checkpoints). Generated-sample
clipping: 0.00% of samples affected (down from 99.8% before either fix),
overall generated std 0.0506 vs real data's 0.02995 -- a normal,
expected level of generation variance, not a systematic distortion. THIS
is the checkpoint whose generated samples are trustworthy for downstream
fidelity/coherence/privacy evaluation. Neither of the two earlier
checkpoints should ever be cited, even informally.

**The methodological lesson worth a sentence in the paper regardless of
which specific numbers end up in the final tables**: sampled output
quality (value distributions, comparison against real data statistics,
not just absence of NaN/errors) was checked directly before trusting any
generated sample for evaluation, and this caught two real, substantial,
jointly-necessary scale corrections that would otherwise have silently
corrupted every downstream fidelity/coherence metric computed against
this model's output. This is exactly the standard of scrutiny that
protects the paper's Results section from a technically sophisticated
reviewer's scrutiny -- worth stating as a verification step taken, not
hiding as an embarrassing false start.

---

## First real fidelity results (2026-09-02) -- honest, not flattering, likely to change

Before trusting eval/fidelity.py's FID computation at all, found and
fixed a real gap: the code computed FID directly on raw, full-dimensional
embeddings (up to 3584-dim), while its own docstring described computing
it in a trained classifier's much lower-dimensional penultimate feature
space -- the classifier step was never actually implemented. Confirmed
this mattered before building the fix: two samples from the IDENTICAL
distribution produced a raw-space FID of 4939 (should be ~0) at our real
sample sizes -- a severe, disqualifying small-n-large-p covariance
estimation problem, not a subtle concern. Built a real tissue-of-origin
classifier (trained on real data, 100% validation accuracy -- plausible
given how biologically well-separated tissue-of-origin is in multi-omic
cancer data) and used its penultimate features for the joint FID, PCA for
per-modality FID. Worth a Methods sentence regardless of final numbers:
"FID is computed in the penultimate feature space of a tissue-of-origin
classifier trained on real data (not raw embedding space, which is
numerically unstable at these sample sizes for the joint 3584-dimensional
representation)."

**First real result, on the 200-epoch checkpoint** (479 real held-out
val cases vs 479 generated): joint FID 18.67, PRDC precision ~0.000
across every modality, C2ST accuracy 0.99-1.00 everywhere. Honest
reading: a real classifier distinguishes real from generated samples
almost perfectly at this stage; generated samples do not yet closely
match real data's local density (near-zero precision), though recall is
higher for the joint representation (0.906) than per-modality, a
mode-covering-but-imprecise pattern. THIS IS NOT NECESSARILY THE NUMBER
THAT GOES IN THE PAPER -- a real, likely-relevant gap was found
immediately after (see below): the training loop had NO learning-rate
decay at all (warmup then held constant for the entire remainder of
training, against standard diffusion-transformer practice), and the
200-epoch run's loss was still oscillating rather than settled in its
final ~150 epochs. Fixed (cosine decay after warmup) and a longer run
(500 epochs) is in progress with this one change, deliberately isolated
from other hyperparameter changes so any improvement is attributable to
it specifically. RE-RUN eval_fidelity_real.py on the new checkpoint and
compare directly before deciding which numbers (if either) belong in the
paper -- do not assume the fix helped without re-measuring, matching
every other verification step in this project.

If the longer, properly-scheduled run still shows a similar
precision/C2ST pattern, that is itself a legitimate, reportable finding
(the model captures a plausible joint distribution but has not yet
achieved sample-level fidelity) rather than something to keep silently
re-running until a flattering number appears -- worth being explicit
about this discipline given the project's stated goal of a
reviewer-proof paper: an honest limitation, clearly measured and
discussed, is defensible; a cherry-picked favorable run is not.

**UPDATE: the 500-epoch run is done, and this is exactly the case
described above.** FID improved for most modalities (a real,
measurable, honestly-reportable gain from the LR-decay fix) but PRDC
precision remained exactly 0.000 across every modality in both the
200-epoch and 500-epoch runs, and C2ST accuracy remained 0.98-1.00 in
both -- a real classifier still nearly perfectly separates real from
generated samples. Pathology's recall specifically got WORSE (0.610 to
0.000) at the longer duration, a real regression. One anomaly flagged
but not yet explained: geno/rna/ehr recall values are IDENTICAL to
three decimal places between the two runs, which is unexpected for two
genuinely different checkpoints and needs investigation before being
treated as a stable, trustworthy number.

**Framing for the paper, if these numbers or ones like them are what
end up being reported**: this is not a failure to hide -- it is a
genuine, useful limitation finding about what a first working version of
a novel five-modality architecture achieves on ~3,800 real training
cases, and the paper should say so plainly rather than obscure it with
selective reporting. A defensible way to frame this in Discussion:
distributional similarity (FID) can improve with training refinements
like proper LR scheduling while individual-sample fidelity (precision,
C2ST) remains a harder problem, plausibly limited by real training data
scale relative to model capacity (67.7M MoDiT parameters against 3,836
cases) rather than by the architecture or guidance mechanism specifically
-- and this is exactly why the CMCS and mechanism-retrieval evaluations
(Section 3.5) matter as much as or more than raw fidelity: a generator
that has not yet achieved tight per-sample fidelity may still capture
useful cross-modal structure, which is the paper's actual central claim,
not "these samples are indistinguishable from real data." Do not
overstate what fidelity numbers like these support.

Next real levers being investigated, in rough order of expected impact:
training data scale (CPTAC and Pfib_423 remain RNA-only, not part of
this joint 5-modality training set at all -- a real, addressable gap),
classifier-free guidance strength at sampling time (not yet tuned),
DDIM step count (currently 200, unexplored whether more helps precision
specifically).

**UPDATE: CFG-scale investigation, genuinely improved understanding.**
A sweep of guidance scale (2 -> 35) substantially improved JOINT FID
(16.28 -> 6.32) and JOINT precision (0.136 -> 0.163), confirming
guidance strength is a real, usable lever. But per-modality precision
stayed at exactly 0.000 regardless -- traced to a real bug in the
EVALUATION, not the model: the per-modality feature space used PCA (fit
on real data), which is unsupervised and was structurally blind to
whatever direction CFG was actually improving. Fixed with a per-modality
trained tissue classifier instead of PCA.

**Real, corrected result -- worth quoting directly in the paper, both
the positive and the limitation**:

| modality | classifier val acc | FID | precision | recall |
|---|---|---|---|---|
| joint | 1.000 | 6.31 | 0.163 | 0.747 |
| geno | 0.246 | 31.84 | 0.000 | 1.000 |
| rna | 0.929 | 21.38 | 0.090 | 0.741 |
| path | 0.480 | 12.86 | 0.785 | 0.935 |
| ehr | 0.996 | 16.01 | 0.025 | 0.313 |

**CORRECTED FRAMING (2026-09-03, after two independent methodological
reviews of the investigation write-up -- see below): the language above
was overconfident and has been revised. Use the corrected language in
any drafting, not the original.**

The classifier-based feature space measures whether generated samples
occupy regions a tissue classifier associates with the correct cancer
type -- call this **tissue-conditional fidelity**, not "fidelity" or
"sample fidelity" unqualified. A classifier trained to separate tissues
is explicitly encouraged to organize its representation around tissue
identity and can discard everything else; samples with correct tissue
identity can score well here even if other biological characteristics
are unrealistic, and vice versa. PCA was NOT a bug to be removed -- it
answers a different, complementary, still-useful question (general
distributional fidelity, tissue-agnostic) and should be RETAINED
alongside the classifier view, not replaced by it. Report both.

**Pathology precision (0.785)**: do not call this "genuinely strong" or
a settled result. Correct framing: "pathology showed high PRDC precision
in the current classifier-derived (tissue-conditional) feature space;
because tissue classification in this modality is only moderate (48%
accuracy, well below RNA's 92.9% or EHR's 99.6%), this result is
provisional pending independent representation, morphology-level
validation, and confirmation of patient-level (not tile-level) split
integrity." No real-vs-real baseline has been established yet for this
number either -- without knowing what real-vs-real precision looks like
in this same feature space, 0.785 has no reference point to be judged
against as "high."

**Genomic's 0.000**: do not call this "explained" or treat the
mutation-burden classifier (in progress) as sufficient on its own.
Correct framing: "the tissue-conditioned genomic metric is poorly
discriminative on held-out real data (classifier accuracy 24.6% vs 12.5%
chance) and is therefore not suitable as the primary genomic fidelity
endpoint; genomic fidelity will instead be assessed with a panel of
cancer-relevant genomic distributions and dependency structures
(mutation burden, gene-level, pathway-level, conditional association),
not mutation burden alone."

**"GTEx is a clean opportunity for more training data"**: also
overconfident. Correct framing: "GTEx is a candidate RNA reference
resource requiring a controlled, source-aware harmonization and
ablation study before use, since normal-tissue vs tumor biology and
study-specific processing differences (TCGA vs GTEx were generated by
different consortia with different pipelines) can introduce confounds
rather than clean additional signal." Do not fold GTEx into training
without first checking for a batch/source effect between TCGA and GTEx
specifically.

Next, in priority order (see the two review documents for full
justification of each): (1) real-vs-real PRDC baseline, computed by
splitting real held-out data in half -- establishes what "good"
precision actually looks like before any generated-vs-real number can
be interpreted; (2) negative controls (shuffled tissue labels, permuted
cross-modal pairings, randomized data) alongside the existing positive
control, to confirm the metric responds correctly in BOTH directions,
not just detects an easy positive case; (3) verify no patient/case
appears with data in both train and val splits, and specifically that
pathology tiles are never split across a single case (should already be
true given splits are done at the case level, but confirm explicitly
rather than assume, especially before making any claim about
pathology's precision); (4) run CMCS (cross-modal coherence) against
real generated output for the first time -- this is the paper's actual
central claim and has not been tested at all yet, a bigger gap than any
single-modality fidelity number; (5) full PRDC quartet (density,
coverage, not just precision/recall) with bootstrap confidence
intervals, not point estimates; (6) baselines (MVAE/MoPoE/copula/etc.)
run under the identical frozen evaluation protocol, so any comparison is
apples-to-apples.

**A general methodological point worth keeping close for the rest of
this project**: prefer language like "different metrics probe different
properties of synthetic data, and no single representation provides a
complete fidelity assessment" over "the metric was wrong" -- the
repeated finding that a given metric wasn't measuring what was assumed
is itself a principled argument for multidimensional, modality-aware,
cross-modal validation, not a series of embarrassing false starts. Frame
it that way in the paper.

Pathology's output format is a genuine architectural point worth
remembering for Methods: it is a per-case BAG of variable-length tile
embeddings (CTransPath, 768-dim per tile, capped at 8,000 tiles/case),
not a single fixed vector like the other three modalities. This is
deliberate -- GatedABMIL (gated attention multiple-instance-learning
aggregation) is untrained by construction at the encoder stage and must
be learned JOINTLY with MoDiT, not run now as a frozen feature extractor.
This is worth stating plainly in Methods as a genuine architectural
choice (attention-based MIL trained end-to-end on the actual downstream
objective, not a frozen off-the-shelf aggregator) — a legitimate,
citable design decision.

UNI (gated HuggingFace access) was the original planned pathology
encoder; access was requested but never approved within this project's
timeline, so CTransPath (open, TCGA/PAIP-native pretraining, smaller/
faster) was used instead. If UNI access comes through later, it remains
available as a secondary tile-encoder comparison/ablation -- a legitimate
addition to the paper, not required for the core result.

Smoke test (simulated data with known ground-truth cross-modal
dependence, not real training data) already shows the expected ordering:
MoDiT beats MVAE/copula/independent-diffusion on both Frechet distance and
CMCS, imputation of a held-out modality achieves R^2=+0.50. This is a
pipeline-correctness check, not a citable result — the simulation's
dependence structure is linear and synthetic, nothing like real biology.
Useful for the Methods section as "verified the implementation reproduces
known ground truth before training on real data," not as a Results claim.

---

## Length and section budget (from the original planning pass, still valid)

Intro 1,200 / Related work 1,300 / Data 1,400 / Method 3,000 / Setup 1,200
/ Results 2,500 / Discussion 1,200 / Conclusion 300 words, plus ~6 figures
and ~8 tables. Consolidate the original draft's duplicated section
numbering and colliding reference lists (confirmed present in the
original file — e.g. reference [13] pointed to two different sources in
two different section-local lists) into one single numbered reference
list before this goes anywhere near a co-author or a submission system.

---

## Writing-style requirements (apply to every section, see /preferences.md
## for the full standing rules — this is a pointer, not a restatement)

Every sentence must be defensible against the paper's own content or
cited literature; no formulaic preambles; banned AI vocabulary/structure
list applies; conclusion sections stay results-free and citation-free;
humanizer skill required before any drafting pass.

---

## PROJECT SCOPE DECISION (2026-09-05) -- read before drafting anything further

**Pijush explicitly chose "full rigor"** when offered a choice between
full rigor (every baseline, every ablation, everything both external
reviews of 2026-09-03 recommended) and a smaller, still-defensible
reduced scope. This is the standing decision -- do not draft toward a
reduced-scope version without this being revisited explicitly.

**What this means for the paper concretely**: Results, Discussion, and
Abstract remain empty (see the placeholder in Section 4) until ALL of
the following exist, not a subset -- CMCS run for real (not started),
the Pfib_423 external-validation test with real mechanism-guided
sampling (not started, blocked on PPN training), baselines including at
minimum CTGAN/TVAE/medGAN/TabDDPM/copula/MVAE/MoPoE run under an
identical frozen protocol (currently zero baselines trained on real
data), the full ablation suite, and privacy metrics against real
generated output. See PROGRESS.md's "PROJECT SCOPE DECISION" section for
the complete phased roadmap and current status of each item -- this file
does not duplicate that list, only points to it.

**Honest scope note for whoever drafts Results eventually**: this is a
large amount of remaining empirical work, comparable to everything built
so far in this project. Do not let a future drafting pass assume more of
this exists than actually does just because the Methods section (which
does not depend on final results) reads as complete.
