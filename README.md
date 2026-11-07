# RareSynth

Code for the study of guided latent diffusion of Geneformer RNA embeddings: generator training, guidance toward a LINCS-derived UFM1 direction, comparison with Gaussian steerers, and the held-out Pfib_423 analysis.

## Setup

Python 3 with `code/requirements.txt`; Geneformer V2-104M installed from its repository into `tools/Geneformer`. Scripts use absolute paths under `/data/pduttapramanik/raresynth`; the repository root is expected at `/data/pduttapramanik/raresynth`. Commands are run from `/data/pduttapramanik/raresynth` with `export PYTHONPATH=/data/pduttapramanik/raresynth/code`.

Inputs, placed under `data/raw/`:

| Input | Location |
|---|---|
| TCGA STAR counts (GDC) | `tcga/` (step 1) |
| LINCS L1000 GSE92742: `Level5_COMPZ.MODZ_n473647x12328.gctx`, `sig_info.txt.gz`, `gene_info.txt.gz` | `mechanism/lincs/` |
| GTEx v8 `gene_reads.gct.gz` and `Annotations_SampleAttributesDS.txt` | `gtex/` |
| Pfib_423 gene-count matrices and sample annotations (non-strand-specific and strand-specific) | `pfib423/` |
| Perturb-seq K562 essential-gene screen, processed file `ReplogleWeissman2022_K562_essential.h5ad` (https://zenodo.org/records/10044268/files/ReplogleWeissman2022_K562_essential.h5ad) | `perturbseq/` |
| Perturb-seq RPE1 raw single-cell file `rpe1_raw_singlecell_01.h5ad` (https://ndownloader.figshare.com/files/35775606), saved as `Replogle2022_RPE1_raw_singlecell.h5ad` | `perturbseq/` |

## Run order

**1. TCGA embeddings**
```
python -m raresynth.data.gdc_pull manifest --projects TCGA-BRCA TCGA-LUAD TCGA-KIRC TCGA-GBM TCGA-LGG TCGA-COAD TCGA-OV TCGA-STAD --out data/manifests/tcga_manifest.json
python -m raresynth.data.gdc_pull download --manifest data/manifests/tcga_manifest.json --types rna --out data/raw/tcga
python -m raresynth.data.case_index tcga --manifest data/manifests/tcga_manifest.json --out data/manifests/case_index_tcga.csv
python -m raresynth.encoders.run_rna_encoder --case-index data/manifests/case_index_tcga.csv --tcga-manifest data/manifests/tcga_manifest.json --raw-dir data/raw/tcga --geneformer-repo tools/Geneformer --out data/embeddings/rna_tcga_full
```

**2. Pfib_423 embeddings** (`NS_COUNTS` and `SS_COUNTS` are the gene-count matrices inside the two folders)
```
NS=data/raw/pfib423/fib_ns--hg19--gencode34
SS=data/raw/pfib423/fib_ss--hg19--gencode34
python -m raresynth.data.case_index kremer --sample-annotation $NS/sample_annotation.tsv $SS/sample_annotation.tsv --out data/manifests/case_index_pfib423.csv
python -m raresynth.encoders.run_kremer_encoder --gene-counts $NS_COUNTS --geneformer-repo tools/Geneformer --out data/embeddings/rna_pfib423_ns
python -m raresynth.encoders.run_kremer_encoder --gene-counts $SS_COUNTS --geneformer-repo tools/Geneformer --out data/embeddings/rna_pfib423_ss
python code/heldout/combine_pfib_embeddings.py
python code/heldout/audit_pfib_identity.py --values
```

**3. Direction and reference** (`mechanism/`)
```
python code/mechanism/build_gtex_fibroblast_h5ad.py
python code/mechanism/run_gtex_geneformer.py
python code/mechanism/build_ufm1_consensus_direction.py
python code/mechanism/build_symbol_to_ensembl.py
python code/mechanism/fix_rna_scale_mismatch.py
python code/mechanism/make_shuffled_control.py
```

**4. Generator and guidance** (`experiments/`)
```
python code/experiments/export_rna_arrays.py
python code/experiments/export_rna_arrays.py --split test
python code/experiments/train_rna_dit.py --seed 0 --out runs/rna_dit_attempt1
python code/experiments/train_rna_dit.py --seed 1 --out runs/rna_dit_attempt1_seed1
python code/experiments/train_rna_dit.py --seed 2 --out runs/rna_dit_attempt1_seed2
python code/experiments/train_rna_dit.py --preprocess global_scale --fixed-clip4-audit --out runs/rna_dit_ablation_globalscale
python code/experiments/verify_clip_claim.py
for r in rna_dit_attempt1 rna_dit_attempt1_seed1 rna_dit_attempt1_seed2; do
  python code/experiments/memorization_audit.py --best runs/$r/best.pt --out runs/$r/memorization_audit.json
done
python code/experiments/map_validation_donor_level.py
python code/experiments/direction_controls_pc_profile.py
python code/experiments/guidance_redo.py
for r in rna_dit_attempt1 rna_dit_attempt1_seed1 rna_dit_attempt1_seed2; do
  python code/experiments/guidance_calibration.py --best runs/$r/best.pt --out runs/$r/guidance_calibration.json
done
python code/experiments/guidance_pc_matched_spill.py
python code/experiments/steering_comparator.py
python code/experiments/steering_comparator_noiserep.py --reps 10
python code/experiments/analyze_noiserep.py
python code/experiments/test_split_eval.py --mode harness
python code/experiments/test_split_eval.py --mode final --expect-sha $(sha256sum code/experiments/test_split_eval.py | cut -d' ' -f1)
```
**5. Held-out analysis** (`heldout/`)
```
SHA=$(sha256sum code/heldout/pfib_primary_test_v2.py | cut -d' ' -f1)
python code/heldout/pfib_primary_test_v2.py --mode dry
python code/heldout/pfib_primary_test_v2.py --mode final --ufm1-ids R20754 --expect-sha $SHA
python code/heldout/pfib_secondary_one_per_donor.py --primary-sha $SHA --ufm1-ids R20754
python code/heldout/gene_label_control.py count
python code/heldout/gene_label_control.py build
python code/heldout/gene_label_control.py score --primary-sha $SHA
python code/heldout/ufm1_expression_check.py
```

**6. Perturb-seq comparison** (`perturbseq/`; Supplementary Information S5; Replogle et al., Cell 185, 2559 (2022), https://doi.org/10.1016/j.cell.2022.05.013)
```
D=data/raw/mechanism/ppn_extracted; L=data/raw/mechanism/lincs; R=runs/perturbseq_validation
K=data/raw/perturbseq/ReplogleWeissman2022_K562_essential.h5ad
python code/perturbseq/extract_lincs_shrna.py $L $D

# embedding space (K562)
python code/perturbseq/perturbseq_pseudobulk.py $K data/embeddings/perturbseq_k562/h5ad_input/cohort.h5ad 100
python code/perturbseq/perturbseq_embed.py tools/Geneformer data/embeddings/perturbseq_k562
python code/perturbseq/perturbseq_score.py data/embeddings/perturbseq_k562/rna_embeddings.npz runs/pfib_test/guidance_by_gene.npz $D/ufm1_rna_embedding_guidance.npz $R/result.json

# gene space and calibration (K562)
python code/perturbseq/perturbseq_genespace.py $K $D/ufm1_guidance_direction.npz $L $R/genespace.json
python code/perturbseq/perturbseq_calibration.py $K $D/trt_sh_matrix.npz $D/trt_sh_metadata.csv $L/GSE92742_Broad_LINCS_gene_info.txt.gz $R/calibration.json

# RPE1 (original release file converted to the same format; minimum 30 cells per perturbation)
python code/perturbseq/convert_rpe1.py data/raw/perturbseq/Replogle2022_RPE1_raw_singlecell.h5ad data/raw/perturbseq/Replogle2022_RPE1.h5ad
MINC=30 python code/perturbseq/perturbseq_genespace.py data/raw/perturbseq/Replogle2022_RPE1.h5ad $D/ufm1_guidance_direction.npz $L $R/genespace_rpe1.json
MINC=30 python code/perturbseq/perturbseq_calibration.py data/raw/perturbseq/Replogle2022_RPE1.h5ad $D/trt_sh_matrix.npz $D/trt_sh_metadata.csv $L/GSE92742_Broad_LINCS_gene_info.txt.gz $R/calibration_rpe1.json
```

## Layout

```
code/raresynth/     package: model (DiT, diffusion), eval (fidelity metrics), data (GDC, case index), encoders (Geneformer), schedule
code/mechanism/     UFM1 direction, mapping to embedding space, GTEx fibroblast reference
code/experiments/   generator training, guidance, steering comparison, test-split evaluation
code/heldout/       Pfib_423 cohort file, primary and secondary analyses, controls
code/perturbseq/    Perturb-seq comparison of the mapped direction, LINCS shRNA extraction (Supplementary Information S5)
code/MANIFEST.sha256
data/manifests/     TCGA GDC manifest, TCGA and Pfib_423 case indexes
```

`code/MANIFEST.sha256` lists SHA-256 checksums of all files. The scripts under `heldout/` and `experiments/steering_comparator_noiserep.py` are the files whose hashes are cited in the manuscript.
