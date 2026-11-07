# Perturb-seq comparison (Supplementary Information S5)

Post hoc comparison of the mapped UFM1 direction with measured knockdown profiles from the K562 and RPE1 essential-gene CRISPRi Perturb-seq screens of Replogle et al., Cell 185, 2559 (2022), https://doi.org/10.1016/j.cell.2022.05.013 (Figshare+ article 20029387; RPE1 raw single-cell file https://ndownloader.figshare.com/files/35775606).

```
D=data/raw/mechanism/ppn_extracted; L=data/raw/mechanism/lincs; K=data/raw/perturbseq/ReplogleWeissman2022_K562_essential.h5ad

# embedding space (K562)
python code/perturbseq/perturbseq_pseudobulk.py $K data/embeddings/perturbseq_k562/h5ad_input/cohort.h5ad 100
python code/perturbseq/perturbseq_embed.py tools/Geneformer data/embeddings/perturbseq_k562
python code/perturbseq/perturbseq_score.py data/embeddings/perturbseq_k562/rna_embeddings.npz runs/pfib_test/guidance_by_gene.npz $D/ufm1_rna_embedding_guidance.npz runs/perturbseq_validation/result.json

# gene space and calibration (K562; MINC is the minimum cells per perturbation)
python code/perturbseq/perturbseq_genespace.py $K $D/ufm1_guidance_direction.npz $L runs/perturbseq_validation/genespace.json
python code/perturbseq/perturbseq_calibration.py $K $D/trt_sh_matrix.npz $D/trt_sh_metadata.csv $L/GSE92742_Broad_LINCS_gene_info.txt.gz runs/perturbseq_validation/calibration.json

# RPE1 (original release file converted to the same format)
python code/perturbseq/convert_rpe1.py data/raw/perturbseq/Replogle2022_RPE1_raw_singlecell.h5ad data/raw/perturbseq/Replogle2022_RPE1.h5ad
MINC=30 python code/perturbseq/perturbseq_genespace.py data/raw/perturbseq/Replogle2022_RPE1.h5ad $D/ufm1_guidance_direction.npz $L runs/perturbseq_validation/genespace_rpe1.json
MINC=30 python code/perturbseq/perturbseq_calibration.py data/raw/perturbseq/Replogle2022_RPE1.h5ad $D/trt_sh_matrix.npz $D/trt_sh_metadata.csv $L/GSE92742_Broad_LINCS_gene_info.txt.gz runs/perturbseq_validation/calibration_rpe1.json
```
