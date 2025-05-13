suppressMessages({
  library(edgeR)
  library(here)
  library(reticulate)
  library(Seurat)
  use_condaenv("too-predict")
  library(tidyverse)
  library(scRNAseq)
  source(here("src", "R", "utils.R"))
  source(here("src", "R", "plotting.R"))
})

ut <- import("too_predict.utils")
## ad <- import("anndata")
tdir <- here("data", "tests")

adata <- ut$training_data_internal_test()
adata <- adata[, 1:200]

dge <- adata2dge(adata)

## * [2025-05-02 Fri] Trying BayesPrism
mapping <- read_tsv(here("data", "mappings", "ensembl_113_id_mapping.tsv"))
symbol2ensembl <- setNames(mapping$ensembl, mapping$symbol)

ref <- readRDS(here(tdir, "scr_ref", "HTCA_ADULT_ADRENAL_GLAND.rds")) |>
  rename_seurat_features(symbol2ensembl, mapping = TRUE)
ref <- SCTransform(ref) # Normalization
ref <- RunPCA(ref)
ref <- FindNeighbors(ref)
ref <- seurat_subcluster_cells(ref, "Cell_Type")

library(BayesPrism)
# Wants genes in columns
counts <- t(dge$counts)
sc_data <- LayerData(ref) |>
  as.matrix() |>
  t()
cell_type_labels <- ref[[]]$Cell_Type
cell_state_labels <- ref[[]]$cell_subclusters
# Get cell states from sub-clustering i.e. extract cells of each distinct type and
# cluster

# Recommend removing outlier genes
# Removing genes on sex chromosomes as recommended by authors due to large variety
#   in the training data
sc_data <- BayesPrism::cleanup.genes(sc_data,
  gene.group = c("Rb", "Mrp", "other_Rb", "chrM", "MALAT1", "chrX", "chrY"),
  species = "hs",
  input.type = "count.matrix"
)

# Deconvolute by protein-coding only to speed things up
sc_data_pc <- select.gene.type(sc_data, gene.type = "protein_coding")

# Should include malignant cells in the reference, but this
#   is problematic because you want to use this with multiple organs so
# there will be multiple malignant cell types for each organ
# Sure you could do it separately for each organ on the training data, but
# what to do when it comes to test???

prism <- BayesPrism::new.prism(
  reference = sc_data_pc,
  mixture = counts,
  input.type = "count.matrix",
  cell.type.labels = cell_type_labels,
  cell.state.labels = cell_state_labels,
  key = NULL # Ideally not null
)
prism_results <- run.prism(prism, n.cores = 8)
saveRDS(prism_results, here("tests", "prim_test.rds"))
