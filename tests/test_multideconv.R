suppressMessages({
  library(BiocParallel)
  library(here)
  library(reticulate)
  use_condaenv("too-predict")
  library(tidyverse)
  library(Seurat)
  source(here("src", "R", "utils.R"))
})

tdir <- here("data", "tests")
scrna <- here(tdir, "scr_ref")
ut <- import("too_predict.utils")
ad <- import("anndata")

library(multideconv)

adata <- ut$training_data_internal_test()

obj <- readRDS("/home/shannc/Bio_SDD/too-predict/data/tests/scr_ref/htca_2025-4-21/HTCA_ADULT_ADRENAL_GLAND.rds")

# Should rely on multideconv's internals for the "common" cells
# give tissue-specific cell types from scrna
multideconv_wrapper <- function(counts, methods = c("Quantiseq", "CBSX", "Epidish", "DeconRNASeq", "DWLS", "MOMF"),
                                sc_obj = NULL, doParallel = TRUE, workers = 8) {

}

counts <- as.matrix(t(adata$X))
rownames(counts) <- adata$var$GENENAME
colnames(counts) <- rownames(adata$obs)

sc_matrix <- as.matrix(LayerData(obj))

sc_meta <- obj[[]] |> dplyr::rename(cell_label = Cell_Type, sample_label = Sample_ID)

# Use symbols only
do_deconv <- function(f) {
  result <- compute.deconvolution(
    raw.counts = counts,
    methods = c("Quantiseq", "Epidish", "DeconRNASeq", "MCP"),
    normalized = FALSE,
    sc_deconv = TRUE,
    sc_metadata = sc_meta,
    sc_matrix = sc_matrix
  )
  saveRDS(result, f)
}

# BUG: [2025-05-07 Wed] can't install multideconv yet, but check performance of other
# ref-based methods before bothering
result <- read_existing(here("data", "tests", "multideconv.rds"), do_deconv, readRDS)
