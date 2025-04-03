suppressMessages({
  library(BiocParallel)
  library(ALDEx2)
  library(edgeR)
  library(here)
  library(reticulate)
  use_condaenv("too-predict")
  library(tidyverse)
  library(zellkonverter)
  library(scRNAseq)
  source(here("src", "R", "utils.R"))
  source(here("src", "R", "plotting.R"))
})

pyutils <- new.env()
fs_dir <- here("data", "output", "feature_selection")
fs_lists <- list.files(here(fs_dir, "feature_lists"), full.names = TRUE)
source_python(here("src", "too_predict", "utils.py"), envir = pyutils)

adata <- pyutils$training_data_internal_test()

adata <- adata[, 1:2000]
counts <- adata$X %>% as.matrix()
