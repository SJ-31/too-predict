suppressMessages({
  library(BiocParallel)
  library(ALDEx2)
  library(here)
  Sys.setenv("RETICULATE_PYTHON" = here(".venv", "bin", "python"))
  library(reticulate)
  library(tidyverse)
  library(zellkonverter)
  library(scRNAseq)
  source(here("src", "R", "utils.R"))
  source(here("src", "R", "plotting.R"))
  use_python(here(".venv", "bin", "python3"))
})

pyutils <- new.env()
fs_dir <- here("data", "output", "feature_selection")
fs_lists <- list.files(here(fs_dir, "feature_lists"), full.names = TRUE)
source_python(here("src", "too_predict", "utils.py"), envir = pyutils)

adata <- pyutils$training_data_internal_test()
adata <- adata[, 1:2000]
counts <- adata$X %>% as.matrix()

# TODO: at least write the functions for this, narrow it down later
# Only need to try the differential proportions, use your own rust implementation
# for the pairs

library(propr)
pd <- propd(counts, adata$obs$tumor_type, alpha = NA)
pd_tb <- getResults(pd) |> as_tibble()
emergent <- setDisjointed(pd)
getResults(emergent) |> as_tibble()
## data <- AnnData2SCE(())
## features <- readLines(fs_lists[1])

## sce <- data[rownames(data) %in% features, ]

x <- counts[, 1]
y <- counts[, 2]

## assays(sce)$X <- as.matrix(assays(sce)$X)
## mode(assays(sce)$X) <- "integer"
## sce <- sce[, 1:200]
