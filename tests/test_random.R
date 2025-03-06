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
data <- AnnData2SCE((pyutils$training_data_internal_test()))
features <- readLines(fs_lists[1])

sce <- data[rownames(data) %in% features, ]

assays(sce)$X <- as.matrix(assays(sce)$X)
mode(assays(sce)$X) <- "integer"
sce <- sce[, 1:200]

aldex_result <- aldex_glm_wrapper(sce, "tumor_type", gene_col = "GENENAME")

counts <- assays(sce)$X

mm <- make_mm("tumor_type", data = colData(sce))

estimate <- aldex.clr(counts, mm, gamma = 1e-3)
scales <- getScaleSamples(estimate)
# <2025-03-05 Wed> What do these values mean exactly???
# Pretty sure these values are (logged) scale estimates i.e. W^{\perp}
# The authors of the VB paper looked at scale between the housekeeping genes
# but how does that work given that scale estimates are at the sample level

gmeans <- exp(colMeans(log(counts + 1), na.rm = TRUE))


meta <- as_tibble(colData(sce)) |> mutate(scale = rowMeans(scales))
meta$scale
