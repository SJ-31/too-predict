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
sce <- sce[1:100, ]

pheatmap_helper(sce = sce, sample_annotations = list(
  tumor_type = "ggthemes::Tableau_20",
  Sample_Type = NULL
), order_on = "Sample_Type", pheatmap_kwargs = list(filename = here("foo.png")))

## --- CODE BLOCK ---
