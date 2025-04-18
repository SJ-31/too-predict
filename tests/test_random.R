suppressMessages({
  library(BiocParallel)
  library(ALDEx2)
  library(edgeR)
  library(here)
  library(reticulate)
  use_condaenv("too-predict")
  library(tidyverse)
  library(DESeq2)
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

counts <- adata$X %>%
  as.matrix() |>
  t()

# [2025-04-09 Wed] Trying to use DESeq2 cause edgeR is acting up
# but the contrast spec is worse here. Can't do what you did in edgeR which
# was to compare one type against average of all other types

## dds <- DESeqDataSetFromMatrix(
##   countData = counts,
##   colData = adata$obs,
##   design = ~tumor_type
## )
## rownames(dds) <- rownames(adata$var)

## mm <- model.matrix(~tumor_type, data = colData(dds))

## dds$Project_ID <- str_replace_all(dds$Project_ID, "-", "_") |> as.factor()

## group_vec <- colnames(mm) |> keep(\(x) str_detect(x, "tumor_type"))
## mean_val <- 1 / (length(group_vec) - 1)
## contrast_str <- map_chr(group_vec, \(x) {
##   mean_others <- paste(mean_val, "*", group_vec[group_vec != x], collapse = "+")
##   paste0(x, "-", "(", mean_others, ")")
## })
## ccs <- makeContrasts(contrasts = contrast_str, levels = mm)

## vals <- levels(dds$tumor_type)

## coefs <- lapply(vals, \(x) {
##   colMeans(mm[dds$tumor_type == x, ])
## }) |> `names<-`(vals)

## lst_remove <- function(lst, name) {
##   copy <- rlang::duplicate(lst, shallow = FALSE)
##   copy[[name]] <- NULL
##   copy
## }


## get_contrast <- function(coefs, target) {
##   first <- coefs[[target]]
##   removed <- lst_remove(coefs, target)
##   mean_val <- 1 / (length(coefs) - 1)
##   applied <- lapply(removed, \(x) mean_val * x) |> purrr::reduce(\(x, y) x - y)
##   first - applied
## }

## dds <- DESeq(dds, test = "LRT", reduced = ~1)
## results <- resultsNames(dds)

## results(dds, contrast = get_contrast(coefs, "BRCA"))
