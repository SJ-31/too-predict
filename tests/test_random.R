suppressMessages({
  library(BiocParallel)
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

tdir <- here("data", "tests")
scrna <- here(tdir, "scr_ref")
ut <- import("too_predict.utils")
## ad <- import("anndata")

adata <- ut$training_data_internal_test()

group <- adata$obs$tumor_type
batch <- adata$obs$Sample_Type == "organoid"
full_mod <- TRUE

batchmod <- model.matrix(~batch) # colnames: levels(batch)
# covariate
group <- as.factor(group)
n_group <- nlevels(group) # number of groups
if (full_mod && nlevels(group) > 1) {
  cat("Using full model in ComBat-seq.\n")
  mod <- model.matrix(~ 0 + group) # model.matrix(~0+group)
} else {
  cat("Using null model in ComBat-seq.\n")
  mod <- model.matrix(~1, data = as.data.frame(t(counts)))
}


## {
##   ad$AnnData(X = t(res$bulk.prop))
## }



# [2025-04-23 Wed] Want only some top-level reactome pathways

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
