suppressMessages({
  library(here)
  library(BiocParallel)
  library(glue)
  library(tidyverse)
  library(zellkonverter)
  library(scRNAseq)
  library(edgeR)
})


suffix <- ""
if (sys.nframe() == 0) {
  library("optparse")
  parser <- OptionParser()
  parser <- add_option(parser, c("-t", "--test"), type = "logical", default = FALSE, action = "store_true")
  parser <- add_option(parser, c("-c", "--cores"), type = "integer", default = 8)
  parser <- add_option(parser, c("-g", "--recode_go"), type = "logical", default = FALSE, action = "store_true")
  args <- parse_args(parser)
  python_path <- here("remote", "envs", "too-predict", "bin", "python")
  if (args$recode_go) {
    suffix <- "_GO"
  }
} else {
  python_path <- here(".venv", "bin", "python")
}
Sys.setenv("RETICULATE_PYTHON" = python_path)
library(reticulate)
source(here("src", "R", "utils.R"))

pyutils <- new.env()
source_python(here("src", "too_predict", "utils.py"), envir = pyutils)
outdir <- here("data", "output", "feature_selection")
if (args$test) {
  print("Using test subset")
  outdir <- here(outdir, "test")
  dir.create(outdir, recursive = TRUE)
  adata <- pyutils$training_data_internal_test()
} else {
  adata <- pyutils$training_data_internal()
}
if (args$recode_go) {
  adata <- pyutils$recode_to_go(adata)
}

data <- AnnData2SCE(adata)
rm(pyutils)

# TODO: can include the sequencing tech and the tumor type as factors to account
# for their effects
## --- CODE BLOCK ---
p_threshold <- 0.05
group <- "tumor_type"
## technical_factors <- c("Sample_Type", "Project_ID") TODO: need to address confounding
technical_factors <- NULL

counts <- assays(data)$X |> as.matrix()
mode(counts) <- "integer"
assays(data, withDimnames = FALSE)$X <- counts

## * DGE with ALDEx2

get_aldex <- function(f) {
  result <- aldex_glm_wrapper(data, group, technical_factors, use_parallel = TRUE)
  # TODO: maybe use the scale aware version by specifying gamma
  effect <- as_tibble(result$effect) # Effect size are standardized mean differences
  test <- as_tibble(result$test)

  # Take the average effects of all comparisons
  between_groups <- effect |> filter(str_starts(contrast, group))
  id_col <- test$gene_id
  tb_list <- between_groups |>
    group_by(contrast) |>
    select(where(is.numeric)) |>
    nest() |>
    pluck("data")
  averaged <- (purrr::reduce(tb_list, \(x, y) x + y) / length(tb_list)) |>
    as_tibble() |>
    mutate(
      gene_id = id_col,
      abs_effect = abs(effect)
    )

  # Features with least change across conditions,
  # possible candidates for ALR
  n_lowest <- averaged |>
    arrange(abs_effect) |>
    dplyr::slice(1:n)

  # Features with most change, for machine learning
  # Must be statistically significant across all comparisons
  significant <- test |>
    select(gene_id, contains(group) & contains("pval.padj")) |>
    filter(if_all(where(is.numeric), \(x) x <= p_threshold)) |>
    pluck("gene_id")

  greatest_change <- averaged |>
    filter(gene_id %in% significant) |>
    arrange(desc(abs_effect)) |>
    slice(1:1000)

  write_tsv(averaged, f)
  write_tsv(effect, here(outdir, "ALDEx2_all_effect.tsv"))
  write_tsv(test, here(outdir, "ALDEx2_test.tsv"))
}

aldex_average_file <- here(outdir, "ALDEx2_averaged_effect.tsv")
## aldex_average <- read_existing(aldex_average_file, get_aldex, read_tsv)
## <2025-02-28 Fri> OOM errors, even on the tiny test dataset...

## * With edgeR

# Goal: finding the top DEGs in each class
get_edgeR <- function(f) {
  dge <- DGEList(counts = counts, samples = colData(data), genes = rowData(data))
  normLibSizes(dge)
  factor_str <- paste0(c(group, technical_factors), collapse = " + ")
  mm <- model.matrix(as.formula(paste("~0+", factor_str)), data = colData(data))
  curious <- dge[, !(colnames(dge) %in% rownames(mm))]
  dge <- estimateDisp(dge, design = mm, robust = TRUE)
  fit <- glmQLFit(dge, mm, robust = TRUE)
  # Fit glm to account for batch effect specified above

  # Make contrasts to get fold changes in one class vs mean of other classes
  group_vec <- colnames(mm) |> keep(\(x) str_detect(x, group))
  mean_val <- 1 / (length(group_vec) - 1)

  contrast_str <- map_chr(group_vec, \(x) {
    mean_others <- paste(mean_val, "*", group_vec[group_vec != x], collapse = "+")
    paste0(x, "-", "(", mean_others, ")")
  })

  ccs <- makeContrasts(contrasts = contrast_str, levels = mm)
  test <- glmQLFTest(fit, contrast = ccs)

  top <- topTags(test, n = nrow(dge), sort.by = "PValue") |>
    as.data.frame() |>
    as_tibble()
  fc_names <- paste0("logFC_", group_vec)
  names(fc_names) <- keep(colnames(top), \(x) str_detect(x, "logFC"))
  top <- rename(top, all_of(fc_names))
  # logFC_<x> are the logFC of x vs the mean of all other classes
  write_tsv(top, f)
}

edgeR_top_file <- here(outdir, glue("edgeR_top_types{suffix}.tsv"))
edgeR_top <- read_existing(edgeR_top_file, get_edgeR, read_tsv)

## * CoDACore

## TODO
