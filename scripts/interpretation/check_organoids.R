suppressMessages({
  library(here)
  library(tidyverse)
  library(glue)
  library(edgeR)
  library(ggridges)
  library(scRNAseq)
  library(zellkonverter)
  Sys.setenv("RETICULATE_PYTHON" = here(".venv", "bin", "python"))
  library(reticulate)
  source(here("src", "R", "utils.R"))
})

outdir_o <- here("data", "output", "chula_organoid_comparison")
outdir <- here("data", "output", "organoid_feature_selection")
dir.create(outdir)


## --- CODE BLOCK ---
lihc_cases <- c("CHULA_LIHC", "TCGA_LIHC")
chol_cases <- c("CHULA_CHOL", "TCGA_CHOL")
coad_cases <- c("CHULA_COAD_READ", "TCGA_COAD_READ")
all_cases <- c(lihc_cases, chol_cases, coad_cases)

edger_analysis <- function(file) {
  sce_file <- here(
    "data", "output", "normalization_comparison", "edgeR_median_lfc_feature_list_3000",
    "none-plus_one.h5ad"
  )
  sce <- readH5AD(sce_file)

  rowData(sce) <- rowData(sce) |>
    as_tibble() |>
    select(-contains("varm.PCs"))

  colData(sce)$Project_ID <- case_match(colData(sce)$Project_ID,
    c("TCGA-COAD", "TCGA-READ") ~ "TCGA-COAD-READ",
    .default = colData(sce)$Project_ID
  ) |> str_replace_all("-", "_")

  sce <- sce[, colData(sce)$Project_ID %in% all_cases]
  ## lihc <- sce[, colData(sce)$Project_ID %in% lihc_cases]
  dge <- sce2dge(sce)

  normLibSizes(dge)
  mm <- model.matrix(~ 0 + Project_ID, dge$samples)

  contrasts <- list(
    lihc = c(0, 0, 1, 0, 0, -1),
    chol = c(1, 0, 0, -1, 0, 0),
    coad_read = c(0, 1, 0, 0, -1, 0)
  )

  dge <- estimateDisp(dge, design = mm, robust = TRUE)

  added_cols <- c("logFC", "logCPM", "F", "PValue", "FDR")
  other_cols <- colnames(rowData(sce))

  get_tags <- function(qlf, count) {
    topTags(qlf, n = count, sort.by = "PValue") |>
      as.data.frame() |>
      as_tibble() |>
      relocate(all_of(added_cols), .after = "GENEID")
  }

  fit <- glmQLFit(dge, mm)

  top_tags <- lapply(names(contrasts), \(n) {
    qlf <- glmQLFTest(fit, contrast = contrasts[[n]])
    message(glue("Comparison: {qlf$comparison}"))
    get_tags(qlf, count = nrow(dge)) |> rename_with(\(cols) {
      map_chr(cols, \(x) {
        if (x %in% added_cols) {
          glue("{n}_{x}")
        } else {
          x
        }
      })
    })
  }) |>
    purrr::reduce(\(x, y) inner_join(x, y, by = other_cols))

  write_csv(top_tags, file)
}

edger_file <- here(outdir, "chula_tcga_dge.csv")
edger_results <- read_existing(edger_file, edger_analysis, read_csv)

## * Compare fold changes using transformed values

compare_fold_changes <- function(f) {
  sc <- import("scanpy")
  ad <- import("anndata")

  sce_file <- here(
    "data", "output", "normalization_comparison", "edgeR_median_lfc_feature_list_3000",
    "clr-plus_one.h5ad"
  )
  adata <- ad$read_h5ad(sce_file)
  wanted_types <- c("LIHC", "CHOL", "COAD_READ")
  wanted_sample <- c("organoid", "primary")

  adata <- adata[(adata$obs$tumor_type %in% wanted_types) & (adata$obs$Sample_Type %in% wanted_sample), ]

  compared_fc <- lapply(wanted_types, \(x) {
    current <- adata[adata$obs$tumor_type == x, ]
    # Compare fold change in organoids vs primary for each tumor type
    sc$tl$rank_genes_groups(current, "Sample_Type", method = "wilcoxon", reference = "primary")
    sc_results <- current$uns$rank_genes_groups
    print(sc_results$params)
    sc_results |>
      within(rm(params)) |>
      as_tibble() |>
      mutate(across(where(is.matrix), \(x) x[, 1])) |>
      mutate(tumor_type_comparison = x, is_sig = pvals_adj < 0.01)
  }) |>
    bind_rows()

  sig_stats <- table(compared_fc$is_sig, compared_fc$tumor_type_comparison) |> table2tb("is_significant")

  fc_plot <- compared_fc |> ggplot(aes(x = logfoldchanges, fill = tumor_type_comparison, y = tumor_type_comparison)) +
    geom_density_ridges()

  summarized <- compared_fc |>
    group_by(tumor_type_comparison) |>
    summarise(
      var_lfc = var(logfoldchanges, na.rm = TRUE),
      min_lfc = min(logfoldchanges, na.rm = TRUE),
      max_lfc = max(logfoldchanges, na.rm = TRUE),
      sd_lfc = sd(logfoldchanges, na.rm = TRUE),
      avg_lfc = mean(logfoldchanges, na.rm = TRUE),
      n_significant = sum(is_sig)
    )

  write_csv(sig_stats, here(outdir_o, "lfc_significant.csv"))
  write_csv(summarized, here(outdir_o, "lfc_summary.csv"))
  write_csv(compared_fc, f)
}

all_fc <- read_existing(here(outdir_o, "genes_all_lfc.csv"), compare_fold_changes, read_csv)

nz_file <- here("data", "output", "feature_selection", "nonzero_features.csv")
nonzero <- read_csv(nz_file)

joined <- inner_join(all_fc, select(nonzero, GENEID, importance), by = join_by(x$names == y$GENEID))

joined |> ggplot(aes(x = tumor_type_comparison, y = log(importance), fill = is_sig)) +
  geom_boxplot()

imp_plot <- joined |> ggplot(aes(x = logfoldchanges, y = log(importance), color = is_sig)) +
  geom_point() +
  facet_wrap(~tumor_type_comparison)

# [2025-03-19 Wed] What you're trying to see here is whether or not lihc and chol
# have different lfcs vs primary in the more important features
# but this doesn't seem to be the case
