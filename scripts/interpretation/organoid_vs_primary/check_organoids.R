suppressMessages({
  library(here)
  library(tidyverse)
  library(glue)
  library(edgeR)
  library(scRNAseq)
  library(paletteer)
  library(zellkonverter)
})

if (path.expand("~") != "/home/shannc") {
  adata_fn <- function() ut$training_data_internal()
} else {
  adata_fn <- function() ut$training_data_internal_test()
}

library(reticulate)
use_condaenv("too-predict")
source(here("src", "R", "utils.R"))

tf <- import("too_predict.filter")
ut <- import("too_predict.utils")

outdir_o <- here("data", "output", "chula_organoid_comparison")
outdir <- here("data", "output", "organoid_feature_selection")
outdir_bl <- here("data", "output", "feature_selection", "blacklists")
dir.create(outdir)

TMP <- ut$ref_feature_lists_internal()
REFS <- TMP[[2]]$edgeR_median_lfc_feature_list_3000


# %%
lihc_cases <- c("CHULA_LIHC", "TCGA_LIHC")
chol_cases <- c("CHULA_CHOL", "TCGA_CHOL")
coad_cases <- c("CHULA_COAD_READ", "TCGA_COAD_READ")
paad_cases <- c("CHULA_PAAD", "TCGA_PAAD")
all_cases <- c(lihc_cases, chol_cases, coad_cases, paad_cases)

TTYPES <- c("PAAD", "COAD_READ", "LIHC", "CHOL")

get_none_sce <- function(save_to) {
  print("Filtering from training data internal...")
  adata <- adata_fn()
  filter <- tf$Filter(REFS, feature_col = "GENEID")
  adata <- filter$fit_transform(adata)
  adata$write_h5ad(save_to)
  readH5AD(save_to)
}


## * Compare fold changes using transformed values

get_clr_adata <- function(save_to) {
  adata <- adata_fn()
  filter <- tf$Filter(REFS, feature_col = "GENEID")
  tt <- import("too_predict.transformer")
  ti <- import("too_predict.imputer")
  impute <- ti$Imputer("plus_one")
  tra <- tt$Transformer("clr", impute, inplace = FALSE)
  adata <- filter$fit_transform(adata)
  adata <- tra$fit_transform(adata)
  adata$write_h5ad(save_to)
  adata
}

compare_fold_changes <- function(f) {
  sc <- import("scanpy")
  ad <- import("anndata")

  sce_file <- here(
    "data", "output", "normalization_comparison", "edgeR_median_lfc_feature_list_3000",
    "clr-plus_one.h5ad"
  )
  adata <- read_existing(sce_file, get_clr_adata, ad$read_h5ad)
  adata$obs$tumor_type <- str_replace(adata$obs$tumor_type, "-", "_")
  wanted_types <- c("LIHC", "CHOL", "COAD_READ", "PAAD")
  wanted_sample <- c("organoid", "primary")

  adata <- adata[(adata$obs$tumor_type %in% wanted_types) & (adata$obs$Sample_Type %in% wanted_sample), ]

  compared_fc <- lapply(wanted_types, \(x) {
    current <- adata[adata$obs$tumor_type == x, ]
    print(glue("Current tumor type {x}"))
    print(current)
    # Compare fold change in organoids vs primary for each tumor type
    sc$tl$rank_genes_groups(current, "Sample_Type", method = "wilcoxon", reference = "primary")
    sc_results <- current$uns$rank_genes_groups
    sc_results |>
      within(rm(params)) |>
      as_tibble() |>
      mutate(across(where(is.matrix), \(x) x[, 1])) |>
      mutate(tumor_type_comparison = x, is_sig = pvals_adj < 0.01)
  }) |>
    bind_rows()

  sig_stats <- table(compared_fc$is_sig, compared_fc$tumor_type_comparison) |> table2tb("is_significant")

  summarized <- compared_fc |>
    group_by(tumor_type_comparison) |>
    summarise(
      var_abs_lfc = var(abs(logfoldchanges), na.rm = TRUE),
      min_lfc = min(logfoldchanges, na.rm = TRUE),
      max_lfc = max(logfoldchanges, na.rm = TRUE),
      median_abs_lfc = median(abs(logfoldchanges), na.rm = TRUE),
      sd_abs_lfc = sd(abs(logfoldchanges), na.rm = TRUE),
      avg_abs_lfc = mean(abs(logfoldchanges), na.rm = TRUE),
      n_significant = sum(is_sig)
    )

  write_csv(sig_stats, here(outdir_o, "lfc_significant.csv"))
  write_csv(summarized, here(outdir_o, "lfc_summary_absolute.csv"))
  write_csv(compared_fc, f)
}

all_fc <- read_existing(here(outdir_o, "genes_all_lfc.csv"), compare_fold_changes, read_csv)

## ** Importances
# Importance metric is tree feature importance
nz_file <- here("data", "output", "feature_selection", "nonzero_features.csv")
nonzero <- read_csv(nz_file)

joined <- inner_join(all_fc, select(nonzero, GENEID, importance), by = join_by(x$names == y$GENEID))

joined |> ggplot(aes(x = tumor_type_comparison, y = log(importance), fill = is_sig)) +
  geom_boxplot()

imp_plot <- joined |> ggplot(aes(x = logfoldchanges, y = log(importance), color = is_sig)) +
  geom_point() +
  facet_wrap(~tumor_type_comparison)
ggsave(here(outdir_o, "tree_importance_plot.png"))

# [2025-03-19 Wed] What you're trying to see here is whether or not lihc and chol
# have different lfcs vs primary in the more important features
# but this doesn't seem to be the case

## ** LFC between stypes vs. between ttypes
# %%
make_plot <- function(tb, label, palette = "ggthemes::Blue-Teal") {
  tb |> ggplot(aes(x = abs(fc_organoid), y = abs(fc_all), color = abs_ratio)) +
    geom_point() +
    xlab(glue("Absolute LFC: {label} organoid vs. {label} primary")) +
    ylab(glue("Absolute LFC: {label} vs. rest")) +
    labs(title = glue("Tumor type: {label}")) +
    scale_color_paletteer_c(palette)
}

between_type_comparison <- function(label, between_stype, between_ttype) {
  lowered <- str_to_lower(label)

  fc_all <- glue("logFC_tumor_type{label}")
  fc_organoid <- glue("{lowered}_logFC")
  cur <- between_stype |>
    select(all_of(c(
      "GENEID", fc_organoid,
      glue("{lowered}_PValue"),
      glue("{lowered}_F")
    ))) |>
    inner_join(select(
      between_ttype,
      all_of(c("GENEID", "PValue", "mean_counts", glue("logFC_tumor_type{label}")))
    ), by = join_by(GENEID)) |>
    dplyr::rename(all_of(c(fc_organoid = fc_organoid, fc_all = fc_all))) |>
    mutate(abs_ratio = abs(fc_all) - abs(fc_organoid)) # Want to maximize this ratio
  # TODO: is there a better way of doing this than using abs?

  # Let's identify statistically significant thresholds

  plot <- make_plot(cur, label)
  ggsave(here(outdir_o, glue("between_types_{label}.png")), plot, height = 8, width = 8)
  cur |>
    mutate(label = label) |>
    select(label, fc_organoid, fc_all, abs_ratio, GENEID)
}

# %%
at_least <- 8
quantile_threshold <- 0.40

formatted <- lapply(TTYPES, \(x) {
  between_type_comparison(x,
    between_stype = edger_results,
    between_ttype = edger_all
  )
}) |>
  bind_rows() |>
  group_by(label) |>
  mutate(
    bin = ntile(abs_ratio, n = 10),
    pass_q_threshold = abs_ratio > quantile(abs_ratio, quantile_threshold)
  ) |>
  ungroup()

filtered <- formatted |>
  group_by(label, GENEID) |>
  filter(pass_q_threshold) |>
  group_by(GENEID) |>
  summarise(across(where(is.numeric), mean))
discarded <- formatted$GENEID |>
  discard(\(x) x %in% filtered$GENEID) |>
  unique()

writeLines(
  discarded,
  here(outdir_bl, "edgeR_median_lfc_feature_list_3000-high_organoid_lfc.txt")
)

summarized_results <- edger_results |>
  select(GENEID, contains("logFC"), contains("PValue")) |>
  pivot_longer(-GENEID) |>
  mutate(
    group = map_chr(name, \(x) str_extract(x, "(.*)_", group = 1)),
    name = str_remove_all(name, ".*_")
  ) |>
  group_by(group) |>
  pivot_wider(id_cols = c(GENEID, group)) |>
  filter(PValue < 0.05) |>
  summarise(
    abs_mean = mean(abs(logFC)),
    abs_median = median(abs(logFC)),
    abs_var = var(abs(logFC)),
    min = min(logFC),
    max = max(logFC),
  )
write_csv(summarized_results, here(outdir_o, "edgeR_chula_summary.csv"))
