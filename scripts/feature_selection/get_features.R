library(ComplexHeatmap)
library(here)
library(glue)
library(ggVennDiagram)
library(tidyverse)
library(reticulate)
use_condaenv("too-predict")
library(ggridges)

source(here("src", "R", "utils.R"))
source(here("src", "R", "plotting.R"))

rs <- import("too_predict._rust_helpers")

fs_dir <- here("data", "output", "feature_selection")
fs_lists <- here(fs_dir, "feature_lists")
ref_lists <- here(fs_dir, "reference_lists")
n_dir <- here("data", "output", "normalization_comparison")

obs_meta <- read_csv(here("data", "training_data_obs.csv"))
gene_meta <- read_csv(here("data", "training_data_var.csv"))
tumor_types <- unique(obs_meta$tumor_type)

## Plotting and filtering results
vtb <- read_csv(here(fs_dir, "sklearn_low_variance.csv"))
vtb_go <- read_csv(here(fs_dir, "sklearn_variance_GO.csv"))
minfo <- read_csv(here(fs_dir, "mutual_info.csv")) |>
  filter(!is.na(feature)) |>
  inner_join(gene_meta, by = join_by(x$feature == y$GENEID)) |>
  rename(GENEID = feature)
edger <- read_tsv(here(fs_dir, "edgeR_top_types.tsv"))

# pct_dropout_by_counts: Percentage of cells the feature doesn't appear in
max_dropout_pct <- 10 # Don't want genes that are missing in > 90% of samples

# To facilitate comparison, will create a common column `value` containing the feature
# selection metric
minfo <- minfo |>
  filter(pct_dropout_by_counts <= max_dropout_pct) |>
  filter(!is.na(GENEID)) |>
  mutate(value = mutual_info)
vtb <- vtb |>
  filter(pct_dropout_by_counts <= max_dropout_pct) |>
  filter(!is.na(GENEID)) |>
  mutate(value = variance)
edger <- edger |> filter(pct_dropout_by_counts <= max_dropout_pct)
edger$median_lfc <- apply(
  select(edger, contains("logFC_tumor_type")),
  1,
  median
)
edger <- edger |>
  relocate(median_lfc, .after = SEQNAME) |>
  filter(!is.na(GENEID)) |>
  mutate(value = abs(median_lfc))

feature_tbs <- list(
  edgeR_median_lfc = edger,
  variance = vtb,
  mutual_info = minfo
)


## * Visualize feature distribution

features_together <- lapply(names(feature_tbs), \(x) {
  if (x != "variance_go") {
    select(feature_tbs[[x]], GENEID, value) |>
      mutate(value = scale(value, center = FALSE)) |>
      rename(!!as.symbol(x) := value) |>
      filter(!is.na(GENEID))
  }
}) |>
  reduce(\(x, y) full_join(x, y, by = join_by(GENEID))) |>
  pivot_longer(cols = -GENEID, names_to = "metric")

feature_dist_plot <- ggplot(features_together, aes(x = value, color = metric)) +
  geom_density() +
  xlab("Scaled value")
ggsave(here(fs_dir, "feature_dist.png"), feature_dist_plot, width = 10)

n_features <- 500

## * Get features

# %%

top_n_features <- lapply(names(feature_tbs), \(x) {
  features <- feature_tbs[[x]] |>
    arrange(desc(value)) |>
    head(n = n_features) |>
    pluck("GENEID")
  writeLines(
    features,
    here(fs_lists, glue("{x}_feature_list_{n_features}.txt"))
  )
  features
}) |>
  `names<-`(names(feature_tbs))
feature_venn <- ggVennDiagram(top_n_features)

# %%

## ** edgeR by type
# Instead of aggregating by median lfc, will get disjoint feature sets for each tumor
# type
ttypes <- colnames(edger) |>
  keep(\(x) str_detect(x, "logFC_tumor_type")) |>
  str_remove("logFC_tumor_type")
n_per <- 50
seen <- c()
pybuiltins <- import_builtins()
type_overlap_list <- list()
edger_type_flist <- lapply(ttypes, \(type) {
  fc_col <- glue("logFC_tumor_type{type}")
  sorted <- arrange(edger, desc(abs(!!as.symbol(fc_col)))) |>
    mutate(from = type)
  type_overlap_list[[type]] <<- pybuiltins$set(sorted$GENEID[1:n_per])
  filtered <- sorted |>
    filter(!GENEID %in% seen) |>
    slice_head(n = n_per)
  seen <<- c(seen, sorted$GENEID[1:n_per])
  filtered
}) |>
  bind_rows()
overlaps <- rs$pairwise_overlaps(type_overlap_list, FALSE) |>
  lapply(\(x) {
    tibble(first = x[[1]][[1]], second = x[[1]][[2]], overlap = x[[2]])
  }) |>
  bind_rows() |>
  arrange(desc(overlap))
writeLines(
  edger_type_flist$GENEID,
  here(fs_lists, glue("edgeR_{n_per}_per_type.txt"))
)
# [2025-05-19 Mon] 725 features in common with top 3000 median list
#     417 in common with the top one thousand

## ** edgeR by type, considering organoid differences
# Like the above, but do not accept features that are heavily DE in an
# ovp (organoid vs primary) comparison
# %%
ovp_tb <- read_tsv(here(
  "data",
  "output",
  "chula_organoid_comparison",
  "de_enrichment",
  "sample_type_top_tags.tsv"
))
tissue_enriched <- read_csv(here(
  "data",
  "reference",
  "hpa_tissue_enriched_2025-5-20.csv"
)) |>
  mutate(tissue = str_replace_all(tissue, " ", "_"))

# %%
seen_ovp <- c()
with_p_value <- TRUE
# [2025-05-19 Mon] TODO: figure out why using only the ratio method fails
with_tissue_enriched <- FALSE
n_per <- 70
source(here("data", "mappings", "misc_mappings.R"))
blacklist <- ovp_tb |>
  filter(PValue >= 0.01) |>
  pull(GENEID)
if (with_tissue_enriched) {
  add_name <- "tissue_enriched"
} else {
  add_name <- ""
}
if (!with_p_value) {
  add_name <- glue("{add_name}_ratio_only")
}
ovp_fs_file <- here(fs_lists, glue("edgeR_{n_per}_per_type_ovp_{add_name}.txt"))

edger_type_flist_ovp <- lapply(ttypes, \(type) {
  fc_col <- glue("logFC_tumor_type{type}")
  sorted <- arrange(edger, desc(abs(!!as.symbol(fc_col)))) |>
    filter(!GENEID %in% seen) |>
    mutate(from = type) |>
    inner_join(ovp_tb, by = join_by(GENEID), suffix = c("", ".ovp")) |>
    mutate(
      logFC = replace(logFC, logFC == 0, 0.000001),
      de_ratio = abs(!!as.symbol(fc_col)) / abs(logFC)
    )
  if (with_tissue_enriched) {
    # Prioritize genes that are tissue enriched
    cur_tissue_enriched <- tissue_enriched |>
      filter(tissue %in% ttype2tissue[[type]]) |>
      pull(Ensembl)
    sorted <- left_join(
      sorted,
      tissue_enriched,
      by = join_by(x$GENEID == y$Ensembl)
    ) |>
      mutate(
        is_tissue_enriched = case_when(
          GENEID %in% cur_tissue_enriched ~ 1,
          .default = 0
        )
      ) |>
      arrange(desc(is_tissue_enriched), desc(abs(!!as.symbol(fc_col))))
  }
  if (with_p_value) {
    # Only keep non-DE genes in the organoid_vs_primary comparison
    passing_p <- sorted |>
      filter(!GENEID %in% blacklist) |>
      slice_head(n = n_per)
    if (nrow(passing_p) != n_per) {
      remaining <- n_per - nrow(passing_p)
      sorted <- sorted |>
        slice_max(order_by = de_ratio, n = remaining, with_ties = FALSE)
      final <- bind_rows(sorted, passing_p)
    } else {
      final <- passing_p
    }
  } else {
    final <- sorted |>
      slice_max(order_by = de_ratio, n = n_per, with_ties = FALSE)
  }
  seen_ovp <<- c(seen_ovp, final$GENEID)
  final
}) |>
  bind_rows()
writeLines(edger_type_flist_ovp$GENEID, ovp_fs_file)
# %%

stop("Done")

## ** Gene ontology

n_ontology <- 500
go_top_n <- vtb_go |>
  group_by(`GO domain`) |>
  nest() |>
  filter(!is.na(`GO domain`)) |>
  mutate(
    data = lapply(data, \(x) head(arrange(x, desc(variance)), n = n_ontology))
  ) |>
  unnest(cols = c(data)) |>
  ungroup()

## head(n = n_ontology)

## writeLines(
##   go_top_n$`GO term accession`,
##   here(ref_lists, glue("variance_go_feature_list_{n_ontology * 3}.txt"))
## )
## write_csv(go_top_n, here(fs_dir, glue("variance_go_{n_ontology * 3}.csv")))

## ggsave(here(fs_dir, glue("selected_ml_features_overlap_{n_features}.png")), feature_venn)

## ** Overlap between top DE genes between tumor types
top_n_de <- round(n_features / length(tumor_types))

# Get the top n features for each tumor type, formatting as a type x feature
# matrix to compute distance with
top_by_types <- lapply(tumor_types, \(x) {
  col <- glue("logFC_tumor_type{x}")
  if (col %in% colnames(edger)) {
    subset <- edger[, "GENEID"]
    subset[[x]] <- 1
    subset[order(abs(edger[[col]]), decreasing = TRUE), ][1:top_n_de, ] |>
      distinct(GENEID, .keep_all = TRUE)
  }
}) |>
  discard(is.null) |>
  reduce(\(x, y) full_join(x, y, by = join_by(GENEID))) |>
  mutate(across(where(is.double), \(x) replace_na(x, 0)))

type_x_feature <- t(top_by_types[, -1]) |>
  as.data.frame() |>
  `colnames<-`(top_by_types$GENEID)

jaccard <- vegan::vegdist(type_x_feature, method = "jaccard") |>
  as.matrix() |>
  as.data.frame() |>
  rownames_to_column(var = "x") |>
  as_tibble() |>
  pivot_longer(-x, names_to = "y", values_to = "value") |>
  mutate(value = round(value, 2))


## * Get features for ALR
n_alr <- 20 # Number of alr features
alr_quantile <- 20 / nrow(vtb) # Quantile to get these features

bottom_n_features <- lapply(names(feature_tbs), \(x) {
  features <- feature_tbs[[x]] |>
    arrange(value) |>
    head(n = n_alr) |>
    pluck("GENEID")
  writeLines(
    features,
    here(ref_lists, glue("{x}_feature_list_lowest_{n_alr}.txt"))
  )
  features
}) |>
  `names<-`(names(feature_tbs))

feature_venn_b <- ggVennDiagram(bottom_n_features)
ggsave(here(fs_dir, "lowest_features_overlap.png"), feature_venn_b)

## * Normalization metrics plotting
n_metrics <- read_csv(here(n_dir, "label_metrics.csv")) |>
  mutate(across(where(is.double), scale)) |>
  pivot_longer(
    cols = where(is.double),
    names_to = "metric",
    values_to = "value"
  )

n_metrics <- n_metrics |> filter(!grepl("feature", normalization))

n_metric_plot <- ggplot(
  n_metrics,
  aes(x = feature_set, y = value, fill = normalization)
) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~metric)
n_metric_plot
ggsave(here(n_dir, "metrics.png"), n_metric_plot)

metric_rankings <- list(
  silhouette_score = TRUE,
  davies_bouldin_score = FALSE,
  calinski_harabasz_score = TRUE
)

to_rank <- n_metrics |>
  filter(label == "tumor_type") |>
  mutate(combination = paste0(feature_set, "_", normalization)) |>
  select(combination, metric, value) |>
  pivot_wider(names_from = metric, values_from = value)

ranked <- rank_by_metrics("combination", NULL, to_rank, metric_rankings)
write_csv(ranked$table, here(n_dir, "ranked_combinations.csv"))

## * Organoid features

orf <- read_csv(here(
  "data",
  "output",
  "organoid_feature_selection",
  "chula_tcga_dge.csv"
))
# Find common
log_fcs <- orf |> select(GENEID, contains("logFC"))


## * edgeR go
# %%

get_edger_go <- function() {
  n_type <- 40
  min_count <- 5 # For robustness to missing genes
  edger_go <- read_tsv(here(fs_dir, "edgeR_top_types_GO.tsv"))
  if (!"GO term accession" %in% colnames(edger_go)) {
    go_map <- read_csv(here("data", "mappings", "go_names2acc.csv"))
    edger_go <- edger_go |>
      distinct(GO.term.name, .keep_all = TRUE) |>
      inner_join(go_map, by = join_by(x$GO.term.name == y$`GO term name`)) |>
      relocate(`GO term accession`, .before = everything())
  }
  lfc_re <- "_tumor_type"
  lfc_groups <- keep(colnames(edger_go), \(x) str_detect(x, "logFC"))
  print(glue("Total GO features: {length(lfc_groups) * n_type}"))
  tracker <- list() # Want to see the overlap between the top D go terms and tumor types
  seen_acc <- c() # Make sure we don't get duplicate accs
  combined_gos <- lapply(lfc_groups, \(x) {
    filtered <- edger_go |>
      filter(PValue < 0.01) |>
      mutate(sort = abs(!!as.symbol(x))) |>
      arrange(desc(sort)) |>
      select(`GO term accession`, sort)

    tracker[[x]] <<- head(filtered$`GO term accession`, n = n_type)

    new_gos <- filtered |> filter(!`GO term accession` %in% seen_acc)
    top_n <- head(new_gos$`GO term accession`, n = n_type)
    seen_acc <<- c(seen_acc, top_n)
    top_n
  }) |>
    unlist()
  writeLines(
    combined_gos,
    here(fs_lists, glue("edgeR_go_feature_list_{n_type}.txt"))
  )

  # Plot jaccard similarity to visualize overlap
  fc_x_go <- as_tibble(tracker) |>
    pivot_longer(everything()) |>
    pivot_wider(
      names_from = value,
      id_cols = name,
      values_fn = \(x) 1,
      values_fill = 0
    )

  fc_dist <- vegan::vegdist(
    select(fc_x_go, where(is.numeric)),
    method = "jaccard"
  )
}

dist_heatmap <- function(dist, vars, var_name = "feature") {
  tb <- dist |>
    as.matrix() |>
    as_tibble() |>
    `colnames<-`(vars)
  tb[[var_name]] <- as.factor(vars)
  tb <- tb |>
    pivot_longer(-!!as.symbol(var_name), names_to = "y") |>
    mutate(y = as.factor(y))
  tb |>
    ggplot(aes(x = !!as.symbol(var_name), y = as.factor(y), fill = value)) +
    geom_tile() +
    xlab(var_name) +
    ylab(var_name)
}
dist_heatmap(fc_dist, lfc_groups)
