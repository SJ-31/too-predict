library(here)
library(glue)
library(ggVennDiagram)
library(tidyverse)
library(ggridges)

source(here("src", "R", "utils.R"))
source(here("src", "R", "plotting.R"))

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
edger <- read_tsv(here(fs_dir, "edgeR_top_types_backup.tsv"))

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
edger$median_lfc <- apply(select(edger, contains("logFC_tumor_type")), 1, median)
edger <- edger |>
  relocate(median_lfc, .after = SEQNAME) |>
  filter(!is.na(GENEID)) |>
  mutate(value = abs(median_lfc))

feature_tbs <- list(edgeR_median_lfc = edger, variance = vtb, mutual_info = minfo)

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

n_features <- 1000

## * Get features

top_n_features <- lapply(names(feature_tbs), \(x) {
  features <- feature_tbs[[x]] |>
    arrange(desc(value)) |>
    head(n = n_features) |>
    pluck("GENEID")
  writeLines(features, here(fs_lists, glue("{x}_feature_list_{n_features}.txt")))
  features
}) |> `names<-`(names(feature_tbs))
feature_venn <- ggVennDiagram(top_n_features)

n_ontology <- 500
go_top_n <- vtb_go |>
  group_by(`GO domain`) |>
  nest() |>
  filter(!is.na(`GO domain`)) |>
  mutate(data = lapply(data, \(x) head(arrange(x, desc(variance)), n = n_ontology))) |>
  unnest(cols = c(data)) |>
  ungroup()

## head(n = n_ontology)

writeLines(
  go_top_n$`GO term accession`,
  here(ref_lists, glue("variance_go_feature_list_{n_ontology * 3}.txt"))
)
write_csv(go_top_n, here(fs_dir, glue("variance_go_{n_ontology * 3}.csv")))

ggsave(here(fs_dir, glue("selected_ml_features_overlap_{n_features}.png")), feature_venn)

## ** Overlap between top DE genes between tumor types
top_n_de <- round(n_features / length(tumor_types))

# Get the top n features for each tumor type, formatting as a type x feature
# matrix to compute distance with
top_by_types <- lapply(tumor_types, \(x) {
  col <- glue("logFC_tumor_type{x}")
  if (col %in% colnames(edger)) {
    subset <- edger[, "GENEID"]
    subset[[x]] <- 1
    subset[order(abs(edger[[col]]), decreasing = TRUE), ][1:top_n_de, ] |> distinct(GENEID, .keep_all = TRUE)
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

## distinct_orderings(jaccard, c("x", "y")) |>
##   ggplot(aes(x = as.factor(x), y = as.factor(y), fill = value)) +
##   geom_tile()

## ggplot(jaccard, aes(x = name, y = y, fill = value)) +
##   geom_tile() #

## * Get features for ALR
n_alr <- 20 # Number of alr features
alr_quantile <- 20 / nrow(vtb) # Quantile to get these features

bottom_n_features <- lapply(names(feature_tbs), \(x) {
  features <- feature_tbs[[x]] |>
    arrange(value) |>
    head(n = n_alr) |>
    pluck("GENEID")
  writeLines(features, here(ref_lists, glue("{x}_feature_list_lowest_{n_alr}.txt")))
  features
}) |> `names<-`(names(feature_tbs))

feature_venn_b <- ggVennDiagram(bottom_n_features)
ggsave(here(fs_dir, "lowest_features_overlap.png"), feature_venn_b)

## * Normalization metrics plotting
n_metrics <- read_csv(here(n_dir, "label_metrics.csv")) |>
  mutate(across(where(is.double), scale)) |>
  pivot_longer(cols = where(is.double), names_to = "metric", values_to = "value")

n_metrics <- n_metrics |> filter(!grepl("feature", normalization))

n_metric_plot <- ggplot(n_metrics, aes(x = feature_set, y = value, fill = normalization)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~metric)
n_metric_plot
ggsave(here(n_dir, "metrics.png"), n_metric_plot)

metric_rankings <- list(
  silhouette_score = TRUE, davies_bouldin_score = FALSE,
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

orf <- read_csv(here("data", "output", "organoid_feature_selection", "chula_tcga_dge.csv"))
# Find common
log_fcs <- orf |> select(GENEID, contains("logFC"))


## * edgeR go
## --- CODE BLOCK ---

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
  writeLines(combined_gos, here(fs_lists, glue("edgeR_go_feature_list_{n_type}.txt")))

  # Plot jaccard similarity to visualize overlap
  fc_x_go <- as_tibble(tracker) |>
    pivot_longer(everything()) |>
    pivot_wider(names_from = value, id_cols = name, values_fn = \(x) 1, values_fill = 0)

  fc_dist <- vegan::vegdist(select(fc_x_go, where(is.numeric)), method = "jaccard")
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


# Need a function to plot dist


## --- CODE BLOCK ---
