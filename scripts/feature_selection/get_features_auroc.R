library(here)
library(glue)
library(tidyverse)
library(reticulate)
use_condaenv("too-predict")

source(here("src", "R", "utils.R"))
source(here("src", "R", "plotting.R"))

fs_dir <- here("data", "output", "feature_selection")
fs_lists <- here(fs_dir, "feature_lists")
ref_lists <- here(fs_dir, "reference_lists")
n_dir <- here("data", "output", "normalization_comparison")
ovp_tb <- read_tsv(here(
  "data",
  "output",
  "chula_organoid_comparison",
  "de_enrichment",
  "sample_type_top_tags.tsv"
))
auroc_tb <- read_csv(here(
  "data",
  "output",
  "feature_selection",
  "gene_auROC_scores.csv"
))

auroc_o_tb <- read_csv(here(
  "data",
  "output",
  "feature_selection",
  "gene_auROC_scores_chula_organoid.csv"
))

together <- inner_join(auroc_o_tb, auroc_tb, by = join_by(gene, target))

## together |>
##   filter(target == "PAAD") |>
##   ggplot(aes(x = AUROC.x, y = AUROC.y)) +
##   geom_point()

ttypes <- unique(auroc_tb$target)
blacklist <- ovp_tb |>
  filter(PValue >= 0.01) |>
  pull(GENEID)

n_per <- 70
filtered <- auroc_tb |> filter(!gene %in% blacklist)
seen <- c()
features <- lapply(ttypes, \(type) {
  current <- filtered |>
    filter(target == type & !gene %in% seen) |>
    slice_max(order_by = AUROC, n = n_per, with_ties = FALSE)
  seen <<- c(seen, current$gene)
  current$gene
}) |>
  unlist()
writeLines(
  features,
  here(fs_lists, glue("auroc_{n_per}_per_type_blacklist.txt"))
)
