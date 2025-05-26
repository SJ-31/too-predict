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
