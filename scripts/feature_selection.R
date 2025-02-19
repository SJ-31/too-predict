library(BiocParallel)
library(ALDEx2)
library(tidyverse)
library(zellkonverter)
library(scRNAseq)
library(here)

register(MulticoreParam(workers = 2))
source(here("src", "R", "utils.R"))

data <- readH5AD(here("data", "tests", "TCGA_CESC-DLBC-ESCA-GBM.h5ad"))
# TODO: replace this with the complete dataset

# TODO: can include the sequencing tech and the tumor type as factors to account
# for their effects
## --- CODE BLOCK ---
p_threshold <- 0.05
group <- "Project_ID"
technical_factors <- c("Sample_Type")

n <- 50 # n features to get
data <- data[1:50, ]
result <- aldex_glm_wrapper(data, group, technical_factors, use_parallel = TRUE)
# TODO: maybe use the scale aware version
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
