library(tidyverse)
library(here)
library(plotly)
library(glue)
Sys.setenv("RETICULATE_PYTHON" = here(".venv", "bin", "python"))
library(reticulate)

sc <- import("scanpy")
ad <- import("anndata")


outdir <- here("data", "output", "find_overlapping")
chosen_feature_set <- "edgeR_median_lfc_feature_list_3000" # Selected because of its
# high metrics in the normalization comparision results
prefix <- "plus_one_clr"

adata <- ad$read_h5ad(here(
  "data", "output", "normalization_comparison",
  "edgeR_median_lfc_feature_list_3000",
  "clr-plus_one.h5ad"
))

## sc$pl$umap(adata,
##   color = c("primary_site"), add_outline = TRUE,
##   legend_loc = "on data"
##   )

umap <- adata$obsm[["X_umap"]]
pca <- adata$obsm[["X_pca"]]

## plot_ly(x = umap[, 1], y = umap[, 2], color = adata$obs$Project_ID)
## plot_ly(x = pca[, 1], y = pca[, 2], color = adata$obs$Project_ID)


pairs <- read_csv(here(outdir, chosen_feature_set, glue("{prefix}_tomek_link_pairs.csv"))) |>
  separate_wider_delim(pair, delim = ",", names = c("x", "y")) |>
  mutate(across(where(is.character), \(x) {
    str_remove_all(x, "[\\[\\]\' ]")
  }))


link_stats <- read_csv(here(outdir, chosen_feature_set, glue("{prefix}_tomek_links.csv"))) |>
  arrange(desc(percentage))

full_links <- link_stats |> filter(percentage == 100)
pairs |>
  filter(x %in% full_links$class | y %in% full_links$class) |>
  print(n = 100)

print(link_stats, n = 100)

matrix <- read_csv(here(outdir, chosen_feature_set, glue("{prefix}_tomek_matrix.csv")))
