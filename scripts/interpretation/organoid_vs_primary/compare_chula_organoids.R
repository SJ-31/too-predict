suppressMessages({
  library(here)
  library(tidyverse)
  library(ggridges)
  library(glue)
  library(edgeR)
  library(scRNAseq)
  library(broom)
  library(zellkonverter)
  source(here("src", "R", "utils.R"))
  source(here("src", "R", "plotting.R"))
  library(reticulate)
  use_condaenv("too-predict")
})

outdir <- here("data", "output", "chula_organoid_comparison")
dir.create(outdir)

sc <- import("scanpy")
ad <- import("anndata")

sce_file <- here(
  "data", "output", "normalization_comparison", "edgeR_median_lfc_feature_list_3000",
  "clr-plus_one.h5ad"
)
sce <- readH5AD(sce_file)

wanted_tumor_type <- c("LIHC", "COAD_READ", "CHOL", "COAD-READ", "PAAD")

sce <- sce[, colData(sce)$tumor_type %in% wanted_tumor_type]

adata <- ad$read_h5ad(sce_file)
adata <- adata[adata$obs$tumor_type %in% wanted_tumor_type, ]
adata$obs$from_chula <- grepl("CHULA", adata$obs$Project_ID)


if (!"pca" %in% adata$uns) {
  sc$pp$pca(adata)
  sc$pp$neighbors(adata)
  sc$tl$umap(adata)
  adata$write_h5ad(sce_file)
}

for (ttype in wanted_tumor_type) {
  cur <- adata[adata$obs$tumor_type == ttype, ]
  sc$pl$umap(cur,
    color = c("Sample_Type", "from_chula"),
    save = here(outdir, glue("umap-{ttype}.png"))
  )
  sc$pl$pca(cur,
    color = c("tumor_type", "from_chula"),
    save = here(outdir, glue("pca-{ttype}.png"))
  )
}

pheatmap_helper(
  sce = sce,
  order_on = "tumor_type",
  sample_annotations = list(
    tumor_type = "ggsci::light_uchicago",
    Sample_Type = "awtools::mpalette"
  ), pheatmap_kwargs = list(
    file = here(outdir, "heatmap.png"),
    show_rownames = FALSE,
    show_colnames = FALSE
  )
)

## * Compare within-label variances
# On lr-transformed data

unique_types <- unique(colData(sce)$tumor_type)
chula_only <- sce[, grepl("CHULA", colData(sce)$Project_ID)]

var_tb <- lapply(unique_types, \(x) {
  filtered <- chula_only[, colData(chula_only)$tumor_type == x]
  vars <- apply(assays(filtered)$X, 1, var) # Get the variance of each feature within subset `x`
  tmp <- list(feature = names(vars))
  tmp[[as.character(x)]] <- vars
  as_tibble(tmp)
}) |>
  purrr::reduce(\(x, y) inner_join(x, y, by = join_by(feature)))

# See if overall within-label variance has statistically significant differences
# between the tumor types
var_list <- select(var_tb, where(is.numeric)) |> as.list()
test_result <- kruskal.test(var_list)

# [2025-03-18 Tue] is signifcant
# now check if COAD-READ has lower variance that the other two

lihc_coad <- wilcox.test(x = var_list$COAD_READ, y = var_list$LIHC, alternative = "less", paired = TRUE) |> tidy()
chol_coad <- wilcox.test(x = var_list$COAD_READ, y = var_list$CHOL, alternative = "less", paired = TRUE) |> tidy()

# Plot
var_plot <- var_tb |>
  pivot_longer(cols = -feature) |>
  ggplot(aes(y = log(value), color = name)) +
  geom_boxplot() +
  ylab("log Variance")
