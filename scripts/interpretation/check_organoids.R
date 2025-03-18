suppressMessages({
  library(here)
  library(tidyverse)
  library(glue)
  library(edgeR)
  library(scRNAseq)
  library(zellkonverter)
  source(here("src", "R", "utils.R"))
})

outdir <- here("data", "output", "organoid_feature_selection")
dir.create(outdir)

sce_file <- here(
  "data", "output", "normalization_comparison", "edgeR_median_lfc_feature_list_3000",
  "none-plus_one.h5ad"
)
sce <- readH5AD(sce_file)

## --- CODE BLOCK ---
lihc_cases <- c("CHULA_LIHC", "TCGA_LIHC")
chol_cases <- c("CHULA_CHOL", "TCGA_CHOL")
coad_cases <- c("CHULA_COAD_READ", "TCGA_COAD_READ")
all_cases <- c(lihc_cases, chol_cases, coad_cases)


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

write_csv(top_tags, here(outdir, "chula_tcga_dge.csv"))
