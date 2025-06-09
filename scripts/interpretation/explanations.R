suppressMessages({
  library(here)
  library(tidyverse)
  library(glue)
  library(edgeR)
  library(ggbeeswarm)
  library(ggpubr)
  library(ggVennDiagram)
  library(here)
  library(ggridges)
  library(reticulate)
  source(here("src", "R", "utils.R"))
  use_condaenv(condaenv = "too-predict")
})

## * Setup
OUTDIR <- here("data", "output", "explanations")
DDIR <- here("data", "output")
SDIR <- here(OUTDIR, "chula_misses", "shap_")
LABEL_COL <- "tumor_type"
ad <- import("anndata")
ex <- import("too_predict.explanation")

## * Helper functions


## * Main

edger_all <- read_tsv(here(DDIR, "feature_selection", "edgeR_top_types_backup.tsv"))
chula_tcga_dge <- read_csv(here(DDIR, "organoid_feature_selection", "chula_tcga_dge.csv"))
organoids_lfc <- read_csv(here(DDIR, "chula_organoid_comparison", "genes_all_lfc.csv"))

# Rank by fold changes
sig_org <- organoids_lfc |>
  filter(is_sig) |>
  group_by(tumor_type_comparison) |>
  arrange(desc(abs(logfoldchanges)))


chula_shapley <- ad$read_h5ad(here(SDIR, "shapley-CHULA.h5ad"))
chula_top <- ex$get_most_important(chula_shapley)

chula_filtered <- sig_org |> filter(names %in% unique(unlist(chula_top)))

## ** Assocation between LFC and shap importance

# Let's see correlation between absolute lfc and importance
# Strong association is an indicator of relibability

# But strong association with the organoid-lfc features
# %%
# You can add different levels of filters to see if the association still holds
filter_levels <- c("none", "alpha", "shap_nonzero")
plot_dir <- here(SDIR, "plots")
alpha <- 0.01
dir.create(plot_dir)

cor_fn <- function(x, y) {
  test <- cor.test(x, y, method = "spearman")
  test$estimate
}
for (f in filter_levels) {
  shap_cor <- lapply(unique(sig_org$tumor_type_comparison), \(label) {
    ttype <- str_replace(label, "_", "-")
    cur_shap <- chula_shapley$obsm[[glue("shap_{ttype}")]]
    agg <- abs(cur_shap) |>
      colMeans() |>
      sort(decreasing = TRUE)
    shap_tb <- tibble(shap = agg, names = names(agg))
    if (f == "shap_nonzero") {
      shap_tb <- shap_tb |> filter(shap > 0)
    }

    sig_org_cur <- filter(sig_org, tumor_type_comparison == label) |>
      inner_join(shap_tb, by = join_by(names))
    if (f == "alpha") {
      sig_org_cur <- sig_org_cur |> filter(pvals_adj <= alpha)
    }
    sig_org_cor <- cor_fn(abs(sig_org_cur$logfoldchanges), sig_org_cur$shap)

    edger_cur <- select(
      edger_all,
      all_of(c("GENEID", glue("logFC_{LABEL_COL}{label}"), "PValue"))
    ) |>
      inner_join(shap_tb, by = join_by(x$GENEID == y$names)) |>
      rename(lfc_all = glue("logFC_{LABEL_COL}{label}"))
    if (f == "alpha") {
      edger_cur <- edger_cur |> filter(PValue <= alpha)
    }
    edger_cor <- cor_fn(abs(edger_cur$lfc_all), edger_cur$shap)

    org_de_sel <- c("GENEID", glue("{str_to_lower(label)}_logFC"), glue("{str_to_lower(label)}_PValue"))
    organoid_de_cur <- select(chula_tcga_dge, all_of(org_de_sel)) |>
      inner_join(shap_tb, by = join_by(x$GENEID == y$names)) |>
      rename(lfc_org = glue("{str_to_lower(label)}_logFC")) |>
      rename(PValue = glue("{str_to_lower(label)}_PValue"))
    if (f == "alpha") {
      organoid_de_cur <- filter(organoid_de_cur, PValue <= alpha)
    }
    organoid_de_cor <- cor_fn(abs(organoid_de_cur$lfc_org), organoid_de_cur$shap)

    all_joined <- inner_join(organoid_de_cur, edger_cur, by = c("shap", "GENEID")) |>
      inner_join(sig_org_cur, by = join_by(x$GENEID == y$names), suffix = c("", "_")) |>
      select(GENEID, lfc_org, lfc_all, logfoldchanges, shap) |>
      pivot_longer(-c(GENEID, shap))

    plot <- ggplot(all_joined, aes(x = shap, y = value)) +
      geom_point() +
      ylab("lfc") +
      facet_wrap(~name) +
      labs(title = "lfc vs. SHAP importances", subtitle = glue("for {LABEL_COL} == {label}"))
    ggsave(here(plot_dir, glue("{label}_{f}_shap.png")), plot, width = 8)

    tibble(!!as.symbol(LABEL_COL) := label,
      edger_cor = edger_cor, organoid_lfc_cor = organoid_de_cor,
      organoid_lfc_cor_scanpy = sig_org_cor
    )
  }) |>
    bind_rows()
  write_tsv(shap_cor, here(SDIR, glue("shap_correlation-{f}.tsv")))
}
