suppressMessages({
  library(here)
  library(glue)
  library(tidyverse)
  library(broom)
  source(here("src", "R", "utils.R"))
  source(here("src", "R", "plotting.R"))
  library(reticulate)
  use_condaenv("too-predict")
})
outdir <- here("data", "output", "explanations", "batch_correction", "GTEs")
ut <- import("too_predict.utils")
FLISTS <- ut$ref_feature_lists_internal()[[2]]
feature_outdir <- here("data", "output", "feature_selection", "feature_lists")

if (path.expand("~") != "/home/shannc") {
  adata_fn <- function() {
    adata <- ut$training_data_internal()
    adata$obs$is_organoid <- adata$obs$Sample_Type == "organoid"
    adata
  }
  gsa_nperm <- NULL
  storage <- here("remote", "repos", "too-predict", "explanations")
} else {
  adata_fn <- function() {
    adata <- ut$training_data_internal_test(minimal = TRUE)
    adata$obs$is_organoid <- adata$obs$Sample_Type != "primary" # not many organoids
    # in this subset
    adata
  }
  gsa_nperm <- 5
  outdir <- here(outdir, "test")
  storage <- outdir
}
dir.create(outdir)
library(scRNAseq)
library(GTEs)

adata <- adata_fn()
adata <- adata[, !is.na(adata$var$GENEID)]

wanted_types <- c("PAAD", "COAD_READ", "LIHC", "CHOL")
wanted_sample <- c("organoid", "primary")

org_compare <- adata[
  (adata$obs$tumor_type %in% wanted_types) &
    (adata$obs$Sample_Type %in% wanted_sample),
]

sce <- zellkonverter::AnnData2SCE(adata)
sce_oc <- zellkonverter::AnnData2SCE(org_compare)

rm(org_compare)
rm(adata)


## * Utility functions

gte_wrapper <- function(
  X,
  meta,
  g_factor,
  b_factor,
  do.scale = FALSE,
  n_feature_subsets = 2000
) {
  indices <- seq_len(nrow(X))
  feature_list <- split(rownames(X), ceiling(indices / n_feature_subsets))
  gte_splits <- lapply(
    feature_list,
    \(x) {
      Run.GroupTechEffects(
        X[x, ],
        meta = meta,
        g_factor = g_factor,
        b_factor = b_factor,
        do.scale = do.scale
      )
    }
  )
  gte <- list()
  gte$GroupTechEffects <- Reduce(
    rbind,
    lapply(gte_splits, \(x) x$GroupTechEffects)
  )
  gte$OverallTechEffects <- Reduce(
    c,
    lapply(gte_splits, \(x) x$OverallTechEffects)
  )
  gte
}

plot_overall <- function(gte, genes) {
  gte <- sort(gte[genes], decreasing = FALSE)
  quantile_nums <- findInterval(quantile(gte, probs = seq(0, 0.9, 0.1)), gte)
  cum_gte <- rev(cumsum(rev(unname(gte))))
  df <- data.frame(
    GTE = rev(cum_gte[quantile_nums]),
    Percentage = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
  )
  ggplot(df, aes(x = Percentage, y = GTE)) +
    geom_bar(stat = "identity") +
    scale_x_continuous(breaks = seq(10, 100, by = 10)) +
    labs(x = "Top GTE genes (%)", y = "Total GTE")
}

filter_hbgs <- function(gte, feature_list_name, out = outdir) {
  blacklist <- Select.HBGs(gte, bins = 0.1, gte.ratio = 0.90)

  original <- FLISTS[feature_list_name]
  removed <- original[!original %in% blacklist]

  new_name <- glue("{feature_list_name}_hbgs_removed.txt")
  write_lines(removed, here(out, new_name))
}

## * Run

gte_all <- read_existing(
  here(outdir, "gte_all.rds"),
  \(f) {
    gte <- gte_wrapper(
      X = assays(sce)$X,
      meta = colData(sce),
      g_factor = "tumor_type",
      b_factor = c("Sample_Type", "Project_ID")
    )
    saveRDS(gte, f)
  },
  readRDS
)

gte_orgs <- read_existing(
  here(outdir, "gte_organoid_compare.rds"),
  \(f) {
    gte <- gte_wrapper(
      X = assays(sce_oc)$X,
      meta = colData(sce_oc),
      g_factor = "tumor_type",
      b_factor = c("Sample_Type", "Project_ID")
    )
    saveRDS(gte, f)
  },
  readRDS
)

q1 <- plot_overall(gte_all$OverallTechEffects, rownames(rowData(sce))) +
  labs(title = "Cumulative GTE of all genes")
ggsave(filename = here(outdir, "cumulative_GTE_all.png"))

q2 <- plot_overall(gte_orgs$OverallTechEffects, rownames(rowData(sce_oc))) +
  labs(title = "Cumulative GTE of all genes")
ggsave(filename = here(outdir, "cumulative_GTE_org_compare.png"))

filter_hbgs(gte_all, "edgeR_median_lfc_feature_list_3000")
