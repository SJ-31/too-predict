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
outdir <- here("data", "output", "explanations", "batch_correction")
utils <- import("too_predict.utils")
ad <- import("anndata")
train_utils <- import("too_predict._train_utils")
NO_CLEANUP <- TRUE # REQUIRED to prevent cleanup after combat-seq call

current <- "clr_xgb3_edger_combat_ref"
spec <- train_utils$read_model_spec(train_utils$MODELS[[current]])
filter <- spec[[1]]
model <- spec[[2]]
transform <- spec[[3]]
correction <- spec[[6]]

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
    adata <- ut$training_data_internal_test()
    adata[, 1:1000]
    adata$obs$is_organoid <- adata$obs$Sample_Type != "primary" # not many organoids
    # in this subset
    adata
  }
  gsa_nperm <- 5
  outdir <- here(outdir, "test")
  storage <- outdir
}
dir.create(outdir)

gs <- gs_internal()
markers <- local({
  lst <- markers_internal()
  names(lst) <- paste0("marker:", names(lst))
  min_marker_size <- 10
  max_marker_size <- 1000
  lengths <- map_dbl(lst, length)
  lst <- lst[lengths >= min_marker_size & lengths <= max_marker_size]
})
gs <- append(gs, markers)

## * Transform data

transform_adata <- function(f) {
  adata <- adata_fn()
  adata <- adata[, !is.na(adata$var$GENEID)]

  adata <- filter$fit_transform(adata)
  corrected <- correction$fit_transform(adata)

  # [2025-05-13 Tue]
  # CLR-transform both because these were the values that the predictive model had
  # access to
  # [2025-05-13 Tue] Actually not a good idea because it introduces negatives and makes
  # correction harder to interpret
  ## adata <- transform$fit_transform(adata)
  ## corrected <- transform$fit_transform(corrected)
  print("Correction complete")
  corr_counts <- corrected$X$toarray()
  counts <- adata$X$toarray()
  # All with original as reference
  replaced_0 <- counts
  replaced_0[counts == 0] <- 1

  fc <- corr_counts / replaced_0 # No inf
  corrected$layers["abs_diff"] <- abs(counts - corr_counts)
  corrected$layers["diff"] <- corr_counts - counts
  corrected$layers["fold_change"] <- fc

  # "primary" is the reference for batch effect correction, will not have been modified
  is_organoid <- corrected$obs$is_organoid
  primary <- fc[!is_organoid, ]
  organoid <- fc[is_organoid, ]
  corrected$obs$bc_mean_fc <- rowMeans(fc)
  corrected$obs$bc_sd_fc <- apply(fc, 1, sd)

  corrected$var$bc_mean_primary_fc <- colMeans(primary)
  corrected$var$bc_mean_organoid_abs_diff <- colMeans(corrected$layers["abs_diff"])
  corrected$var$bc_mean_organoid_diff <- colMeans(corrected$layers["diff"])
  corrected$var$bc_mean_organoid_fc <- colMeans(organoid)
  # EX: bc_mean_organoid_fc = 2 is two-fold increase in organoids from primary sample
  corrected$var$bc_sd_primary_fc <- apply(primary, 2, sd)
  corrected$var$bc_sd_organoid_fc <- apply(organoid, 2, sd)
  # We expect this to be low or 0 because the combat_ref didn't
  # consider information within batches

  corrected$write_h5ad(f)
  corrected
}

stored_corrected <- here(storage, "combat_ref_corrected_organoid.h5ad")
adata <- read_existing(stored_corrected, transform_adata, ad$read_h5ad)

obs <- adata$obs
var <- adata$var
fc <- adata$layers["fold_change"]

## * Analyses

# Check if correction is consistent across genes by batch
# The lower the sd for a given sample, the more uniformly the bc was applied to
#     the genes. If sd is 0, then all genes in that sample were corrected to the
#     same degree
mean_bc_boxplot <- obs |> ggplot(aes(
  x = is_organoid, y = , fill = is_organoid,
)) +
  geom_boxplot() +
  ylab("Mean") +
  labs(title = "Mean FC in CLR-transformed expression at sample level, pre- and post-correction")

ggsave(plot = mean_bc_boxplot, filename = here(outdir, "sample_mean_diff.png"))

write_csv(var, here(outdir, "var.csv"))

## ** FGSEA

# Discard those with no correction
ranked <- setNames(var$bc_mean_organoid_fc, rownames(var)) |>
  discard(\(x) x <= 0) |>
  sort(decreasing = TRUE)
#

gs_meta <- read_tsv(gs_meta_internal()) |>
  bind_rows(mutate(markers_meta_internal(), category = "cell marker", set_name = paste0("marker:", set_name))) |>
  inner_join(gene_set_average(gs, var, ref_val_col = "bc_mean_organoid_fc"), by = join_by(set_name)) |>
  rename(mean_batch_effect_fc = average)

write_tsv(gs_meta, here(outdir, "gene_set_metadata.tsv"))

alpha <- 0.05
fgsea_result <- fgsea::fgsea(pathways = gs, stats = ranked)
fgsea_result <- fgsea_result[fgsea_result$padj <= alpha, ]
dir.create(here(outdir, "fgsea"))
plot_fgsea_gseavis(fgsea_result, ranked, gs, here(outdir, "fgsea"))
write_tsv(fgsea_result, here(outdir, "fgsea.tsv"))


## ** GSA

to_gsa <- t(fc[, var$bc_sd != 0]) |> `rownames<-`(var$GENEID)
gsa_result <- tryCatch(
  {
    GSALightning::GSALight(eset = to_gsa, fac = obs$is_organoid, gs = gs, rmGSGenes = "gene", nperm = gsa_nperm)
  },
  error = function(cnd) {
    print(glue("GSA failed, error message: {cnd}"))
    data.frame()
  }
)
if (nrow(gsa_result) > 0) {
  write_tsv(gsa_result, here(outdir, "gsa.tsv"))
}

## ** Differences by gene biotype

biotypes_unique <- unique(adata$var$GENEBIOTYPE)
biotype_fcs <- lapply(biotypes_unique, \(x) {
  adata$var |>
    filter(GENEBIOTYPE == x & !is.na(bc_mean_organoid_fc)) |>
    pluck("bc_mean_organoid_fc")
}) |>
  `names<-`(biotypes_unique)
biotypes <- adata$var$GENEBIOTYPE

adata$var$broad_biotype <- case_when(
  str_detect(biotypes, "_pseudogene") ~ "Pseudogene",
  str_detect(biotypes, "^IG") ~ "Immunoglobulin",
  str_detect(biotypes, "RNA") & !biotypes %in% c("lncRNA", "lincRNA") ~ "ncRNA",
  biotypes == "protein_coding" ~ "Protein-coding",
  .default = "Misc."
)
adata$var$genebiotype <- case_when(
  str_detect(biotypes, "_pseudogene") ~ str_replace(biotypes, "_pseudogene", "_pg"),
  str_detect(biotypes, "_gene") ~ str_remove(biotypes, "_gene"),
  .default = biotypes
)

biotype_box <- ggplot(adata$var, aes(
  x = genebiotype, y = log(bc_mean_organoid_fc),
  fill = broad_biotype
)) +
  geom_boxplot()
ggsave(here(outdir, "biotype_fc_boxplot.png"), plot = biotype_box, width = 9, height = 8)

ktest <- kruskal.test(biotype_fcs) |> tidy()
write_csv(ktest, here(outdir, "kruskal_biotype.csv"))
if (ktest$p.value <= 0.05) {
  wtest <- local({
    two_sided <- pairwise.wilcox.test(x = adata$var$bc_mean_organoid_fc, g = adata$var$GENEBIOTYPE) |>
      tidy() |>
      mutate(alternative = "two-sided")
    greater <- pairwise.wilcox.test(
      x = adata$var$bc_mean_organoid_fc, g = adata$var$GENEBIOTYPE,
      alternative = "greater"
    ) |>
      tidy() |>
      mutate(alternative = "greater")
    bind_rows(two_sided, greater) |>
      mutate(comparison = paste(group1, "vs", group2)) |>
      select(-group1, -group2) |>
      pivot_wider(id_cols = comparison, values_from = p.value, names_from = alternative) |>
      rename(two_sided_pval = "two-sided", greater_pval = "greater") |>
      filter(two_sided_pval <= 0.05)
  })
  write_csv(wtest, here(outdir, "wilcox_biotype.csv"))
}
