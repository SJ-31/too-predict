suppressMessages({
  library(here)
  library(glue)
  library(tidyverse)
  source(here("src", "R", "utils.R"))
  library(reticulate)
  use_condaenv("too-predict")
})
outdir <- here("data", "output", "explanations", "batch_correction")
utils <- import("too_predict.utils")
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
} else {
  adata_fn <- function() {
    adata <- ut$training_data_internal_test()
    adata[, 1:1000]
    adata$obs$is_organoid <- adata$obs$Sample_Type != "primary" # not many organoids
    # in this subset
    adata
  }
  gsa_nperm <- 5
}
adata <- adata[, !is.na(adata$var$GENEID)]

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

adata <- adata_fn()

adata <- filter$fit_transform(adata)
corrected <- correction$fit_transform(adata)
adata <- transform$fit_transform(adata)
corrected <- transform$fit_transform(corrected)
diff <- abs(as.matrix(adata$X) - as.matrix(corrected$X))

obs <- adata$obs
var <- adata$var

primary <- diff[!obs$is_organoid, ]
organoid <- diff[obs$is_organoid, ]

# Check if correction is consistent across genes by batch
# The lower the sd for a given sample, the more uniformly the bc was applied to
#     the genes. If sd is 0, then all genes in that sample were corrected to the
#     same degree
obs$bc_mean <- rowMeans(diff)
obs$bc_sd <- apply(diff, 1, sd)
mean_bc_boxplot <- obs |> ggplot(aes(
  x = is_organoid, y = bc_mean, fill = is_organoid,
)) +
  geom_boxplot() +
  ylab("Mean") +
  labs(title = "Mean absolute difference in CLR-transformed expression at sample level, pre- and post-correction")

# [2025-05-09 Fri] Looks like primary was the reference, you're not seeing any corrections
# to it
var$bc_mean_primary <- colMeans(primary)
var$bc_mean_organoid <- colMeans(organoid)
var$bc_mean_diff <- abs(var$bc_mean_primary - var$bc_mean_organoid)
var$bc_sd <- apply(diff, 2, sd)
var$bc_sd_primary <- apply(primary, 2, sd)
var$bc_sd_organoid <- apply(organoid, 2, sd) # We expect this to be low or 0 because the combat_ref didn't
# consider information within batches

# Discard those with no correction
ranked <- setNames(var$bc_mean_diff, rownames(var)) |>
  discard(\(x) x <= 0) |>
  sort(decreasing = TRUE)

gs_average_diff <- gene_set_average(gs, var, ref_val_col = "bc_mean_organoid")

alpha <- 0.05
fgsea_result <- fgsea::fgsea(pathways = gs, stats = ranked)
fgsea_result <- fgsea_result[fgsea_result$padj <= alpha, ]
dir.create(here(outdir, "fgsea"))
plot_fgsea_gseavis(fgsea_result, ranked, gs, here(outdir, "fgsea"))

to_gsa <- t(diff[, var$bc_sd != 0]) |> `rownames<-`(var$GENEID)

gsa_result <- tryCatch(
  {
    GSALightning::GSALight(
      eset = to_gsa, fac = obs$is_organoid,
      gs = gs, rmGSGenes = "gene", nperm = gsa_nperm
    )
  },
  error = function(cnd) {
    print(glue("GSA failed, error message: {cnd}"))
    data.frame()
  }
)
if (nrow(gsa_result) > 0) {

}
