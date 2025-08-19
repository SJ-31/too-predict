# TODO: Trying to make own synthetic data generation with edgeR
# Use negative binomial model, and edgeR to estimate the parameters

# NOTE: we assume that the data we estimate the parameters from belong the same batch
# or the data have been corrected somehow

# TODO:
# Get means from glmFit
# but what does it mean to fit the model?
# you had the impression that fitting was estimating the dispersions
# See https://pmc.ncbi.nlm.nih.gov/articles/PMC3378882/

# The fit gives you the means under the NB model, where each sample
# has a specific mean, but the dispersions are genewise

#' Simulate count data using edgeR's NB model
#'
#' @param group_col Column of dge$samples specifying groups
#' @param group_prop Proportion of each group to generate. Either a named list,
#' or vector of floats. If passed as a list, only simulated samples for the groups
#' present will be generated
#' @param n number of simulated samples to generate
#' @param estimate_disp Use the estimateDisp method instead of the estimateGLM... sequence
#'   to find dispersions
#' @description
#' This function uses genewise dispersion parameters
#'    i.e. estimateDisp(..., tagwise = TRUE)
#' If a `group_col` is provided, then the fitted values of the GLM are averaged
#'   across all samples in the same group
nb_simulate <- function(
    dge,
    n,
    group_col = NULL,
    group_prop = NULL,
    sample_mus = FALSE,
    estimate_disp = TRUE,
    ...) {
  dge <- normLibSizes(dge)
  if (is.null(group_col)) {
    mm <- NULL
    group_counts <- NULL
  } else {
    n_groups <- nlevels(dge$samples[[group_col]])
    levels <- levels(dge$samples[[group_col]])
    if (is.null(group_prop)) {
      group_prop <- rep(1 / n_groups, n_groups)
    } else if (!is.list(group_prop)) {
      stopifnot(
        "The length of `group_prop` is not equal to the number of groups!" = length(
          group_prop
        ) ==
          n_groups
      )
    } else if (is.list(group_prop)) {
      levels <- levels[levels %in% names(group_prop)]
      n_groups <- length(levels)
    }
    group_counts <- round(unlist(group_prop) * n)
    mm <- model.matrix(as.formula(paste0("~0+", group_col)), data = dge$samples)
  }
  if (estimate_disp) {
    dge <- estimateDisp(dge, design = mm, tagwise = TRUE, ...)
  } else {
    dge <- estimateGLMCommonDisp(dge, design = mm)
    dge <- estimateGLMTagwiseDisp(
      dge,
      design = mm,
      trend = FALSE,
      ...
    )
  }

  glm_f <- glmFit(dge, design = mm) # glmFit better than glmQLFit

  sim_from_dge <- function(
      dge_obj,
      tagwise_avg,
      n_sim,
      mu_matrix = NULL) {
    sapply(seq_len(dim(dge_obj)[1]), \(i) {
      disp <- dge_obj$tagwise.dispersion[i]
      if (!sample_mus) {
        mu <- tagwise_avg[i]
      } else {
        mu <- sample(mu_matrix[i, ], 1)
      }
      rnbinom(n = n_sim, mu = mu, size = 1 / disp)
    }) |>
      cbind()
  }

  if (!is.null(group_col)) {
    simulations <- lapply(
      seq_len(n_groups),
      \(i) {
        mask <- dge$samples[[group_col]] == levels[i]
        # Get average mu estimate per-group for
        # each gene
        cur_avg <- rowMeans(glm_f$fitted.values[, mask])
        sim_from_dge(
          dge,
          cur_avg,
          n_sim = group_counts[i],
          mu_matrix = glm_f$fitted.values[, mask]
        )
      }
    )
    simulations <- do.call(rbind, simulations)

    new_groups <- lapply(
      seq_len(n_groups),
      \(i) rep(levels[i], group_counts[i])
    ) |>
      unlist()
    new_samples <- data.frame(row.names = seq_along(new_groups))
    new_samples[[group_col]] <- new_groups
  } else {
    new_samples <- NULL
    simulations <- sim_from_dge(
      dge,
      rowMeans(glm_f$fitted.values),
      n_sim = n,
      mu_matrix = dge$counts
    )
  }

  DGEList(
    counts = t(simulations),
    genes = dge$genes,
    samples = new_samples
  )
}
