#' References
#' [1] DESeq2-MultiBatch: Batch Correction for Multi-Factorial RNA-seq Experiments
#'      Julien Roy, Adrian S. Monthony, Davoud Torkamaneh
#'  bioRxiv 2025.04.20.649392; doi: https://doi.org/10.1101/2025.04.20.649392


check_confounded <- function(design, n_batch) {
  if (qr(design)$rank < ncol(design)) {
    if (ncol(design) == (n_batch + 1)) {
      stop("The covariate is confounded with batch! Remove the covariate")
    }
    if (ncol(design) > (n_batch + 1)) {
      if ((qr(design[, -c(1:n_batch)])$rank < ncol(design[, -c(1:n_batch)]))) {
        stop("The covariates are confounded!
Please remove one or more of the covariates so the design is not confounded")
      } else {
        stop("At least one covariate is confounded with batch!")
      }
    }
  }
}


#' LFC-based batch-effect correction
#'
#' @description
#' Uses the method described in [1]
deseq2_batch <- function(counts, batch, group = NULL, full_mod = TRUE) {
  batch <- as.factor(batch)
  batchmod <- model.matrix(~batch)
  group <- as.factor(group)
  if (full_mod && nlevels(group) > 1) {
    cat("Using full model\n")
    mod <- model.matrix(~ 0 + group)
  } else {
    cat("Using null model\n")
    mod <- model.matrix(~1, data = as.data.frame(t(counts)))
  }
  design <- cbind(batchmod, mod)
  check <- apply(design, 2, function(x) all(x == 1))
  design <- as.matrix(design[, !check])
  check_confounded(design, nlevels(batch))

  # TODO: can get the size factors
  coldata <- data.frame(row.names = colnames(counts), batch = batch, group = group)
  dds <- DESeq2::DESeqDataSetFromMatrix(
    countData = counts, design = design,
    colData = coldata
  ) |> DESeq2::DESeq()
  batch_cols <- colnames(design) |> purrr::keep(\(x) stringr::str_detect(x, "^batch.*"))
  scaling_factors <- vector(mode = "list", length = nlevels(batch)) |> `names<-`(levels(batch))
  batch_df <- DESeq2::results(dds, name = batch_cols, independentFiltering = FALSE) |> as.data.frame()
  ref_batch <- stringr::str_remove(batch_cols, "^batch")

  other_b <- levels(batch) |> purrr::discard(\(x) x == ref_batch)

  batch_df[[ref_batch]] <- sqrt(2^(batch_df$log2FoldChange))
  batch_df[[other_b]] <- 1 / sqrt(2^(batch_df$log2FoldChange))
  # TODO: this only works for batch of two levels. Don't need to worry about others
  # for now

  n_counts <- DESeq2::counts(dds, normalized = TRUE)
  scaled <- n_counts
  for (i in seq_along(ncol(n_counts))) {
    b <- batch[i]
    if (b == ref_batch) {
      s_factors <- batch_df[[ref_batch]]
    } else {
      s_factors <- batch_df[[other_b]]
    }
    id <- colnames(dds)[i]
    scaled[, id] <- n_counts[, id] * s_factors
  }
  scaled
}


combat_ref_params <- function(counts, batch, group = NULL, full_mod = TRUE,
                              genewise.disp = FALSE) {
  pars <- list()
  counts <- as.matrix(counts)
  batch <- as.factor(batch)
  if (any(table(batch) <= 1)) {
    stop("ComBat-req doesn't support 1 sample per batch yet")
  }
  keep_lst <- lapply(levels(batch), function(b) {
    which(apply(counts[, batch == b], 1, function(x) {
      !all(x == 0)
    }))
  })
  keep <- Reduce(intersect, keep_lst)
  rm <- setdiff(1:nrow(counts), keep)
  pars$removed_gene_indices <- rm
  countsOri <- counts
  counts <- counts[keep, ]

  dge_obj <- DGEList(counts = counts)

  ## Prepare characteristics on batches
  n_batch <- nlevels(batch) # number of batches
  batches_ind <- lapply(1:n_batch, function(i) which(batch == levels(batch)[i])) # list of samples in each batch
  n_batches <- sapply(batches_ind, length)
  n_sample <- sum(n_batches)
  cat("Found", n_batch, "batches\n")

  ## Make design matrix
  # batch, use the first batch as the reference
  batchmod <- model.matrix(~batch) # colnames: levels(batch)
  # covariate
  group <- as.factor(group)

  if (full_mod && nlevels(group) > 1) {
    cat("Using full model in ComBat-seq.\n")
    mod <- model.matrix(~ 0 + group) # model.matrix(~0+group)
  } else {
    cat("Using null model in ComBat-seq.\n")
    mod <- model.matrix(~1, data = as.data.frame(t(counts)))
  }

  design <- cbind(batchmod, mod)
  ## Check for intercept in covariates, and drop if present
  check <- apply(design, 2, function(x) all(x == 1))
  design <- as.matrix(design[, !check])

  ## Check if the design is confounded
  if (qr(design)$rank < ncol(design)) {
    if (ncol(design) == (n_batch + 1)) {
      stop("The covariate is confounded with batch! Remove the covariate and rerun ComBat-Seq")
    }
    if (ncol(design) > (n_batch + 1)) {
      if ((qr(design[, -c(1:n_batch)])$rank < ncol(design[, -c(1:n_batch)]))) {
        stop("The covariates are confounded!
Please remove one or more of the covariates so the design is not confounded")
      } else {
        stop("At least one covariate is confounded with batch!
Please remove confounded covariates and rerun ComBat-Seq")
      }
    }
  }

  cat("Estimating dispersions\n")
  ## Estimate common dispersion within each batch
  disp_common <- sapply(1:n_batch, function(i) {
    if ((n_batches[i] <= ncol(design) - ncol(batchmod) + 1) |
      qr(mod[batches_ind[[i]], ])$rank < ncol(mod)) {
      # not enough residual degree of freedom
      estimateGLMCommonDisp(counts[, batches_ind[[i]]], design = NULL, subset = nrow(counts))
    } else {
      estimateGLMCommonDisp(counts[, batches_ind[[i]]], design = mod[batches_ind[[i]], ], subset = nrow(counts))
    }
  })
  for (i in 1:n_batch) {
    cat("Batch ", levels(batch)[i], "(", i, ") dispersion = ", disp_common[i], "\n")
  }
  names(disp_common) <- levels(batch)
  pars$disp_common <- disp_common
  # Choose the batch with the smallest dispersion as the reference batch
  ref_batch <- 1
  for (i in 2:n_batch) {
    if (disp_common[i] < disp_common[ref_batch]) {
      ref_batch <- i
    }
  }
  ref_batch_name <- levels(batch)[ref_batch]
  pars$reference_batch <- ref_batch_name
  cat("Reference batch: ", ref_batch_name, " (", ref_batch, ")\n")
  # Set reference batch as batch 1
  if (ref_batch != 1) {
    # swap disp_common
    tmp <- disp_common[1]
    disp_common[1] <- disp_common[ref_batch]
    disp_common[ref_batch] <- tmp
    for (i in 1:n_sample) {
      if (batch[i] == ref_batch) {
        batch[i] <- 1
      } else if (batch[i] == 1) {
        batch[i] <- ref_batch
      }
    }
    batches_ind <- lapply(1:n_batch, function(i) which(batch == levels(batch)[i]))
    n_batches <- sapply(batches_ind, length)
    batchmod <- model.matrix(~batch) # colnames: levels(batch)
    design <- cbind(batchmod, mod)
    check <- apply(design, 2, function(x) all(x == 1))
    design <- as.matrix(design[, !check])
  }
  # re-compute batches_ind after swapping reference batch

  if (genewise.disp) {
    genewise_disp_lst <- lapply(1:n_batch, function(j) {
      if ((n_batches[j] <= ncol(design) - ncol(batchmod) + 1) | qr(mod[batches_ind[[j]], ])$rank < ncol(mod)) {
        # not enough residual degrees of freedom - use the common dispersion
        rep(disp_common[j], nrow(counts))
      } else {
        estimateGLMTagwiseDisp(counts[, batches_ind[[j]]],
          design = mod[batches_ind[[j]], ],
          dispersion = disp_common[j], prior.df = 0
        )
      }
    })
  } else {
    genewise_disp_lst <- lapply(1:n_batch, function(j) {
      rep(disp_common[j], nrow(counts))
    }) # Just use the same dispersion parameter for each gene in the same batch
  }

  names(genewise_disp_lst) <- paste0("batch", levels(batch))
  pars$genewise_disp <- genewise_disp_lst
  # Dispersion parameters for each gene in each batch

  return(pars)
}

combat_ref_adjust <- function(counts, batch, zero_genes, genewise_disp, group = NULL) {
  adjusted <- matrix(NA, nrow = nrow(counts), ncol = ncol(counts))
  adjusted[zero_genes, ] <- counts[zero_genes, ]
  dimnames(adjusted) <- dimnames(counts)

  counts <- counts[-zero_genes, ]

  batch <- as.factor(batch)
  n_batch <- nlevels(batch)
  batches_ind <- lapply(levels(batch), \(b) which(batch == b))
  n_batch_s <- sapply(batches_ind, length)
  phi_matrix <- matrix(NA, nrow = nrow(counts), ncol = ncol(counts))
  for (k in 1:n_batch) {
    phi_matrix[, batches_ind[[k]]] <- vec2mat(genewise_disp[[k]], n_batch_s[k])
  }
  phi_hat <- do.call(cbind, genewise_disp)

  dge_obj <- DGEList(counts)
  # TODO: Problem here is that you won't have access to group...
  design <- model.matrix(~batch)
  if (!is.null(group)) {
    mod <- model.matrix(~ 0 + group)
  } else {
    mod <- model.matrix(~1, data = as.data.frame(t(counts)))
  }
  design <- cbind(design, mod)
  check <- apply(design, 2, function(x) all(x == 1))
  design <- as.matrix(design[, !check])

  glm_f <- glmFit(dge_obj, design = design, dispersion = phi_matrix, prior.count = 1e-4)
  # Estimate from the original data

  gamma_hat <- as.matrix(glm_f$coefficients[, 1:(n_batch - 1)])
  mu_hat <- glm_f$fitted.values

  adjust_counts <- counts
  # adjust batches except for the first one (reference batch)
  for (kk in 2:n_batch) {
    counts_sub <- counts[, batches_ind[[kk]]]
    old_mu <- pmax(mu_hat[, batches_ind[[kk]]], 1e-4) # numerical stability
    old_phi <- phi_hat[, kk]
    new_mu <- exp(log(old_mu) - vec2mat(gamma_hat[, kk - 1], n_batch_s[kk]))
    # avoid exploding count (mu), new_mu shouldn't increase to more than the ref batch if gamma_hat < 0
    increased_genes <- which(gamma_hat[, kk - 1] < -0.2)
    ncol <- ncol(new_mu)
    ref_max <- vec2mat(rowMaxs(mu_hat[, batches_ind[[1]]]), ncol)
    new_mu[increased_genes, ] <- pmin(new_mu[increased_genes, ], ref_max[increased_genes, ])
    new_phi <- phi_hat[, 1]
    adjust_counts[, batches_ind[[kk]]] <- match_quantiles(
      counts_sub = counts_sub,
      old_mu = old_mu, old_phi = old_phi,
      new_mu = new_mu, new_phi = new_phi, keep_zero = FALSE
    )
  }

  adjusted[-zero_genes, ] <- adjust_counts
  adjusted
}
