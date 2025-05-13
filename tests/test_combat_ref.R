suppressMessages({
  library(here)
  library(reticulate)
  use_condaenv("too-predict")
  library(tidyverse)
  source(here("src", "R", "utils.R"))
  source(here("src", "R", "plotting.R"))
})

tdir <- here("data", "tests")
ut <- import("too_predict.utils")

source(here("src", "R", "combat_ref.R"))

stype <- adata$obs$Sample_Type
stype <- replace(stype, is.na(stype), "organoid")
counts <- t(adata$X)

source(here("src", "R", "correction.R"))
counts <- as.matrix(counts)
pars <- combat_ref_params(counts, batch = stype, group = adata$obs$tumor_type, full_mod = FALSE)
adjusted <- combat_ref_adjust(counts,
  batch = stype, zero_genes = pars$removed_gene_indices,
  genewise_disp = pars$genewise_disp, group = adata$obs$tumor_type
)
result <- ComBat_ref(counts,
  batch = stype, group = adata$obs$tumor_type,
  full_mod = TRUE
)

combat_ref_runs <- lapply(1:100, \(x) {
  r1 <- ComBat_ref(counts,
    batch = stype, group = adata$obs$tumor_type,
    full_mod = TRUE
  )
  r2 <- ComBat_ref(counts,
    batch = stype, group = adata$obs$tumor_type,
    full_mod = TRUE
  )
  r3 <- combat_ref_adjust(counts,
    batch = stype, zero_genes = pars$removed_gene_indices,
    genewise_disp = pars$genewise_disp, group = adata$obs$tumor_type
  )
  matrix(c(mean(r1 - r2), mean(r3 - r1)), ncol = 2, nrow = 1)
}) |>
  bind_rows() |>
  as.data.frame() |>
  `colnames<-`(c("old", "new_fn"))
cf_file <- here("data", "tests", "combat_ref_result.rds")
saveRDS(combat_ref_runs, cf_file)
