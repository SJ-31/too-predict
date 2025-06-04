library(tidyverse)
library(limma)
library(glue)
library(here)
library(broom)
library(igraph)
library(ComplexHeatmap)
source(here("src", "R", "utils.R"))

outdir_o <- here("data", "output", "chula_organoid_comparison", "de_enrichment")
contrasts <- list(
  sample_type = c(1, -1, 1, -1, 1, -1, 1, -1),
  coad_read = "COAD_READ.organoid - COAD_READ.primary",
  lihc = "LIHC.organoid - LIHC.primary",
  paad = "PAAD.organoid - PAAD.primary",
  chol = "CHOL.organoid - CHOL.primary"
)
ttypes <- c("lihc", "chol", "coad_read", "paad")

ensembl2symbol <- read_tsv(here("data", "mappings", "ensembl_113_id_mapping.tsv")) |>
  distinct(ensembl, .keep_all = TRUE)

## * Gene families

fhierarchy <- read_csv(here("data", "reference", "2025-5-16_hgnc_family_hierarchy.csv"))
families <- read_csv(here("data", "reference", "2025-5-16-hgnc_families.csv"))

hgnc_graph <- graph_from_edgelist(as.matrix(fhierarchy))
# Check that it's acyclic
print(is_acyclic(hgnc_graph))

root_families <- local({
  ids <- as.character(V(hgnc_graph)[degree(hgnc_graph, mode = "in") == 0])
  setNames(families$name, families$id)[ids] |> discard(is.na)
  # node is a root if indegree is 0
})

fmap <- read_tsv(here("data", "mappings", "2025-5-16-hgnc_families_map.tsv"))

with_root_fam <- fmap |>
  group_by(`Ensembl gene ID`) |>
  nest() |>
  mutate(family = lapply(data, \(x) {
    fam <- x$`Family name`
    if (length(fam) > 1) {
      first(keep(fam, \(f) f %in% root_families))
      # Unfortunately genes can belong to multiple root families
    } else {
      fam
    }
  })) |>
  select(-data) |>
  mutate(family = as.character(family))


top_tags <- sapply(names(contrasts), \(n) {
  read_tsv(here(outdir_o, glue("{n}_top_tags.tsv"))) |>
    inner_join(with_root_fam, by = join_by(x$GENEID == y$`Ensembl gene ID`)) |>
    arrange(logFC) |>
    filter(PValue <= 0.05 & !is.na(GENEID))
}, simplify = FALSE, USE.NAMES = TRUE)

batch_var <- read_csv(here("data", "output", "explanations", "batch_correction", "var.csv"))

top_tags$sample_type %>%
  arrange(desc(abs(logFC))) |>
  slice(1:1000) |>
  pluck("GENEID") |>
  writeLines(here("data", "output", "feature_selection", "blacklists", "organoid_vs_primary_lfc-1000.txt"))

get_to_upset <- function(tb_list, target_col, direction, direction_col = "logFC") {
  sapply(tb_list, \(x) {
    if (direction == "pos") {
      cur <- x[x[[direction_col]] > 0, ]
    } else if (direction == "neg") {
      cur <- x[x[[direction_col]] < 0, ]
    } else {
      stop("Direction must be 'pos' or 'neg'")
    }
    cur[[target_col]]
  }, simplify = FALSE, USE.NAMES = TRUE)
}

# Upset plot of top most downreg DE genes
upset_helper <- function(tb_list, filename, target_col = "GENENAME", direction_col = "logFC") {
  validate <- function(lst) {
    !any(map_dbl(lst, length) == 0)
  }
  to_upset_pos <- get_to_upset(tb_list, target_col, "pos", direction_col = direction_col)

  to_upset_neg <- get_to_upset(tb_list, target_col, "neg", direction_col = direction_col)
  has_pos <- validate(to_upset_pos)
  has_neg <- validate(to_upset_neg)
  if (has_pos) {
    mt_pos <- make_comb_mat(to_upset_pos)
  } else {
    mt_pos <- NULL
  }
  if (has_neg) {
    mt_neg <- make_comb_mat(to_upset_neg)
  } else {
    mt_neg <- NULL
  }
  if (!(has_pos || has_neg)) {
    return(NULL)
  }

  if (has_pos && has_neg) {
    ht <- UpSet(mt_pos, row_title = "Upregulated DE genes") %v%
      UpSet(mt_neg, row_title = "Downregulated DE genes")
  } else if (has_pos) {
    ht <- UpSet(mt_pos, row_title = "Upregulated DE genes")
  } else if (has_neg) {
    ht <- UpSet(mt_neg, row_title = "Upregulated DE genes")
  }
  png(here(outdir_o, filename), width = 1080, height = 1920)
  # PNG doesn't work here for some reason
  draw(ht)
  dev.off()
  list(plot = ht, mt_pos = mt_pos, mt_neg = mt_neg)
}

upset_res <- upset_helper(list_modify(top_tags, sample_type = zap()), filename = "de_gene_upset_no_filter.png")


# Shared characteristics of uniquely downreg genes
# Interpretation
unique_downreg <- local({
  to_get <- list(paad = "0010", coad_read = "1000", lihc = "0100", chol = "0001")
  lst <- sapply(to_get, \(x) extract_comb(upset_res$mt_neg, x), simplify = FALSE, USE.NAMES = TRUE)
  tibble(tumor_type = names(lst), GENENAME = lst) |>
    unnest(GENENAME) |>
    left_join(select(ensembl2symbol, ensembl, symbol), by = join_by(x$GENENAME == y$symbol)) |>
    left_join(with_root_fam, by = join_by(x$ensembl == y$`Ensembl gene ID`))
})
ud_families <- table(unique_downreg$tumor_type, unique_downreg$family)


common_expr <- local({
  genes <- top_tags$sample_type$GENEID |> unique()
  for (t in ttypes) {
    tb <- read_tsv(here(outdir_o, glue("{str_to_upper(t)}_ovr_primary_top_tags.tsv")))
    cur <- tb |>
      filter(PValue > 0.05) |>
      pull(GENEID)
    genes <- intersect(cur, genes)
  }
  lst <- list_modify(top_tags, sample_type = zap())
  lst |>
    lmap(\(x) {
      name <- names(x[1])
      mutate(x[[1]], tumor_type = name) |>
        filter(GENEID %in% genes)
    }) |>
    `names<-`(names(lst))
}) # Genes that are DE between organoid-primary but not DE
# in any tumor types for primary

upset_helper(common_expr, filename = "de_gene_upset.pdf")

# Test if gene family is significantly associated with which genes are DE
chi_square_fams <- read_existing(here(outdir_o, "chisq_org_families.rds"), \(f) {
  results <- list()
  tb <- bind_rows(common_expr)
  directions <- list(
    downregulated = filter(tb, logFC < 0),
    upregulated = filter(tb, logFC > 0)
  )
  fams <- unique(tb$family)
  for (n in names(directions)) {
    test_fams <- lapply(fams, \(fam) {
      tab <- table(directions[[n]]$tumor_type, directions[[n]]$family == fam)
      if (dim(tab)[2] == 1) {
        return(NULL)
      }
      tb <- table2tb(tab, fam)
      tab |>
        chisq.test() |>
        tidy() |>
        mutate(family = fam, data = list(tb))
    }) |>
      bind_rows() |>
      mutate(padj = p.adjust(p.value))
    results[[n]] <- test_fams
  }
  saveRDS(results, f)
  results
}, readRDS)


# Data col contains the the nx2 contingency matrix
# TRUE are the number of genes that are downregulated/upregulated in the
# primary vs organoid comparison
chi2_sig_down <- chi_square_fams$downregulated |>
  filter(p.value <= 0.05) |>
  filter(map_lgl(data, \(d) sum(d$`TRUE`) > 50))

# TODO: check the overlap with genes DE between organoids and what is informative
# for primary tumors
# Maximize absolute primary ovr, minimize
edgeR_top <- read_tsv(here("data", "output", "feature_selection", "edgeR_top_types.tsv"))

# Problematic ones are LIHC and CHOL, so...
tmp <- select(edgeR_top, "GENEID", "logFC_tumor_typeLIHC") |>
  inner_join(top_tags$sample_type, by = join_by(GENEID)) |>
  mutate(abs_primary_ovr = abs(logFC_tumor_typeLIHC), abs_org_vs_primary = abs(logFC))

## ** OVR consistency

# Ideally lfc should be correlated, which indicates that the relationships in gene
# expression are preserved by the organoids. But probably won't be observed
# due to differences in organoid culture
ovrs <- sapply(ttypes, \(t) {
  p_ovr <- read_tsv(here(outdir_o, glue("{str_to_upper(t)}_ovr_primary_top_tags.tsv"))) |>
    distinct(GENEID, .keep_all = TRUE) |>
    filter(PValue <= 0.05)
  o_ovr <- read_tsv(here(outdir_o, glue("{str_to_upper(t)}_ovr_organoid_top_tags.tsv"))) |>
    distinct(GENEID, .keep_all = TRUE) |>
    filter(PValue <= 0.05)
  joined <- inner_join(p_ovr, o_ovr, by = join_by(GENEID), suffix = c(".primary", ".organoid"))
  joined
},
simplify = FALSE, USE.NAMES = TRUE
)

# Though really if it's not preserved for one ttype, likely not going to be for others
ovr_corrs <- lmap(ovrs, \(x) {
  cor.test(x[[1]]$logFC.primary, x[[1]]$logFC.organoid) |>
    tidy() |>
    mutate(tumor_type = names(x[1]))
}) |>
  bind_rows()
geneid2biotype <- setNames(edgeR_top$GENEBIOTYPE, edgeR_top$GENEID)
ttype2correlation <- setNames(ovr_corrs$estimate, ovr_corrs$tumor_type)

ovr_plot <- ovrs |>
  list_rbind(names_to = "tumor_type") |>
  mutate(
    biotype = geneid2biotype[GENEID],
    tumor_type = str_to_upper(tumor_type)
  ) |>
  ggplot(aes(x = logFC.primary, y = logFC.organoid, color = biotype)) +
  geom_point() +
  facet_wrap(~tumor_type)
ggsave(plot = ovr_plot, filename = here(outdir_o, "ovr_lfc_consistency.png"), width = 20, height = 12)


# Check overlap of most discriminatory features for each ttype
n <- 70
# TODO: to check if it is worth including organoid features, you need to compare the
# primary auROC scores of primary_best and org_best
lmap(ovrs, \(x) {
  tb <- x[[1]]
  primary_best <- slice_max(tb, abs(logFC.primary), n = n) |> pull(GENEID)
  org_best <- slice_max(tb, abs(logFC.organoid), n = n) |> pull(GENEID)
  i_length <- length(intersect(primary_best, org_best))
  tibble(n_intersect = i_length, percentage = i_length / n, tumor_type = names(x[1]))
}) |>
  bind_rows()


## * BC
bc_top_tags <- inner_join(top_tags$sample_type, batch_var, by = join_by(GENEID))
bc_corr <- cor.test(log(bc_top_tags$bc_mean_organoid_fc), bc_top_tags$logFC)

corr_string <- glue("Correlation: {round(bc_corr$estimate, 3)}")
bc_corr_plot <- ggplot(bc_top_tags, aes(x = logFC, y = log(bc_mean_organoid_fc), color = GENEBIOTYPE)) +
  geom_point() +
  ylab("Mean lfc of organoid samples after correction") +
  labs(
    title = "Correlation between combat-ref batch correction and lfc",
    subtitle = corr_string
  )
ggsave(here(outdir_o, "bc_fc_correlation.png"), bc_corr_plot, width = 13)



## * Gene set overlaps

fgsea <- sapply(names(contrasts), \(x) {
  read_tsv(here(outdir_o, "gene_sets", glue("fgsea_{x}.tsv"))) |>
    mutate(direction = as.numeric(case_match(direction, "pos" ~ 1, "neg" ~ -1)))
}, simplify = FALSE, USE.NAMES = TRUE)

upset_helper(list_modify(fgsea, sample_type = zap()), "de_pathway_upset_no_filter.pdf",
  target_col = "pathway", direction_col = "direction"
)
