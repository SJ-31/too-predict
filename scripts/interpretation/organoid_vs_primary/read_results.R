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
to_upset_pos <- get_to_upset(list_modify(top_tags, sample_type = zap()), "GENENAME", "pos")
to_upset_neg <- get_to_upset(list_modify(top_tags, sample_type = zap()), "GENENAME", "neg")
mt_pos <- make_comb_mat(to_upset_pos)
mt_neg <- make_comb_mat(to_upset_neg)

png(here(outdir_o, "de_gene_upset.png"), width = 10, height = 15, units = "in", res = 1080)
UpSet(mt_pos, row_title = "Upregulated DE genes") %v%
  UpSet(mt_neg, row_title = "Downregulated DE genes")
dev.off()

# Shared characteristics of uniquely downreg genes
unique_downreg <- local({
  to_get <- list(paad = "0010", coad_read = "1000", lihc = "0100", chol = "0001")
  lst <- sapply(to_get, \(x) extract_comb(mt, x), simplify = FALSE, USE.NAMES = TRUE)
  tibble(tumor_type = names(lst), GENENAME = lst) |>
    unnest(GENENAME) |>
    left_join(select(ensembl2symbol, ensembl, symbol), by = join_by(x$GENENAME == y$symbol)) |>
    left_join(with_root_fam, by = join_by(x$ensembl == y$`Ensembl gene ID`))
})
ud_families <- table(unique_downreg$tumor_type, unique_downreg$family)


top_tags_all <- list_modify(top_tags, sample_type = zap()) |>
  lmap(\(x) mutate(x[[1]], tumor_type = names(x[1]))) |>
  bind_rows()

# TODO: you should include only the genes that are not DE between any of the tumor
# types in primary samples. Then you'd know that the significant famlies below
# are DE as a result of the organoid treatment
chi_square_fams <- read_existing(here(outdir_o, "chisq_org_families.rds"), \(f) {
  results <- list()
  directions <- list(
    downregulated = filter(top_tags_all, logFC < 0),
    upregulated = filter(top_tags_all, logFC > 0)
  )
  fams <- unique(top_tags_all$family)
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

to_upset_pos <- get_to_upset(list_modify(fgsea, sample_type = zap()), "pathway", "pos", "direction")
to_upset_neg <- get_to_upset(list_modify(fgsea, sample_type = zap()), "pathway", "neg", "direction")
## mt_pos <- make_comb_mat(to_upset_pos)
# [2025-05-16 Fri] Only two pathways are upregulated
mt_neg <- make_comb_mat(to_upset_neg)
png(here(outdir_o, "de_pathway_upset.png"), width = 8, height = 8, units = "in", res = 1080)
UpSet(mt_neg, row_title = "Downregulated pathways")
dev.off()
