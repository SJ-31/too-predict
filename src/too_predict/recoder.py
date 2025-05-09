#!/usr/bin/env ipython

from pathlib import Path
from typing import Literal, Sequence

import anndata as ad
import autogenes as ag
import gseapy as gp
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
import yaml
from scipy import sparse

import too_predict.go_utils as gt
import too_predict.r_utils as ru
import too_predict.transformer as tt
from too_predict.imputer import Imputer
from too_predict.transformer import Transformer

IMPLEMENTED_RECODING = {
    "go",
    "bisque_marker",
    "bisque_reference",
    "plage",
    "gsva",
    "bayesprism",
    "autogenes",
}


class Recoder:
    def __init__(
        self, method: str, layer: str = None, added_fcol: str = "accession", **kwargs
    ) -> None:
        if method.lower() not in IMPLEMENTED_RECODING:
            raise ValueError(f"method {method} not supported!")
        self.adata: ad.AnnData
        self.was_sparse: bool = False
        self.counts: np.ndarray
        self.method: str = method.lower()
        self.layer: str | None = layer
        self.fcol: str = added_fcol
        self.kwargs: dict = kwargs

    def _counts_into_r(self) -> None:
        ru.counts_into_r(self.adata, self.counts)

    @ru.r_cleanup
    def _plage(
        self, reference: Path | dict, metadata: Path | pd.DataFrame | None = None
    ) -> ad.AnnData:
        if isinstance(metadata, Path):
            metadata = pd.read_csv(metadata, sep="\t")
        ro.r("library(GSVA)")
        self._counts_into_r()
        if isinstance(reference, Path):
            ro.r(f"gs <- yaml::read_yaml('{str(reference.absolute())}')")
        else:
            ro.globalenv["gs"] = ro.ListVector(
                {k: ro.StrVector(v) for k, v in reference.items()}
            )
        ro.r("params <- plageParam(exprData = counts, geneSets = gs)")
        ro.r("plage <- gsva(params)")
        vals: np.ndarray = np.transpose(ru.np_from_r(ro.globalenv["plage"]))
        var = pd.DataFrame(index=ro.r("rownames(plage)"))
        if metadata is not None:
            var = var.merge(metadata, left_index=True, right_index=True, how="left")
        return ad.AnnData(X=vals, var=var, obs=self.adata.obs)

    def _autogenes(
        self,
        reference: Path | ad.AnnData,
        hvg_kws: dict | None = None,
        init_kws: dict | None = None,
        transform_kws: dict | None = None,
        optimize_kws: dict | None = None,
        select_kws: dict | None = None,
        deconvolve_kws: dict | None = None,
    ) -> ad.AnnData:
        """Wrapper function for deconvolution by autogenes

        Parameters
        ----------
        reference : scRNA-seq reference data. Assumed to be raw if `process_ref`,
            otherwise, reference counts should be converted to CPM

        Notes
        -----
        Normalizes scRNA-seq reference (if process_ref) and bulk by CPM and TPM
            respectively, as done in the original paper
        """
        defaults: dict[str, dict] = {
            "hvg": {"flavor": "seurat"},
            "init": {"celltype_key": "Cell_Type", "use_highly_variable": True},
            "optimize": {
                "ngen": 1000,
                "objectives": ("correlation", "distance"),
                "mode": "fixed",
                "nfeatures": 400,
                "weights": (-1, 1),
            },
            "transform": {"length_col": "SEQLENGTH"},
            "select": {"weights": (-1, 1)},
            "deconvolve": {"model": "nusvr"},
        }
        for k, v in {
            "hvg": hvg_kws,
            "init": init_kws,
            "optimize": optimize_kws,
            "deconvolve": deconvolve_kws,
            "select": select_kws,
            "transform": transform_kws,
        }.items():
            if v is not None:
                defaults[k].update(v)
        if isinstance(reference, Path):
            reference = ad.read_h5ad(reference)
        else:
            reference = reference.copy()

        # Get common genes
        common_genes = set(reference.var.index) & set(self.adata.var.index)
        if len(common_genes) == 0:
            raise ValueError("No genes in common between reference and bulk!")
        else:
            print(f"{len(common_genes)} common genes found")
        reference = reference[:, reference.var.index.isin(common_genes)]
        mask = self.adata.var.index.isin(common_genes)
        self.adata = self.adata[:, mask].copy()
        self.counts = self.counts[:, mask]

        # Normalize
        sc.pp.normalize_per_cell(reference, counts_per_cell_after=1e4)
        if (
            defaults["init"].get("use_highly_variable", False)
            and "highly_variable" not in reference.var.columns
        ):
            log_ref: ad.AnnData = sc.pp.log1p(reference, copy=True)
            sc.pp.highly_variable_genes(log_ref, **defaults["hvg"], inplace=True)
            hv: pd.Series = log_ref.var["highly_variable"]
            if hv.sum() <= (n_feats := defaults["optimize"]["nfeatures"]):
                raise ValueError(
                    f"The number of highly variable features ({hv.sum()}) is less than the number of requested features ({n_feats})!"
                )
            reference.var.loc[:, "highly_variable"] = hv
        lengths = self.adata.var[defaults["transform"]["length_col"]].values
        trans: tt.Transformer = tt.Transformer(
            "tpm",
            impute_fn=Imputer("plus_one"),
            inplace=False,
            **defaults["transform"],
            gene_lengths=lengths,
        )
        self.adata.X = trans.fit_transform(self.counts)
        if sparse.issparse(self.adata.X):
            self.adata.X = self.adata.X.toarray()

        # Run autogenes
        ag.init(data=reference, **defaults["init"])
        ag.optimize(**defaults["optimize"])
        soln: ad.AnnData = ag.select(copy=True, **defaults["select"])
        coef: np.ndarray = ag.deconvolve(bulk=self.adata, **defaults["deconvolve"])

        # Record data
        recoded: ad.AnnData = ad.AnnData(
            X=coef, obs=self.adata.obs, var=pd.DataFrame(index=soln.obs.index)
        )
        recoded.uns["autogenes_markers"] = list(
            soln.var.query("autogenes")["autogenes"].index
        )
        fmat: np.ndarray = ag.fitness_matrix()
        recoded.uns["autogenes_fitness_matrix"] = pd.DataFrame(
            fmat,
            index=[f"solution_{i}" for i in range(fmat.shape[0])],
            columns=defaults["optimize"]["objectives"],
        )
        return recoded

    def _gsva(
        self,
        reference: Path | dict,
        metadata: Path | pd.DataFrame | None = None,
        **kwargs,
    ) -> ad.AnnData:
        if isinstance(metadata, Path):
            metadata = pd.read_csv(metadata, sep="\t")
        if isinstance(reference, Path):
            with open(reference, "r") as f:
                reference = yaml.safe_load(f)
        for_gp = pd.DataFrame(
            np.transpose(self.counts),
            index=self.adata.var.index,
            columns=self.adata.obs.index,
        )
        if not kwargs:
            kwargs = {"kcdf": "Gaussian", "mx_diff": True}
        result = gp.gsva(for_gp, gene_sets=reference, **kwargs)
        vals = result.res2d.pivot(index="Name", columns="Term", values="ES")
        var = pd.DataFrame(index=vals.columns)
        if metadata is not None:
            var = var.merge(metadata, left_index=True, right_index=True, how="left")
        return ad.AnnData(X=vals.values.astype(np.float64), var=var, obs=self.adata.obs)

    @ru.r_cleanup
    def _bisque(
        self,
        markers: Path | dict | Sequence | None = None,
        reference: Path | ad.AnnData | None = None,
        mode: str = Literal["reference", "marker"],
        cell_type_col: str = "cell_type",
        subject_col: str = "subject",
    ) -> ad.AnnData:
        ru.source("utils.R", in_r=True)
        self._counts_into_r()
        if mode == "marker" and markers is None:
            raise ValueError("Markers must be provided!")
        elif mode == "reference" and reference is None:
            raise ValueError("Single-cell reference must be provided!")
        if mode == "marker":
            if isinstance(markers, Path):
                ro.globalenv["marker_ref"] = str(markers.absolute())
            else:
                ro.globalenv["marker_ref"] = ro.ListVector(
                    {k: ro.StrVector(v) for k, v in markers.items()}
                )
            ro.r("result <- bisque_marker_wrapper(counts, markers = marker_ref)")
            matrix = ru.np_from_r(ro.r("result$bulk.props"))
            genes_used = pd.DataFrame(
                {
                    "set_name": ro.r("names(result$genes.used)"),
                    "genes_used": ro.r("result$genes.used"),
                }
            )
            genes_used.index = genes_used["set_name"]
            genes_used.loc[:, "used_size"] = genes_used["genes_used"].apply(
                lambda x: len(x)
            )
            result = ad.AnnData(
                X=np.transpose(matrix), obs=self.adata.obs, var=genes_used
            )
        else:
            if isinstance(reference, Path):
                reference = ad.read_h5ad(reference)
            ru.counts_into_r(reference, symbol="ref")
            ru.r_null_if_none(subject_col, symbol="subject_col")
            ru.r_null_if_none(markers, symbol="markers", conversion=ro.StrVector)
            if markers is not None and not isinstance(markers, Sequence):
                raise ValueError("Markers must be a sequence in reference mode!")
            ro.globalenv["cell_type_col"] = cell_type_col
            ro.globalenv["subject_col"] = subject_col
            ru.df_to_r(reference.obs, r_symbol="ref_obs")
            ro.r("""
            result <- bisque_reference_wrapper(counts, ref, ref_obs = ref_obs,
                markers = markers,
                cell_type_col = cell_type_col,
                subject_col = subject_col)
            """)
            types = list(ro.r("rownames(result$bulk.props)"))
            matrix = ru.np_from_r(ro.r("result$bulk.props"))
            result = ad.AnnData(
                X=np.transpose(matrix),
                var=pd.DataFrame(index=types),
                obs=self.adata.obs,
            )
            result.uns["sc.props"] = ru.df_from_r(
                ro.r("as.data.frame(result$sc.props)")
            )
            result.obs.loc[:, "rnorm"] = list(ro.r("result$rnorm"))
        return result

    def fit(self, adata: ad.AnnData) -> None:
        if self.method == "gsva":
            # [2025-04-25 Fri] Original gsva uses cqn but this is too slow
            # use tmm instead
            T = Transformer("tmm", log=False, inplace=False, impute_fn=None)
            self.adata = T.fit_transform(adata)
        else:
            self.adata = adata.copy()
        self.counts = adata.X if self.layer is None else adata.layers[self.layer]
        self.was_sparse = sparse.issparse(self.counts)
        self.counts = self.counts.toarray() if self.was_sparse else self.counts

    @ru.r_cleanup
    def _BayesPrism(
        self,
        adata: ad.AnnData,
        reference: ad.AnnData,
        filtered_genes=("Rb", "Mrp", "other_Rb", "chrM", "MALAT1", "chrX", "chrY"),
        state_col="cell_states",
        type_col="cell_types",
        gene_type=("protein_coding"),  # Combination of "protein_coding", "pseudogene",
        # "lincrna"
        malignant_cell_name: str | None = None,
    ) -> ad.AnnData:
        ro.r("library(BayesPrism)")
        ro.globalenv["cell_states"] = ro.StrVector(reference.obs[state_col])
        ro.globalenv["cell_types"] = ro.StrVector(reference.obs[type_col])
        ro.globalenv["gene_types"] = ro.StrVector(gene_type)
        ro.globalenv["to_filter"] = ro.StrVector(filtered_genes)
        ru.r_null_if_none(malignant_cell_name, "malignant")
        ru.counts_into_r(adata, symbol="bulk", transpose=False)
        ru.counts_into_r(reference, symbol="ref", transpose=False)
        ro.r("""ref <- cleanup.genes(ref,
            gene.group = to_filter,
            species = 'hs',
            input.type = 'count.matrix')""")
        ro.r("ref <- select.gene.type(ref, gene.type = gene_types)")
        ro.r("""
        prism <- new.prism(reference = ref,
            mixture = bulk,
            input.type = "count.matrix",
            cell.type.labels = cell_types,
            cell.state.labels = cell_states,
            key = malignant)""")
        ro.r("result <- run.prism(prism)")  # TODO: don't know yet how to retrieve
        ro.r("""
        fraction <- get.fraction(result, which.theta = 'final', state.or.type = 'type')
        """)
        samples = list(ro.r("rownames(fraction)"))
        types = list(ro.r("colnames(fraction)"))
        recoded: pd.DataFrame = pd.DataFrame(
            data=ru.np_from_r(ro.globalenv["fraction"]), columns=types, index=samples
        )
        result = ad.AnnData(
            X=recoded, obs=self.adata.obs, var=pd.DataFrame(index=types)
        )
        return result

    @ru.r_cleanup
    def _multideconv(self, reference: ad.AnnData | Path | None = None) -> ad.AnnData:
        self._counts_into_r()
        if reference is not None:
            if isinstance(reference, Path):
                reference = ad.read_h5ad(reference)
            ru.counts_into_r(reference, symbol="ref")
        # TODO: need to normalize ref

        ro.globalenv["use_sc"] = reference is None
        ro.r("""
        result <- compute.deconvolution(


            )
        """)

    def transform(self) -> ad.AnnData:
        if self.method == "go":
            rg = gt.RecodeGO(**self.kwargs)
            rg.adata = self.adata
            recoded = rg.transform()
        elif self.method == "bisque_marker":
            recoded = self._bisque(mode="marker", **self.kwargs)
        elif self.method == "plage":
            recoded = self._plage(**self.kwargs)
        elif self.method == "gsva":
            recoded = self._gsva(**self.kwargs)
        elif self.method == "bayesprism":
            recoded = self._BayesPrism(**self.kwargs)
        elif self.method == "autogenes":
            recoded = self._autogenes(**self.kwargs)
        elif self.method == "bisque_reference":
            recoded = self._bisque(mode="reference", **self.kwargs)
        else:
            raise ValueError()
        if self.was_sparse:
            recoded.X = sparse.csc_array(recoded.X)
        recoded.var[self.fcol] = recoded.var.index
        return recoded

    def fit_transform(self, adata: ad.AnnData) -> ad.AnnData:
        self.fit(adata)
        return self.transform()
