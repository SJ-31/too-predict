#!/usr/bin/env ipython


import anndata as ad
import cellxgene_census
import pandas as pd
import too_predict.utils as ut
import yaml
from pyhere import here

census = cellxgene_census.open_soma()

wanted_tissue_general = [
    "breast",
    "bone marrow",
    "brain",
    "central nervous system",
    "colon",
    "eye",
    "heart",
    "kidney",
    "ovary",
    "musculature",
    "skeletal system",
    "uterus",
    "lymph node",
    "immune system",  # Can categorize this and the two below as 'common'
    "blood",
    "vasculature",
]

# Filters for the column in adata.obs
wanted_tissues_in_obs: dict = {
    "breast": ["breast", "upper outer quadrant of breast"],
    "bone": ["bone marrow", "bone spine"],
    "brain": [
        "brain",
        "midbrain",
        "midbrain tectum",
        "midbrain tegmentum",
        "basal forebrain",
        "brain",
        "brain white matter",
        "forebrain",
        "hindbrain",
    ],
    "colon": ["colon"],
    "eye": [
        "anterior segment of eyeball",
        "eye",
        "eyelid",
        "pigment epithelium of eye",
        "posterior segment of eyeball",
    ],
    "heart": [
        "apex of heart",
        "heart left ventricle",
        "heart right ventricle",
        "basal zone of heart",
        "heart",
    ],
    "kidney": ["kidney", "cortex of kidney", "kidney blood vessel"],
    "ovary": ["ovary", "left ovary", "right ovary"],
    "musculature": [
        "intercostal muscle",
        "muscle organ",
        "skeletal muscle tissue",
        "intercostal muscle",
        "skeletal muscle organ, vertebrate",
    ],
    "uterus": ["uterus", "adnexa of uterus"],
    "lymph": [
        "lymph node",
        "mesenteric lymph node",
        "thoracic lymph node",
        "cervical lymph node",
        "external iliac lymph node",
        "inguinal lymph node",
    ],
    "blood": ["blood", "venous blood"],
    "vasculature": ["vein", "vasculature"],
}

wanted_collection_name = [
    "Tabula Sapiens",
    "Cell Atlas of The Human Fovea and Peripheral Retina",
    "HVS: Human variation study",  # TODO: this might only have snRNA data though
    "Single-cell Atlas of common variable immunodeficiency shows germinal center-associated epigenetic dysregulation in B-cell responses",
    "A Balanced Bone Marrow Reference Map of Hematopoietic Development",
    "A Coding and Non-Coding Atlas of the Human Arterial Cell",
    "A human breast atlas integrating single-cell proteomics and transcriptomics",
    "A molecular atlas of the human postmenopausal fallopian tube and ovary from single-cell RNA and ATAC sequencing",
    "A single-cell transcriptome atlas of the adult human retina",
    "Automatic cell-type harmonization and integration across Human Cell Atlas datasets",
    "Cells of the adult human heart",
    "Cell atlas of antigen-presenting and stromal cells from the human intestine",
    "Construction of a human cell landscape at single-cell level",
    "Cross-tissue immune cell analysis reveals tissue-specific features in humans",
    "Healthy living donor kidney",
    "Single cell atlas of the human retina",
]

summary_table = census["census_info"]["summary_cell_counts"].read().concat().to_pandas()

available_tissues = summary_table.query(
    "organism == 'Homo sapiens' & category == 'tissue_general' & label in @wanted_tissue_general"
)

census_datasets: pd.DataFrame = (
    census["census_info"]["datasets"].read().concat().to_pandas()
)
census_datasets = census_datasets.set_index("dataset_id").query(
    "collection_name in @wanted_collection_name"
)
census_datasets.loc[:, "output_name"] = census_datasets["collection_name"].combine(
    census_datasets["soma_joinid"], lambda x, y: f"{x.lower().replace(' ', '_')}-{y}"
)
dataset_id_map = dict(zip(census_datasets.index, census_datasets["output_name"]))

storage_dir = here("remote", "public_data", "cellxgene-census")


def get_tissue_obs(file):
    wanted_tissues = [t for tlist in wanted_tissues_in_obs.values() for t in tlist]
    filter_conditions = [
        f"tissue_general in {wanted_tissue_general}",
        "is_primary_data == True",
        "suspension_type == 'cell'",
        f"tissue in {wanted_tissues}",
        "suspension_type == 'cell'",
        "disease == 'normal'",
        f"dataset_id in {list(census_datasets.index)}",
    ]
    filter_conditions = " and ".join(filter_conditions)
    # Filter duplicate cells by filtering is_primary_data for True
    obs: pd.DataFrame = cellxgene_census.get_obs(
        census, "Homo sapiens", value_filter=filter_conditions
    )
    tissue_counts = obs["tissue"].value_counts().to_dict()
    with open(here("data", "cellxgene_tissue_counts.yaml"), "w") as f:
        yaml.safe_dump(tissue_counts, f)
    with open(here("data", "cellxgene_disease.yaml"), "w") as e:
        yaml.safe_dump(obs["disease"].value_counts().to_dict(), e)

    relevant_datasets = census_datasets.loc[
        census_datasets.index.isin(obs["dataset_id"]), :
    ]
    relevant_datasets.to_csv(
        here("data", "cellxgene_potential_datasets.csv"), index=False
    )

    obs.to_csv(file, index=False)
    adata = cellxgene_census.get_anndata(
        census, "Homo sapiens", obs_value_filter=filter_conditions
    )
    for dataset_id in adata.obs["dataset_id"].unique():
        cur: ad.AnnData = adata[adata.obs["dataset_id"] == dataset_id, :]
        collection = dataset_id_map.get(dataset_id)
        cur.write_h5ad(storage_dir.joinpath(f"{collection}.h5ad"))


adata = ut.read_existing(
    here("data", "cellxgene_tissue_obs.csv"), get_tissue_obs, lambda x: x
)
