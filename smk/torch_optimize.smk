include: "Snakefile"


store_dir = f"{REPOS}/{config['out']['optuna']['storage']}"
artifact_dir = f"{REPOS}/{config['out']['optuna']['artifacts']}"
outpath = f"{OUT}/optimization/{config.get('date', TODAY)}"
model_cache = f"{REPOS}/adatas/optuna"


def results_spec(name, input: bool = False):
    out = {"df": f"{outpath}/{name}.csv", "study_obj": f"{outpath}/{name}.pkl"}
    if input:
        out["cv"] = f"{outpath}/{name}_cv"
    else:
        out["cv"] = directory(f"{outpath}/{name}_cv")
    return out


rule all:
    input:
        **results_spec("torch_hpo", True),


rule preprocess:
    output:
        f"{model_cache}/optimze.h5ad",
    script:
        f"{config['scripts']}/torch_hpo.py"


rule main_hpo:
    input:
        rules.preprocess.output,
    output:
        **results_spec("torch_hpo", False),
        outdir=directory(f"{outpath}/tensorboard"),
    params:
        storage_file=f"{store_dir}/optim.db",
        artifact_dir=f"{artifact_dir}/optim",
        date=DATE,
    script:
        f"{config['scripts']}/torch_hpo.py"


# TODO: make these
# rule choose_precision:
