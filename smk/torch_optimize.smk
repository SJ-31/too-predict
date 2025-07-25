include: "Snakefile"


task_options = ("scheduling", "adam", "precision", "task_weights")

hpo_task = config.get("hpo_task", "").lower()
if hpo_task not in task_options:
    raise ValueError(
        f"""
    The top-level config parameter `hpo_task` must be specified
        Options: {task_options}
    """
    )

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
        out["log"] = directory(f"{outpath}/{name}_tensorboard")
    return out


rule all:
    input:
        **results_spec(hpo_task, True),


rule preprocess:
    output:
        f"{model_cache}/optimze.h5ad",
    script:
        f"{config['scripts']}/torch_hpo.py"


rule main:
    input:
        rules.preprocess.output,
    output:
        **results_spec(hpo_task, False),
    params:
        storage_file=f"{store_dir}/optim.db",
        artifact_dir=f"{artifact_dir}/optim",
        date=DATE,
    script:
        f"{config['scripts']}/torch_hpo.py"
