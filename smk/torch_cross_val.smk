import yaml
import pandas as pd
from pathlib import Path


include: "Snakefile"


outdir = f"{OUT}/deep/cross_validation/{config.get('date', TODAY)}{RUN}"
config["do_cv"] = True
config["do_holdout"] = False

if config["test"]:
    config["dl"]["trainer"]["accelerator"] = "cpu"
    config["dl"]["trainer"]["max_epochs"] = 2
    config["cv_n_repeats"] = 2
    config["dl"]["trainer"]["log_every_n_steps"] = 1
    config["do_kd"] = True


model_dict = config["models"]["dl"]

print("----CROSS VALIDATION----")
if only := config.get("run_only", []):
    if isinstance(only, str):
        only = only.split(",")
    models = list(set(model_dict.keys()) & set(only))
    print(f"Running with {models}")
else:
    models = [k for k in model_dict.keys() if not model_dict[k].get("skip")]
    print("Running with models defined in yaml")
    print("Use the run_only key to run specific models")
print("------------------------")

results = {}
log_paths = {}
output_files = [
    "cv_results",
    *expand(
        "{lab}-{prefix}_cm", lab=config["multi_labels"], prefix=["average", "total"]
    ),
]
for out_file_type, suffix in zip(["cv", "cv_kd"], ["", "_kd"]):
    log_paths[out_file_type] = [
        directory(d)
        for d in expand(
            "{out}/{model}{s}/tensorboard", out=outdir, s=suffix, model=models
        )
    ]
    results[out_file_type] = expand(
        "{out}/{model}{s}/{f}.csv", out=outdir, s=suffix, model=models, f=output_files
    )


models = models + ["baseline"]
baseline_cv = f"{outdir}/baseline_cv.csv"
all_cv = f"{outdir}/cv_all.csv"

for_all = {"cv": results["cv"]}
if config["do_kd"]:
    for_all["cv_kd"] = results["cv_kd"]
    print("Using knowledge distillation for training")

evaluations = {
    "omnibus": f"{outdir}/friedman_omnibus.csv",
    "post_hoc": f"{outdir}/wilcox_post_hoc.csv",
    "metric_plot": f"{outdir}/metrics_plot.png",
}


rule all:
    input:
        **for_all,
        **evaluations,
        all_cv=all_cv,
        baseline_cv=baseline_cv,


rule preprocess:
    output:
        main=expand(
            "{storage}/adatas/torch_cv_{date}/{models}.h5ad",
            storage=REPOS,
            date=DATE,
            models=models,
        ),
        baseline=f"{REPOS}/adatas/torch_cv_{DATE}/baseline.h5ad",
    script:
        "scripts/torch_main.py"


rule baseline:
    input:
        rules.preprocess.output.baseline,
    output:
        **{m: f"{outdir}/baseline_{m}.csv" for m in config["multi_labels"]},
        cv=rules.all.input.baseline_cv,
    script:
        "scripts/torch_main.py"


rule cross_validate:
    input:
        rules.preprocess.output.main,
    params:
        outdir=outdir,
        date=DATE,
    output:
        cv=results["cv"],
        log=log_paths["cv"],
    script:
        "scripts/torch_main.py"


rule distillation:
    input:
        rules.preprocess.output.main,
    params:
        outdir=outdir,
        date=DATE,
    output:
        cv=results["cv_kd"],
        log=log_paths["cv_kd"],
    script:
        "scripts/torch_main.py"


to_combine = [*rules.cross_validate.output.cv, rules.baseline.output.cv]
if config["do_kd"]:
    to_combine.extend(rules.distillation.output.cv)


rule combine_cvs:
    input:
        to_combine,
    output:
        all_cv,
    run:
        dfs = []
        for csv in input:
            csv = Path(csv)
            if csv.stem != "cv_results":
                continue
            name = Path(csv).absolute().parent.stem
            df = pd.read_csv(csv).assign(model=name)
            dfs.append(df)
        pd.concat(dfs).to_csv(output[0], index=False)


module evaluation:
    snakefile:
        "rules/evaluations.smk"


use rule model_evaluation from evaluation as evaluate with:
    input:
        rules.combine_cvs.output,
    output:
        **evaluations,
    params:
        var="fold",
        src=config["src"]["R"],


onsuccess:
    with open(f"{outdir}/snakemake_config.yaml", "w") as f:
        yaml.safe_dump(config, f)
