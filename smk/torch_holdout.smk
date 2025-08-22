import yaml
import pandas as pd


include: "Snakefile"


outdir = f"{OUT}/deep/holdout/{config.get('date', TODAY)}-{RUN}"
config["do_cv"] = False
config["do_holdout"] = True

if config["test"]:
    splits = {
        "test_2": {
            "pipeline": "clr_edgeR_old",
            "spec": {"RANDOM": {"5": "exact", "6": "exact"}},
        },
        "test_1": {
            "pipeline": "clr_edgeR_old",
            "spec": {"RANDOM": {"8": "exact", "3": "exact"}},
        },
    }
    config["dl"]["trainer"]["accelerator"] = "cpu"

else:
    splits = config["dl"]["holdout"]["splits"]
split_names = splits.keys()

model_dict = config["models"]["dl"]

print("----HOLDOUT----")
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
for out_file_type, suffix in zip(["holdout", "holdout_kd"], ["", "_kd"]):
    log_paths[out_file_type] = [
        directory(d)
        for d in expand(
            "{out}/{splits}/{model}{s}_tensorboard",
            out=outdir,
            splits=split_names,
            s=suffix,
            model=models,
        )
    ]
    results[out_file_type] = expand(
        "{out}/{splits}/{f}{model}{s}.csv",
        out=outdir,
        splits=split_names,
        s=suffix,
        model=models,
        f=["", *expand("{lab}_cm-", lab=config["multi_labels"])],
    )


baseline_results = expand("{out}/{splits}/baseline.csv", out=outdir, splits=split_names)
all_holdout = f"{outdir}/holdout_summary.csv"

for_all = {"holdout": results["holdout"], "baseline": baseline_results}
if config["do_kd"]:
    for_all["holdout_kd"] = results["holdout_kd"]
    print("Using knowledge distillation for training")


rule all:
    input:
        **for_all,
        all_holdout=all_holdout,


train_test_files = {
    k: [f"{REPOS}/adatas/torch_holdout_{DATE}/{s}_{k}.{ext}" for s in split_names]
    for k, ext in zip(["train", "test", "spec"], ["h5ad", "h5ad", "yaml"])
}


rule preprocess:
    output:
        **train_test_files,
    params:
        outdir=REPOS,
        split_dct=splits,
    script:
        "scripts/torch_main.py"


rule holdout:
    input:
        rules.preprocess.output.train,
    params:
        outdir=outdir,
        models=models,
        date=DATE,
    output:
        holdout=results["holdout"],
        baseline=for_all["baseline"],
        log=log_paths["holdout"],
    script:
        "scripts/torch_main.py"


rule distillation:
    input:
        rules.preprocess.output.train,
    params:
        outdir=outdir,
        date=DATE,
    output:
        holdout=results["holdout_kd"],
        log=log_paths["holdout_kd"],
    script:
        "scripts/torch_main.py"


rule combine:
    input:
        rules.holdout.output.holdout,
        rules.holdout.output.baseline,
    output:
        all_holdout,
    run:
        dfs = []
        for csv in input:
            f = Path(csv)
            split_name = f.absolute().parent.stem
            model_name = f.stem
            df = pd.read_csv(csv).assign(model=model_name, split=split_name)
            dfs.append(df)
        pd.concat(dfs).to_csv(output[0], index=False)


onsuccess:
    with open(f"{outdir}/snakemake_config.yaml", "w") as f:
        yaml.safe_dump(config, f)
