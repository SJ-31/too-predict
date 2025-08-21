import yaml


include: "Snakefile"


outpath = f"{OUT}/deep/holdout/{config.get('date', TODAY)}-{RUN}"
config["do_cv"] = False
config["do_holdout"] = True


model_dict = config["models"]["dl"]
split_names = config["dl"]["holdout"]["splits"].keys()

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
            out=outpath,
            splits=split_names,
            s=suffix,
            model=models,
        )
    ]
    results[out_file_type] = expand(
        "{out}/{splits}/{model}{s}.csv",
        out=outpath,
        splits=split_names,
        s=suffix,
        model=models,
    )

baseline_results = expand(
    "{out}/{splits}/baseline.csv", out=outpath, splits=split_names
)

for_all = {"holdout": results["holdout"], "baseline": baselien_results}
if config["do_kd"]:
    for_all["holdout_kd"] = results["holdout_kd"]
    print("Using knowledge distillation for training")


rule all:
    input:
        **for_all,
        all_holdout=all_holdout,


train_test_files = {
    k: [f"{REPOS}/adatas/torch_holdout_{DATE}/{s}_{suffix}.h5ad" for s in split_names]
    for k in ["train", "test"]
}


rule preprocess:
    output:
        **train_test_files,
    script:
        "scripts/torch_main.py"


rule holdout:
    input:
        rules.preprocess.output.train,
    params:
        outdir=outpath,
        models=models,
        date=DATE,
    output:
        holdout=results["holdout"],
        log=log_paths["holdout"],
    script:
        "scripts/torch_main.py"


rule distillation:
    input:
        rules.preprocess.output.main,
    params:
        outdir=outpath,
        date=DATE,
    output:
        holdout=results["holdout_kd"],
        log=log_paths["holdout_kd"],
    script:
        "scripts/torch_main.py"


rule combine:
    input:
        rules.cross_validate.output.holdout,
    output:
        all_holdout,
    run:
        dfs = []
        for csv in input:
            name = Path(csv).absolute().parent.stem
            df = pd.read_csv(csv).assign(model=name)
            dfs.append(df)
        pd.concat(dfs).to_csv(output[0])


onsuccess:
    with open(f"{outpath}/snakemake_config.yaml", "w") as f:
        yaml.safe_dump(config, f)
