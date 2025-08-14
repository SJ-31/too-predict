import yaml


include: "Snakefile"


outpath = f"{OUT}/deep/cross_validation/{config.get('date', TODAY)}-{RUN}"


model_dict = config["models"]["dl"]

if only := config.get("run_only", []):
    if isinstance(only, str):
        only = only.split(",")
     models = list(set(model_dict.keys()) & set(only))
else:
    models = [k for k in model_dict.keys() if not model_dict[k].get("skip")]

results = {}
log_paths = {}
for variant, suffix in zip(["cv", "cv_kd"], ["", "_kd"]):
    log_paths[variant] = [
        directory(d)
        for d in expand(
            "{out}/{model}{s}/tensorboard", out=outpath, s=suffix, model=models
        )
    ]
    results[variant] = expand(
        "{out}/{model}{s}/cv_results.csv", out=outpath, s=suffix, model=models
    )

models = models + ["baseline"]
baseline_cv = f"{outpath}/baseline_cv.csv"
all_cv = f"{output}/cv_all.csv"

for_all = {"cv": results["cv"]}
if config["do_kd"]:
    for_all["cv_kd"] = results["cv_kd"]
    print("Using knowledge distillation for training")


rule all:
    input:
        **for_all,
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
        f"{config['scripts']}/torch_cross_val.py"


rule baseline:
    input:
        rules.preprocess.output.baseline,
    output:
        **{m: f"{outpath}/baseline_{m}.csv" for m in config["multi_labels"]},
        cv=rules.all.input.baseline_cv,
    script:
        f"{config['scripts']}/torch_cross_val.py"


rule cross_validate:
    input:
        rules.preprocess.output.main,
    params:
        outdir=outpath,
        date=DATE,
    output:
        cv=results["cv"],
        log=log_paths["cv"],
    script:
        f"{config['scripts']}/torch_cross_val.py"


rule distillation:
    input:
        rules.preprocess.output.main,
    params:
        outdir=outpath,
        date=DATE,
    output:
        cv=results["cv_kd"],
        log=log_paths["cv_kd"],
    script:
        f"{config['scripts']}/torch_cross_val.py"


rule combine_cvs:
    input:
        rules.cross_validate.output.cv,
    output:
        all_cv,
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
