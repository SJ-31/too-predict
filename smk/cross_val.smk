import yaml
import pandas as pd
from pathlib import Path
from too_predict.evaluation import make_grid


include: "Snakefile"


if config["test"]:
    splits = {
        "test_2": {"RANDOM": {"5": "exact", "6": "exact"}},
        "test_1": {"RANDOM": {"8": "exact", "3": "exact"}},
    }
    config["shallow"]["cv"]["n_splits"] = 3
    config["cv_n_repeats"] = 1
    config["do_holdout"] = True

else:
    splits = config["dl"]["holdout"]["splits"]
split_names = splits.keys()


cv_outdir = f"{OUT}/shallow/cross_validation/{config.get('date', TODAY)}{RUN}"
holdout_outdir = f"{OUT}/shallow/holdout/{config.get('date', TODAY)}{RUN}"

model_dct = config["models"]["shallow"]
del model_dct["nothing"]
label = config["single_label"]

print("----CROSS VALIDATION----")
if config.get("do_grid"):
    print("Running grid search...")
    try_opt = config.get("grid_search")
    if not try_opt:
        print("Using default options. Override with config parameter `grid_search`")
        grid_vals = config["shallow"]["grid"]["spec"]
    else:
        grid_vals = try_opt
    disallowed = config["shallow"]["grid"].get("disallowed")
    disallowed = disallowed if disallowed is not None else []
    model_dct = {
        m: {"spec": v}
        for m, v in make_grid(grid_vals, disallowed=disallowed, named=True).items()
    }
    models = list(model_dct.keys())
else:
    if only := config.get("run_only", []):
        if isinstance(only, str):
            only = only.split(",")
        models = list(set(model_dct.keys()) & set(only))
        print(f"Running with {models}")
    else:
        models = [k for k in model_dct.keys() if not model_dct[k].get("skip")]
        print("Running with models defined in yaml")
        print("Use the run_only key to run specific models")
print("------------------------")


outputs = {}
cv_all = f"{cv_outdir}/all_results.csv"
holdout_all = f"{holdout_outdir}/all_results.csv"

if config["do_cv"]:
    outputs["cv"] = cv_all
if config["do_holdout"]:
    outputs["holdout"] = holdout_all


rule all:
    input:
        **outputs,


rule cross_validate:
    params:
        outdir=cv_outdir,
        models={m: model_dct[m] for m in models},
    output:
        dirs=[directory(f"{cv_outdir}/{m}") for m in models],
        cv_metrics=expand("{out}/{m}/{l}-misc.csv", out=cv_outdir, m=models, l=label),
    script:
        "scripts/shallow_main.py"


rule holdout:
    params:
        outdir=holdout_outdir,
        models={m: model_dct[m] for m in models},
        split_dct=splits,
    output:
        dirs=[directory(f"{holdout_outdir}/{m}") for m in models],
        metrics=expand(
            "{out}/{m}/{s}-misc.csv", out=holdout_outdir, m=models, s=splits.keys()
        ),
    script:
        "scripts/shallow_main.py"


def get_misc(dir_path, model_name, id_vars="fold"):
    dfs = []
    for csv in dir_path.glob("*-misc.csv"):
        df = pd.read_csv(csv)
        df = df.melt(id_vars=id_vars, var_name="metric", value_name="value").assign(
            model=model_name
        )
        dfs.append(df)
    return dfs


rule combine_cv:
    input:
        rules.cross_validate.output.cv_metrics,
    output:
        cv_all,
    run:
        dfs = []
        for csv in input:
            rdir = Path(csv).parent
            model_name = rdir.stem
            dfs.extend(get_misc(rdir, model_name=model_name))
        pd.concat(dfs).to_csv(output[0], index=False)


rule combine_holdout:
    input:
        rules.holdout.output.metrics,
    output:
        holdout_all,
    run:
        dfs = []
        for rdir in input:
            rdir = Path(rdir).parent
            model_name = rdir.stem
            dfs.extend(get_misc(rdir, model_name=model_name, id_vars="test_set"))
        pd.concat(dfs).to_csv(output[0], index=False)


# rule evaluate_cv:
#     input:
#         rules.cross_validate.output.dirs,
#     output:
#         out=rules.all.input.cv,
#     script:
#         "scripts/cross_val_combine.R"


# rule evaluate_holdout:
#     input:
#         rules.holdout.output.dirs,
#     output:
#         out=rules.all.input.holdout,
#     script:
#         ""


onsuccess:
    with open(f"{cv_outdir}/snakemake_config.yaml", "w") as f:
        yaml.safe_dump(config, f)
