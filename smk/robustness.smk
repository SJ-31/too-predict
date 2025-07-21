include: "Snakefile"


script = f"{config['scripts']}/robustness.py"


rule all:
    input:
        beta=f"{OUT}/effective_robustness_beta.pkl",
        # evaluation =


rule prep:
    output:
        expand(
            "{out}/{adatas}",
            out=f"{REPOS}/effective_robustness",
            adatas=["train.h5ad", "shifted_test.h5ad", "standard_test.h5ad"],
        ),
    script:
        script


rule get_beta:
    # Obtain the linear regression model for computing effective robustness
    input:
        rules.prep.output,
    output:
        rules.all.input.beta,
    script:
        script


# rule evaluate:
#     input:
#         beta_path=rules.all.input.beta,
#         train=rules.prep.output[0],
#         shifted_test=rules.prep.output[1],
#         standard_test=rules.prep.output[2],
#     output:
#         rules.all.input.evaluation,
#     script:
#         script
