evaluation_results = None
evaluations = {}


rule model_evaluation:
    input:
        evaluation_results,
    output:
        **evaluations,
    params:
        var=None,
        src=None,
    shell:
        """ Rscript scripts/format_evaluation.R --var {params.var} \
            --input {input} \
            --post_hoc {output.post_hoc} \
            --omnibus {output.omnibus} \
            --plot {output.metric_plot} \
            --src {params.src}
        """
