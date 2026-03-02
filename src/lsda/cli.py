"""
cli.py -- Click-based CLI for the LSDA benchmark pipeline.
"""

import click


@click.group()
def main():
    """LSDA -- Higgs Boson Classification Benchmark (SciKit-Learn vs PySpark)."""
    pass



@main.command()
def eda():
    """Task 1: Exploratory data analysis."""
    from lsda.eda import run_eda
    run_eda()



@main.command()
@click.option("--framework", "-f",
              type=click.Choice(["sklearn", "spark", "all"]),
              default="all", show_default=True,
              help="Which framework(s) to train with.")
@click.option("--model", "-m",
              type=click.Choice(["lr", "rf", "gbt", "all"]),
              default="all", show_default=True,
              help="Which classifier(s) to train.")
@click.option("--optuna", is_flag=True, default=False,
              help="Use Optuna for hyper-parameter search instead of grid search.")
@click.option("--n-trials", type=int, default=None,
              help="Number of Optuna trials (overrides config default).")
@click.option("--n-jobs", type=int, default=-1, show_default=True,
              help="Parallelism for sklearn training.")
@click.option("--n-cores", type=int, default=None,
              help="Number of Spark local cores (default: all).")
def train(framework, model, optuna, n_trials, n_jobs, n_cores):
    """Task 2+3: Feature engineering, model training, tuning & CV."""
    # Override Optuna trial count if provided
    if n_trials is not None:
        from lsda import config
        config.OPTUNA_N_TRIALS = n_trials

    models = None if model == "all" else [model]

    if framework in ("sklearn", "all"):
        from lsda.models.sklearn_models import train_all as sk_train
        sk_train(models=models, use_optuna=optuna, n_jobs=n_jobs)

    if framework in ("spark", "all"):
        from lsda.models.spark_models import train_all as sp_train
        sp_train(models=models, use_optuna=optuna, n_cores=n_cores)



@main.command()
@click.option("--cores", "-c", default="1,2,4,8", show_default=True,
              help="Comma-separated list of core counts to benchmark.")
@click.option("--model", "-m",
              type=click.Choice(["lr", "rf", "gbt", "all"]),
              default="all", show_default=True,
              help="Which classifier(s) to benchmark.")
def benchmark(cores, model):
    """Task 4: Scalability benchmarking across core counts."""
    core_list = [int(c.strip()) for c in cores.split(",")]
    models = None if model == "all" else [model]
    from lsda.benchmark import run_benchmark
    run_benchmark(cores=core_list, models=models)



@main.command()
def evaluate():
    """Task 5: Evaluate all models and produce comparison report."""
    from lsda.evaluate import run_evaluation
    run_evaluation()



@main.command("run-all")
@click.option("--optuna", is_flag=True, default=False,
              help="Use Optuna for hyper-parameter search.")
@click.option("--n-trials", type=int, default=None,
              help="Number of Optuna trials (overrides config default).")
@click.option("--cores", "-c", default="1,2,4", show_default=True,
              help="Comma-separated core counts for the benchmark step.")
@click.pass_context
def run_all(ctx, optuna, n_trials, cores):
    """Run the full pipeline: EDA, Train, Benchmark, Evaluate."""
    click.echo("\nRunning full LSDA pipeline...\n")
    ctx.invoke(eda)
    ctx.invoke(train, framework="all", model="all", optuna=optuna,
               n_trials=n_trials, n_jobs=-1, n_cores=None)
    ctx.invoke(benchmark, cores=cores, model="all")
    ctx.invoke(evaluate)
    click.echo("\nPipeline complete.")


if __name__ == "__main__":
    main()
