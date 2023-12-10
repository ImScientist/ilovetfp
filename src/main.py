import click
import logging
from example_categorical_features.main import main as ex_categorical_features
from example_lin_regression.main import main as ex_lin_regression
from example_stochastic_process.main import main as ex_stochastic_process

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@click.group()
def cli():
    pass


@cli.command("example-categorical-features")
def example_categorical_features_fn():
    """ Compare model training with raw and aggregated data
    that has only categorical features """

    ex_categorical_features()


@cli.command("example-stochastic-process")
def example_stochastic_process_fn():
    """ Learn the parameters of a local level model with drift w """

    ex_stochastic_process(save_dir='outputs_stochastic_process')


@cli.command("example-linear-regression")
def example_lnear_regression_fn():
    """ Learn the weights of a linear regression using the Bayesian inference approach
    """

    ex_lin_regression(save_dir='outputs_linear_regression')


if __name__ == "__main__":
    cli()
