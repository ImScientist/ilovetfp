import click
import logging
from example_categorical_features.main import main as ex_categorical_features

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


if __name__ == "__main__":
    cli()
