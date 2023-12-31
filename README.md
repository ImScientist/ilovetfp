# I :black_heart: TensorFlow probability

The repository contains several examples where tensorflow probability is utilized:

- Variational inference in probabilistic models: an analytically solvable
  example: [blog](https://medium.com/@imscientist/variational-inference-in-probabilistic-models-an-analytically-solvable-example-b14d954783b3)

- Bayesian inference for stochastic processes: an analytically solvable
  problem [blog](https://medium.com/@imscientist/bayesian-inference-for-stochastic-processes-an-analytically-solvable-problem-7ae8608a82b9)
  | [colab-notebook](https://colab.research.google.com/drive/1UYKfdcOAzLXVgyPSqsKqpzDhclpog-BF?usp=sharing)

- Simplify likelihood when regressing against categorical variables: [blog]()
  | [colab-notebook](https://colab.research.google.com/drive/11c8W9Sy3GleRkK7d6xs081Tv3tnYafhf?usp=sharing)

To test the examples execute:

```shell
export TF_CPP_MIN_LOG_LEVEL=2

PYTHONPATH=$(pwd) python src/main.py example-linear-regression

PYTHONPATH=$(pwd) python src/main.py example-stochastic-process 

PYTHONPATH=$(pwd) python src/main.py example-categorical-features
```
