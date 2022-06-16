## [Coordinated Double Machine Learning](https://arxiv.org/pdf/2206.00885.pdf)

Three demos are in `notebooks`:
1. `bias_in_double_machine_learning.ipynb` demonstrates the bias resulting from DML's estimation as we analyzed it in the paper.
2. `synthetic_experiment.ipynb` presents our algorithm, C-DML, and runs a synthetic experiment step by step to show how it works.
3. `synthetic_regularization.ipynb` shows the effect of our regularization on the estimation bias.

To understand the algorithm and use it, please look on the method `run_cv` in `src/synthetic_experiment.py`. \
This code runs C-DML with cross-validation. You may follow the code to apply it to new datasets.