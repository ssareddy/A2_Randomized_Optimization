import time
import util

import mlrose_hiive as mlrose
import pandas as pd
from sklearn.metrics import roc_auc_score


def GA_gridsearch(file_name, classifier_col, save_file='results/genetic_algorithm.csv'):
    """
    Simulates gridsearch method for genetic algorithm
    :param file_name: data file name
    :param classifier_col: classifier column or y column for data
    :param save_file: save path to save results
    :return: None
    """

    print('Running Genetic Algorithm Training')

    # Data storage
    outcomes = []
    test_index = 0

    # Test variables
    population = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    learning_rate = [1, 5, 10]
    max_attempts = [10, 50, 100]

    # Constant variables
    algorithm = 'genetic_alg'
    nodes = [128, 128, 128, 128]
    activation = 'relu'
    iterations = 100
    dec = 0.92
    clip = 10
    mutation = 0.1

    # Load dataset
    X_train, X_test, y_train, y_test = util.data_load(file_name, classifier_col)

    for lr in learning_rate:
        for max_attempt in max_attempts:
            for pop in population:
                start = time.time()

                print(f'Test: {test_index}, Learning Rate: {lr}, Max Attempts: {max_attempt}, Population: {pop}')

                nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes,
                                                activation=activation,
                                                max_iters=iterations,
                                                algorithm=algorithm,
                                                pop_size=pop,
                                                bias=True,
                                                is_classifier=True,
                                                learning_rate=lr,
                                                early_stopping=True,
                                                clip_max=clip,
                                                max_attempts=max_attempt,
                                                random_state=1,
                                                curve=True,
                                                mutation_prob=mutation)

                nn_model.fit(X_train, y_train)

                y_train_pred = nn_model.predict(X_train)
                y_train_roc = roc_auc_score(y_train, y_train_pred, multi_class="ovr", average="weighted")

                y_test_pred = nn_model.predict(X_test)
                y_test_roc = roc_auc_score(y_test, y_test_pred, multi_class="ovr", average="weighted")

                runtime = time.time() - start

                outcome = {'activation': activation,
                           'learning_rate': lr,
                           'max_iters': iterations,
                           'population': pop,
                           'mutation': mutation,
                           'decay_rates': dec,
                           'max_attempts': max_attempt,
                           'clip': clip,
                           'y_train_roc': y_train_roc,
                           'y_test_roc': y_test_roc,
                           'runtime': runtime,
                           'loss': nn_model.loss}

                print(f'Storing Test {test_index} results. Total run time: {runtime}. Model Loss: {nn_model.loss}')
                outcomes.append(outcome)
                test_index += 1

                pd.DataFrame(outcomes).to_csv(save_file)