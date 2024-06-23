import time
import util

import mlrose_hiive as mlrose
import pandas as pd
from sklearn.metrics import roc_auc_score


def RHC_gridsearch(file_name, classifier_col, save_file='results/random_hill_climb.csv'):
    """
    Simulates gridsearch method for random hill climb
    :param file_name: data file name
    :param classifier_col: classifier column or y column for data
    :param save_file: save path to save results
    :return: None
    """

    print('Running Random Hill Climb Training')

    # Data storage
    outcomes = []
    test_index = 0

    # Test variables
    iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    learning_rate = [1, 5, 10]
    max_attempts = [10, 50, 100]

    # Constant variables
    algorithm = 'random_hill_climb'
    nodes = [128, 128, 128, 128]
    activation = 'relu'
    restart = 10
    dec = 0.92
    clip = 10

    # Load dataset
    X_train, X_test, y_train, y_test = util.data_load(file_name, classifier_col)

    for lr in learning_rate:
        for max_attempt in max_attempts:
            for iteration in iterations:
                start = time.time()

                print(f'Test: {test_index}, Learning Rate: {lr}, Max Attempts: {max_attempt}, Iterations: {iteration}')

                nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes,
                                                activation=activation,
                                                max_iters=iteration,
                                                algorithm=algorithm,
                                                bias=True,
                                                is_classifier=True,
                                                learning_rate=lr,
                                                early_stopping=True,
                                                clip_max=clip,
                                                max_attempts=max_attempt,
                                                random_state=1,
                                                curve=True,
                                                restarts=restart)

                nn_model.fit(X_train, y_train)

                y_train_pred = nn_model.predict(X_train)
                y_train_roc = roc_auc_score(y_train, y_train_pred, multi_class="ovr", average="weighted")

                y_test_pred = nn_model.predict(X_test)
                y_test_roc = roc_auc_score(y_test, y_test_pred, multi_class="ovr", average="weighted")

                runtime = time.time() - start

                outcome = {'activation': activation,
                           'learning_rate': lr,
                           'max_iters': iteration,
                           'restarts': restart,
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
