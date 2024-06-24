import time

import mlrose_hiive as mlrose
import pandas as pd


def RHC_gridsearch(problem, save_file='results/random_hill_climb_four_peak_results.csv'):
    """
    Simulates gridsearch method for random hill climb
    :param problem: Optimization problem object
    :param save_file: save path to save results
    :return: None
    """

    print('Running Simulated Annealing Training')

    # Data storage
    outcomes = []
    test_index = 0

    # Common Test variables
    iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    max_attempts = [10, 25, 50]

    # Algorithm Specific Test variable
    restarts = [0, 5, 10]

    for iterate in iterations:
        for max_a in max_attempts:
            for restart in restarts:
                start = time.time()

                print(f'Test: {test_index}, Restarts: {restart}, Max Attempts: {max_a}, Iterations: {iterate}')

                _, fitness, _ = mlrose.random_hill_climb(problem=problem,
                                                         restarts=restart,
                                                         max_attempts=max_a,
                                                         max_iters=iterate)

                runtime = time.time() - start

                outcome = {'iteration': iterate,
                           'max_attempts': max_a,
                           'restart': restart,
                           'fitness': fitness,
                           'runtime': runtime}

                print(f'Storing Test {test_index} results. Test run time: {runtime}. Fitness: {fitness}')
                outcomes.append(outcome)
                test_index += 1

                pd.DataFrame(outcomes).to_csv(save_file)
