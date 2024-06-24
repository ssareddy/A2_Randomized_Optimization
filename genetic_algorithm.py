import time

import mlrose_hiive as mlrose
import pandas as pd


def GA_gridsearch(problem, save_file='results/genetic_algorithm_four_peak_results.csv'):
    """
    Simulates gridsearch method for genetic algorithm
    :param problem: Optimization problem object
    :param save_file: save path to save results
    :return: None
    """

    print('Running Genetic Algorithm Training')

    # Data storage
    outcomes = []
    test_index = 0

    # Common Test variables
    iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    max_attempts = [10, 25, 50]

    # Algorithm Specific Test Variable
    population = [100, 500, 1000]

    for iterate in iterations:
        for max_attempt in max_attempts:
            for pop in population:
                start = time.time()

                print(f'Test: {test_index}, Iterations: {iterate}, Max Attempts: {max_attempt}, Population: {pop}')

                _, fitness, _ = mlrose.genetic_alg(problem=problem,
                                                   pop_size=pop,
                                                   max_attempts=max_attempt,
                                                   max_iters=iterate)

                runtime = time.time() - start

                outcome = {'iteration': iterate,
                           'max_attempts': max_attempt,
                           'population': pop,
                           'fitness': fitness,
                           'runtime': runtime}

                print(f'Storing Test {test_index} results. Test run time: {runtime}. Fitness: {fitness}')
                outcomes.append(outcome)
                test_index += 1

                pd.DataFrame(outcomes).to_csv(save_file)
