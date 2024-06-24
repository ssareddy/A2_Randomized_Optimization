import time

import mlrose_hiive as mlrose
import pandas as pd


def SA_gridsearch(problem, save_file='results/simulated_annealing_four_peak_results.csv'):
    """
    Simulates gridsearch method for simulated annealing
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

    # Algorithm Specific Test Variable
    schedules = [mlrose.GeomDecay, mlrose.ExpDecay, mlrose.ArithDecay]

    for iterate in iterations:
        for max_a in max_attempts:
            for schedule in schedules:
                start = time.time()

                print(f'Test: {test_index}, Schedule: {schedule.__name__}, Max Attempts: {max_a}, Iterations: {iterate}')

                _, fitness, _ = mlrose.simulated_annealing(problem=problem,
                                                           schedule=schedule(),
                                                           max_attempts=max_a,
                                                           max_iters=iterate)

                runtime = time.time() - start

                outcome = {'iteration': iterate,
                           'max_attempts': max_a,
                           'schedule': schedule.__name__,
                           'fitness': fitness,
                           'runtime': runtime}

                print(f'Storing Test {test_index} results. Test run time: {runtime}. Fitness: {fitness}')
                outcomes.append(outcome)
                test_index += 1

                pd.DataFrame(outcomes).to_csv(save_file)
