import pandas as pd
import mlrose_hiive as mlrose
import matplotlib.pyplot as plt


def plotter():
    for problem in ['four_peak', 'travelling_salesman']:
        for model in ['random_hill_climb', 'simulated_annealing', 'genetic_algorithm']:
            max_fitness = []
            for iteration in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
                data = pd.read_csv(f'results/{model}_{problem}_results.csv')

                data = max(data[data['iteration'] == iteration]['fitness'])

                max_fitness.append(data)

            plt.plot([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], max_fitness, label=model)
        plt.title('Iteration vs. Fitness')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.legend(loc='best')
        plt.savefig(f'plots/iteration_fitness_comparison_for_{problem}.jpg')
        plt.clf()

    for problem in ['four_peak', 'travelling_salesman']:
        for model in ['random_hill_climb', 'simulated_annealing', 'genetic_algorithm']:
            max_fitness = []
            for max_attempts in [10, 25, 50]:
                data = pd.read_csv(f'results/{model}_{problem}_results.csv')

                data = max(data[data['max_attempts'] == max_attempts]['fitness'])

                max_fitness.append(data)

            plt.plot([10, 25, 50], max_fitness, label=model)
        plt.title('Max Attempts vs. Fitness')
        plt.xlabel('Max Attempts')
        plt.ylabel('Fitness')
        plt.legend(loc='best')
        plt.savefig(f'plots/max_attempt_fitness_comparison_for_{problem}.jpg')
        plt.clf()

    for problem in ['four_peak', 'travelling_salesman']:
        max_fitness = []
        for restarts in [0, 5, 10]:
            data = pd.read_csv(f'results/random_hill_climb_{problem}_results.csv')

            data = max(data[data['restart'] == restarts]['fitness'])

            max_fitness.append(data)

        plt.plot([0, 5, 10], max_fitness, label=problem)
    plt.title('Restarts vs. Fitness')
    plt.xlabel('Restarts')
    plt.ylabel('Fitness')
    plt.legend(loc='best')
    plt.savefig(f'plots/restarts_fitness_comparison_for_random_hill_search.jpg')
    plt.clf()

    for problem in ['four_peak', 'travelling_salesman']:
        max_fitness = []
        for schedule in [mlrose.GeomDecay.__name__, mlrose.ExpDecay.__name__, mlrose.ArithDecay.__name__]:
            data = pd.read_csv(f'results/simulated_annealing_{problem}_results.csv')

            data = max(data[data['schedule'] == schedule]['fitness'])

            max_fitness.append(data)

        plt.plot([mlrose.GeomDecay.__name__, mlrose.ExpDecay.__name__, mlrose.ArithDecay.__name__], max_fitness,
                 label=problem)
    plt.title('Schedule vs. Fitness')
    plt.xlabel('Schedule')
    plt.ylabel('Fitness')
    plt.legend(loc='best')
    plt.savefig(f'plots/schedule_fitness_comparison_for_simulated_annealing.jpg')
    plt.clf()

    for problem in ['four_peak', 'travelling_salesman']:
        max_fitness = []
        for population in [100, 500, 1000]:
            data = pd.read_csv(f'results/genetic_algorithm_{problem}_results.csv')

            data = max(data[data['population'] == population]['fitness'])

            max_fitness.append(data)

        plt.plot([100, 500, 1000], max_fitness, label=problem)
    plt.title('Population vs. Fitness')
    plt.xlabel('Population')
    plt.ylabel('Fitness')
    plt.legend(loc='best')
    plt.savefig(f'plots/population_fitness_comparison_for_genetic_algorithm.jpg')
    plt.clf()
