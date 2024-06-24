import pandas as pd
import matplotlib.pyplot as plt

import mlrose_hiive as mlrose

from plotting import plotter
from genetic_algorithm import GA_gridsearch
from random_hill_climb import RHC_gridsearch
from simulated_annealing import SA_gridsearch

problem_length = 1000

problem = mlrose.DiscreteOpt(length=problem_length, fitness_fn=mlrose.FourPeaks(), maximize=True)

RHC_gridsearch(problem)
SA_gridsearch(problem)
GA_gridsearch(problem)

coords = []
df = pd.read_csv('data/UK_Cities.csv')

for row in df.iterrows():
    coords.append((row[1]['Latitude'], row[1]['Longitude']))

fitness_coords = mlrose.TravellingSales(coords=coords)

problem = mlrose.TSPOpt(length=len(coords), fitness_fn=fitness_coords, maximize=True)

RHC_gridsearch(problem, 'results/random_hill_climb_travelling_salesman_results.csv')
SA_gridsearch(problem, 'results/simulated_annealing_travelling_salesman_results.csv')
GA_gridsearch(problem, 'results/genetic_algorithm_travelling_salesman_results.csv')


data = pd.read_csv('results/random_hill_climb_four_peak_results.csv')
data1 = pd.read_csv('results/simulated_annealing_four_peak_results.csv')
data2 = pd.read_csv('results/genetic_algorithm_four_peak_results.csv')

plt.plot(data['index'], data['runtime'], label='RHC')
plt.plot(data1['index'], data1['runtime'], label='SA')
plt.plot(data2['index'], data2['runtime'], label='GA')
plt.title('Random Optimization Model vs Runtime for Four Peaks')
plt.xlabel('Test Index')
plt.ylabel('Runtime(secs)')
plt.legend(loc="best")
plt.savefig('plots/four_peak_runtime_comparison.jpg')
plt.clf()

plt.plot(data['index'], data['fitness'], label='RHC')
plt.plot(data1['index'], data1['fitness'], label='SA')
plt.plot(data2['index'], data2['fitness'], label='GA')
plt.title('Random Optimization Model vs Fitness for Four Peaks')
plt.xlabel('Test Index')
plt.ylabel('Fitness')
plt.legend(loc="best")
plt.savefig('plots/four_peak_fitness_comparison.jpg')
plt.clf()

data = pd.read_csv('results/random_hill_climb_travelling_salesman_results.csv')
data1 = pd.read_csv('results/simulated_annealing_travelling_salesman_results.csv')
data2 = pd.read_csv('results/genetic_algorithm_travelling_salesman_results.csv')

plt.plot(data['index'], data['runtime'], label='RHC')
plt.plot(data1['index'], data1['runtime'], label='SA')
plt.plot(data2['index'], data2['runtime'], label='GA')
plt.title('Random Optimization Model vs Runtime for Travelling Salesman')
plt.xlabel('Test Index')
plt.ylabel('Runtime(secs)')
plt.legend(loc="best")
plt.savefig('plots/travelling_salesman_runtime_comparison.jpg')
plt.clf()

plt.plot(data['index'], data['fitness'], label='RHC')
plt.plot(data1['index'], data1['fitness'], label='SA')
plt.plot(data2['index'], data2['fitness'], label='GA')
plt.title('Random Optimization Model vs Fitness for Travelling Salesman')
plt.xlabel('Test Index')
plt.ylabel('Fitness')
plt.legend(loc="best")
plt.savefig('plots/travelling_salesman_fitness_comparison.jpg')
plt.clf()

plotter()
