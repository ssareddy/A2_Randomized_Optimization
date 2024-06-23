from genetic_algorithm import GA_gridsearch
from random_hill_climb import RHC_gridsearch
from simulated_annealing import SA_gridsearch

RHC_gridsearch('mobile_price_train.csv', 'price_range')
SA_gridsearch('mobile_price_train.csv', 'price_range')
GA_gridsearch('mobile_price_train.csv', 'price_range')
