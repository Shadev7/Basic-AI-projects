import math
import random
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm

def fitness(max_volume,volumes,prices):
    '''
    This should return a scalar which is to be maximized.
    max_volume is the maximum volume that the knapsack can contain.
    volumes is a list containing the volume of each item in the knapsack.
    prices is a list containing the price of each item in the knapsack, which is aligned with 'volumes'.
    '''

    tot_volume = 0;
    tot_price = 0;
    for volume in volumes:
        tot_volume = tot_volume + volume
    if tot_volume > max_volume :
        return 0

    for price in prices :
        tot_price = tot_price + price
    return tot_price

def randomSelection(population,fitnesses):
    '''
    This should return a single chromosome from the population. The selection process should be random, but with weighted probabilities proportional
    to the corresponding 'fitnesses' values.
    '''
    sum_fitness = sum (fitnesses)
    if sum_fitness == 0:
        return random.choice (population)

    randNo = random.uniform(0,sum_fitness)

    f_counter = 0
    pop_size = len (population)

    #selection based on probability ,i.e., fitness value
    for i in range (0 , pop_size):
        f_counter = f_counter + fitnesses[i]
        if f_counter >= randNo:
            return population[i]

     #In case of some error, return random chromosome   
    return random.choice (population)

def reproduce(mom,dad):
    "This does genetic algorithm crossover. This takes two chromosomes, mom and dad, and returns two chromosomes."
    size_gene = len(mom)

    crossover = random.randint(0,size_gene - 2)
    #fixed crossover
    #child1 = np.append(mom[0: size_gene/2],dad[(size_gene)/2 :])
    #child2 = np.append(dad[0:size_gene/2], mom[(size_gene)/2 :])
    child1 = np.append(mom[0: crossover],dad[crossover :])
    child2 = np.append(dad[0: crossover], mom[crossover :])
    return [child1,child2]
    #pass

def mutate(child):
    "Takes a child, produces a mutated child."
    size_gene = len(child)
    mutate_gene = random.randint(0,size_gene-1);
    if child[mutate_gene] == 0:
        child[mutate_gene] = 1
    else:
        child[mutate_gene] = 0
    return child

def compute_fitnesses(world,chromosomes):
    '''
    Takes an instance of the knapsack problem and a list of chromosomes and returns the fitness of these chromosomes, according to your 'fitness' function.
    Using this is by no means required, if you want to calculate the fitnesses in your own way, but having a single definition for this is convenient because
    (at least in my solution) it is necessary to calculate fitnesses at two distinct points in the loop (making a function abstraction desirable).

    Note, 'chromosomes' is required to be a 2D numpy array of boolean values (a fixed-size boolean array is the recommended encoding of a chromosome, and there should be multiple of these arrays, hence the matrix).
    '''
    
    #return [fitness(world[0], world[1] * chromosome, world[2] * chromosome) for chromosome in chromosomes]
    return [fitness(world[0],chromosome * world[1],chromosome * world[2]) for chromosome in chromosomes] #altered for the sake of understanding

def genetic_algorithm(world,popsize,max_years,mutation_probability):
    '''
    world is a data structure describing the problem to be solved, which has a form like 'easy' or 'medium' as defined in the 'run' function.
    The other arguments to this function are what they sound like.
    genetic_algorithm *must* return a list of (chromosomes,fitnesses) tuples, where chromosomes is the current population of chromosomes, and fitnesses is
    the list of fitnesses of these chromosomes. 
    '''

    final = []
    gene_size = len (world[1])

    #1st random chromosomes
    population = np.zeros((popsize,gene_size), dtype=bool) #initialization
    for i in range (0,popsize):
        for j in range (0,gene_size):
          population[i][j] =  bool(random.getrandbits(1))

    new_population = population

    for year in range(0,max_years+1):
        fitnesses = compute_fitnesses(world,new_population)
        final.append([new_population,fitnesses])
        for pops in range (0,popsize/2):    
            mom = randomSelection (new_population,fitnesses)
            dad = randomSelection (new_population,fitnesses)

            child1,child2 = reproduce(mom,dad)
            roll_mutate = random.uniform(0,1)
            if roll_mutate <=mutation_probability :
                child1 = mutate(child1)
            roll_mutate = random.uniform(0,1)

            if roll_mutate <=mutation_probability :
                child2 = mutate(child2)
            offspring = [child1,child2]
            new_population[(2*pops)] = child1
            new_population[(2*pops) + 1] = child2
            population =  np.concatenate((population,offspring),axis=0)

    return tuple(final)
        

    

def run(popsize,max_years,mutation_probability):
    '''
    The arguments to this function are what they sound like.
    Runs genetic_algorithm on various knapsack problem instances and keeps track of tabular information with this schema:
    DIFFICULTY YEAR HIGH_SCORE AVERAGE_SCORE BEST_PLAN
    '''
    table = pd.DataFrame(columns=["DIFFICULTY", "YEAR", "HIGH_SCORE", "AVERAGE_SCORE", "BEST_PLAN"])
    sanity_check = (10, [10, 5, 8], [100,50,80])
    chromosomes = genetic_algorithm(sanity_check,popsize,max_years,mutation_probability)
    for year, data in enumerate(chromosomes):
        year_chromosomes, fitnesses = data
        table = table.append({'DIFFICULTY' : 'sanity_check', 'YEAR' : year, 'HIGH_SCORE' : max(fitnesses),
            'AVERAGE_SCORE' : np.mean(fitnesses), 'BEST_PLAN' : year_chromosomes[np.argmax(fitnesses)]}, ignore_index=True)
        
    easy = (20, [20, 5, 15, 8, 13], [10, 4, 11, 2, 9] )
    chromosomes = genetic_algorithm(easy,popsize,max_years,mutation_probability)
    for year, data in enumerate(chromosomes):
        year_chromosomes, fitnesses = data
        table = table.append({'DIFFICULTY' : 'easy', 'YEAR' : year, 'HIGH_SCORE' : max(fitnesses),
                              'AVERAGE_SCORE' : np.mean(fitnesses), 'BEST_PLAN' : year_chromosomes[np.argmax(fitnesses)]}, ignore_index=True)
        
    medium = (100, [13, 19, 34, 1, 20, 4, 8, 24, 7, 18, 1, 31, 10, 23, 9, 27, 50, 6, 36, 9, 15],
                   [26, 7, 34, 8, 29, 3, 11, 33, 7, 23, 8, 25, 13, 5, 16, 35, 50, 9, 30, 13, 14])
    chromosomes = genetic_algorithm(medium,popsize,max_years,mutation_probability)
    for year, data in enumerate(chromosomes):
        year_chromosomes, fitnesses = data
        table = table.append({'DIFFICULTY' : 'medium', 'YEAR' : year, 'HIGH_SCORE' : max(fitnesses),
            'AVERAGE_SCORE' : np.mean(fitnesses), 'BEST_PLAN' : year_chromosomes[np.argmax(fitnesses)]}, ignore_index=True)
        
    hard = (5000, norm.rvs(50,15,size=100), norm.rvs(200,60,size=100))
    chromosomes = genetic_algorithm(hard,popsize,max_years,mutation_probability)
    for year, data in enumerate(chromosomes):
        year_chromosomes, fitnesses = data
        table = table.append({'DIFFICULTY' : 'hard', 'YEAR' : year, 'HIGH_SCORE' : max(fitnesses),
            'AVERAGE_SCORE' : np.mean(fitnesses), 'BEST_PLAN' : year_chromosomes[np.argmax(fitnesses)]}, ignore_index=True)
        
    for difficulty_group in ['sanity_check','easy','medium','hard']:
        group = table[table['DIFFICULTY'] == difficulty_group]
        bestrow = group.ix[group['HIGH_SCORE'].argmax()]
        print("Best year for difficulty {} is {} with high score {} and chromosome {}".format(difficulty_group,int(bestrow['YEAR']), bestrow['HIGH_SCORE'], bestrow['BEST_PLAN']))
    table.to_pickle("results.pkl") #saves the performance data, in case you want to refer to it later. pickled python objects can be loaded back at any later point.

run(80,40,0.02)
#run(100,50,0.2) #best solution
