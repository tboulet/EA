"""
Genetic algorithm class.
"""

import random
import math
from numpy.core.fromnumeric import sort
from tqdm import tqdm

#Utils
def best_score(pop):
    pop = sorted(pop, key = lambda x: x.fitness_score)
    return pop[-1].fitness_score

#The genetic algo   
class AlgoGen:
    def __init__(self, Indiv, pop_size, fitness_function, mutation_rate, crossover_rate, crossover_method = "Tournoi", elitism = False):

        self.pop_size = pop_size

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crossover_method = crossover_method
        self.elitism = elitism
        self.g = 0

        self.mutation_improve = 0
        self.crossover_improve = 0
        self.mutation_improve_tab = [0]
        self.crossover_improve_tab = [0]

        # Generate initial population
        self.population = [Indiv(fitness_function, name = f"G{0}_I{k}") for k in range(pop_size)]


    def parents_selection(self):
        method = self.crossover_method
        if method == 'Tournoi':         #Les deux individus les meilleurs parmi une SOUS population se reproduisent
            k = self.pop_size//10 + 2
            subpopulation = random.sample(self.population, k = k)
            parent1, parent2 = sorted(subpopulation, key = lambda x: x.fitness_score)[-2:]

        elif method == "2Meilleurs":    #Les deux individus alpha (meilleurs score) se reproduisent. Amène à une population homogène
            parent1, parent2 = sorted(self.population, key = lambda x: x.fitness_score)[-2:]

        elif method == "Troncature":    #Les deux individus les meilleurs parmi une SOUS population ne contenant pas les 50% pire se reproduisent.
            k = self.pop_size//10 + 2
            subpopulation = sorted(self.population, key = lambda x: x.fitness_score)[-self.pop_size//2:]
            subpopulation = random.sample(self.subpopulation, k = k)
            parent1, parent2 = sorted(subpopulation, key = lambda x: x.fitness_score)[-2:]
            

        elif method == "Roulette":  #Chaque individu se reproduit avec un % de chance croissant avec sa fitness.
            parent1, parent2 = random.choices(self.population, weights = [math.exp(indiv.fitness_score) for indiv in self.population], k = 2)

        elif method == "SUS":       #On sélectionne k individus (2 ?) à l'aide d'une roulette à k flêches espacées également, de sorte qu'un individu ne soit pas toujours choisi.
            random.shuffle(self.population)
            somme = sum([math.exp(indiv.fitness_score) for indiv in self.population])
            f1 = random.random() * somme
            f2 = ((random.random()+0.5) % 1) * somme
            somme1 = 0
            somme2 = 0
            for indiv in self.population:
                somme1 += math.exp(indiv.fitness_score)
                if somme1 > f1:
                    parent1 = indiv
                    break
            for indiv in self.population:
                somme2 += math.exp(indiv.fitness_score)
                if somme2 > f2:
                    parent2 = indiv
                    break



        return parent1, parent2
        """
        Renvoie deux individus de la population pour qu'ils effectuent un crossover.
        


        Return : parents : Tuple de deux individus de la population.
        """

    def fit(self, do_tqdm = True):
        itera = self.population
        if do_tqdm: itera = tqdm(itera)
        for indiv in itera:
            indiv.fit()

    def survivor_selection(self):
        #Evaluate each individual : saving model and fitness score
        print(f"Evaluation of generation {self.g} (pop : {len(self.population)})...")
        
        #Pop is sorted according to the fittest
        self.population = sorted(self.population, key = lambda x: x.fitness_score)[-self.pop_size:]
        self.best_indv = self.population[-1]


    def mutation(self):
        print(f"Mutation (pop : {len(self.population)})...")
        if self.elitism: #Elitism : we keep a fraction of the best individuals unmuted, then we mute the whole population.
            elite = sorted(self.population, key = lambda x: x.fitness_score)[-self.pop_size//5:]
            elite_copy = [indiv.copy() for indiv in elite]

            for indiv in self.population:
                indiv.mutate()
                
            self.population += elite_copy
            
        else:
            L_name = list()
            for indiv in self.population:
                if random.random() < self.mutation_rate:
                    L_name.append(indiv.name)
                    indiv.mutate()
            print("Mutation occured for : ", L_name)

        # if best_score(self.population) > best_score_previous:
        #     self.mutation_improve += 1
        #     best_score_previous = best_score(self.population)
        # self.mutation_improve_tab.append(self.mutation_improve)

    def crossover(self):

        print(f"Crossover (pop : {len(self.population)}) ...")
        n_indiv = 0                 #Numero of indiv of this generation.
        for _ in range(self.pop_size//2):
            if random.random() < self.crossover_rate:
                parent1, parent2 = self.parents_selection()
                children1, children2 = parent1.crossover(parent2)
                children1.name = f"G{self.g+1}_I{n_indiv}"
                children2.name = f"G{self.g+1}_I{n_indiv+1}"
                n_indiv += 2
                self.population += [children1, children2]

        # if best_score(self.population) > best_score_previous:   #Pour information
        #     self.crossover_improve += 1
        #     best_score_previous = best_score(self.population)
        # self.crossover_improve_tab.append(self.crossover_improve)


    def sort_pop(self):
        self.population = sorted(self.population, key = lambda x: x.fitness_score)
        self.best_indv = self.population[-1]


    def step(self):
        
        print(f"Generation {self.g}")

        # Fit
        self.fit()

        # Selection 
        self.survivor_selection()

        # Crossover
        self.crossover()

        # Mutation
        self.mutation()

        # Sort by fitness_score
        self.sort_pop()


        print(f"Best adult: {self.best_indv.name} \nSCORE : {self.best_indv.fitness_score} \nCHR : {self.best_indv.chromosomes}")
        # print([indiv.name for indiv in self.population])
        print('\n')
        self.g += 1
    

