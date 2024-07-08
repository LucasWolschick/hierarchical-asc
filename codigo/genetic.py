from dataclasses import dataclass
import random
import multiprocessing

import hiclass
import numpy as np

from codigo.dataset import tau2019dev
from codigo.experiment import Experiment
from codigo.hierarchy import add_path
from codigo.features.feature_vector import fv_lbp


def update_set(lst, idx, val):
    lst = lst.copy()
    lst[idx] = val
    return lst


def partitions(s):
    if len(s) == 0:
        return []
    elif len(s) == 1:
        return [[[s[0]]]]
    else:
        rp = partitions(s[1:])
        resp = []
        for p in rp:
            for i, subset in enumerate(p):
                # add s[0] to the ith subset
                resp.append(update_set(p, i, [s[0]] + subset))
            # also add a new partition
            resp.append([[s[0]]] + p)
        return resp


def partitions_min2(s):
    ls = []
    for partition in partitions(s):
        if all(len(subset) >= 2 for subset in partition):
            ls.append(partition)
    return ls


lst = [
    "airport",
    "shopping_mall",
    "metro_station",
    "street_pedestrian",
    "public_square",
    "street_traffic",
    "park",
    "tram",
    "bus",
    "metro",
]


@dataclass
class Individual:
    partition: list
    fitness: int


all_partitions = partitions(lst)


def partition_to_label(partition):
    labels = [0] * len(lst)
    for i, subset in enumerate(partition):
        for element in subset:
            labels[lst.index(element)] = i
    return labels


def label_to_partition(label):
    sets = [[] for _ in range(max(label) + 1)]
    for element_i, subset in enumerate(label):
        sets[subset].append(lst[element_i])
    return sets


individual_cache = {}


def get_individual(partition):
    k = str(partition)
    if k not in individual_cache:
        i = Individual(partition, -1)
        individual_cache[k] = i
    return individual_cache[k]


all_partitions = [get_individual(partition_to_label(p)) for p in all_partitions]


POP_SIZE = 100
GENERATIONS = 100

dataset = tau2019dev().frac(k=0.15)
feats = [fv_lbp]  # [fv_lbp, fv_lpq, fv_glcm, fv_mfcc]
rule = lambda r: np.prod(r, axis=0)


def fitness(partition):
    if partition.fitness == -1:
        # run a whole experiment...
        splitted = label_to_partition(partition.partition)
        hier = {}
        hier["_root"] = ["_" + str(s) for s in range(len(splitted))]
        for i, s in enumerate(splitted):
            hier["_" + str(i)] = s
        paths = {}
        add_path(paths, hier, "_root", [])

        experiment = Experiment(
            dataset=dataset, feature_sets=feats, rule=rule, log=False
        )
        res = experiment.run_hier_inner(
            paths=paths, classifier=hiclass.LocalClassifierPerNode
        )
        # jesus
        partition.fitness = res[0]

    return partition.fitness


def selection(population):
    # roleta

    # accumulate fitness
    pool = multiprocessing.Pool()
    scores = pool.map(fitness, population)
    # update caches

    # scores = [fitness(individual) for individual in population]
    accum_scores = [scores[0]]
    for i in range(1, len(scores)):
        accum_scores.append(accum_scores[-1] + scores[i])
    # select two parents
    p1 = random.random() * accum_scores[-1]
    p2 = random.random() * accum_scores[-1]
    while p1 == p2:
        p2 = random.random() * accum_scores[-1]
    # find the parents
    parent1 = population[0]
    for i, score in enumerate(accum_scores):
        if score >= p1:
            parent1 = population[i]
            break
    parent2 = population[1]
    for i, score in enumerate(accum_scores):
        if score >= p2:
            parent2 = population[i]
            break
    return parent1, parent2


def crossover(parent1, parent2):
    # one point crossover

    # select a random position
    point = random.randint(0, len(lst) - 1)

    # create the new partition
    labels = parent1.partition[:point] + parent2.partition[point:]

    # compress
    subsets = sorted(set(labels))
    mapping = {subset: i for i, subset in enumerate(subsets)}
    labels = [mapping[label] for label in labels]

    return get_individual(
        labels,
    )


def mutation(individual):
    new_labels = individual.partition.copy()
    for i in range(len(new_labels)):
        if random.random() < 0.01:
            if random.random() < 0.10:
                # create a new subset
                new_labels[i] = max(new_labels) + 1
            else:
                # move to another subset
                new_labels[i] = random.choice(list(set(new_labels)))

            # compress
            subsets = sorted(set(new_labels))
            mapping = {subset: i for i, subset in enumerate(subsets)}
            new_labels = [mapping[label] for label in new_labels]

    return get_individual(new_labels)


def update(population, new_population):
    lst = population + new_population
    lst.sort(key=lambda x: fitness(x), reverse=True)
    return lst[:POP_SIZE]


def genetic(initial_population):
    population = initial_population
    for gen in range(GENERATIONS):
        print(f"Generation {gen+1}...")
        # selection + crossover + mutation
        new_population = []
        for i in range(POP_SIZE):
            p1, p2 = selection(population)
            i = crossover(p1, p2)
            i = mutation(i)
            new_population.append(i)

        # update
        population = update(population, new_population)

        # best so far
        best_partition = max(population, key=lambda x: fitness(x))
        best_score = fitness(best_partition)
        # worst
        worst_partition = min(population, key=lambda x: fitness(x))
        worst_score = fitness(worst_partition)
        print(
            f"Generation {gen+1}: {best_score} ({label_to_partition(best_partition.partition)}), worst: {worst_score}"
        )

    return population


def run():
    genetic(random.sample(all_partitions, POP_SIZE))
