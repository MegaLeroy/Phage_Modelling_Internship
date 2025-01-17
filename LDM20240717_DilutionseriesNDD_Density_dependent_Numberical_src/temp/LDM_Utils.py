# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 06:48:45 2024

@author: Le Roy
Utility variables, methods and classes
"""

import argparse
import os
import pathlib
from copy import deepcopy
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

cellTypes = {'Bacteria': 0, 'Unused': 1}

# cellstates
cellStates = {'Healthy': 0, 'Infected': 1, 'mutated': 2, 'damaged': 3}

# grid colors
__WHITE = [255, 255, 255]
__BLACK = [0, 0, 0]
__LIGHT_RED = [255, 125, 100]
__RED = [230, 80, 50]
__DARK_RED = [150, 30, 10]
__LIGHT_GREEN = [150, 240, 100]
__GREEN = [120, 230, 60]
__DARK_GREEN = [40, 100, 10]
__BLUE = [100, 190, 210]
# __DARK_BLUE = [0, 0, 255]


colors = {cellTypes['Bacteria']: {cellStates['Healthy']: __GREEN, cellStates['Infected']: __LIGHT_GREEN,
                                cellStates['mutated']: __DARK_GREEN, cellStates['damaged']: __BLUE},
          cellTypes['Unused']: {cellStates['Healthy']: __RED, cellStates['Infected']: __LIGHT_RED,
                                 cellStates['mutated']: __DARK_RED, cellStates['damaged']: __BLUE},
          'empty': __WHITE}

class Cell(object):
    """ Represents a basic cell, of stable or mutant tyes and with healthy,
    Infected or mutated states. """

    def __init__(self,
                 cellType: str = 'Bacteria',
                 cellState: str = 'Healthy',
                 replicationRate: float = 1,
                 repairProb: float = 0.95):

        self.type = cellTypes[cellType]
        self.state = cellStates[cellState]
        self.replicationRate = replicationRate
        self.repairProb = repairProb


def init_world(world: List[List[Cell]],
               totalDensity: float,
               BacterialDensity: float,
               args,
               rng: np.random.default_rng) -> None:
    """ Initialise grid with Bacteria or Unused cells.
        There are density * len(world) cells placed, half of them
        being Bacteria and half of them being Unused. """

    rows = rng.choice(len(world), size=round(len(world) * len(world) *
                                             totalDensity), replace=True)
    cols = rng.choice(len(world), size=round(len(world) * len(world) *
                                             totalDensity), replace=True)

    for i in range(len(rows)):  # Replace the stableRR and repair agrugents.
        if rng.uniform() < BacterialDensity:
            world[rows[i]][cols[i]] = Cell('Bacteria', replicationRate=args.stableRR,
                                           repairProb=args.stableRepairProb)
        else:
            world[rows[i]][cols[i]] = Cell('Unused', replicationRate=args.mutatorRR,
                                           repairProb=args.mutatorRepairProb)

    return


def precompute_moore_domain(world):
    mooreDomain = deepcopy(world)
    for i in range(len(mooreDomain)):
        for j in range(len(mooreDomain)):
            domain = [[i - 1, j - 1], [i - 1, j], [i - 1, j + 1],
                      [i,     j - 1],             [i,     j + 1],
                      [i + 1, j - 1], [i + 1, j], [i + 1, j + 1]]

            # parse invalid indexes (edges)
            for t in range(len(domain)):
                if domain[t][0] >= len(world) or domain[t][0] < 0:
                    domain[t][0] = None
                if domain[t][1] >= len(world) or domain[t][1] < 0:
                    domain[t][1] = None

            # delete invalid indexes (edges)
            mooreDomain[i][j] = [index for index in domain if index[0] is not None and index[1] is not None]

    return mooreDomain


def choose_moore_domain(row, col, mooreDomain, n, rng) -> np.ndarray:
    """ Choose n indexes w/o replacement in the moore domain
        centered around i, j. Does not consider grid edges.

    :return: a list of indexes as lists. """

    return rng.choice(mooreDomain[row][col], size=n, replace=False)


def update_statistics(world, prevStats):
    """ Updates a dict with statistics of the current world """

    Bacteria = 0
    stableMutated = 0
    stableDamaged = 0
    BacteriaHealthy = 0

    mutatorCells = 0
    mutatorMutated = 0
    mutatorDamaged = 0
    mutatorHealthy = 0

    for i in range(len(world)):  #stableDamaged & stableMutated can be changed
        for j in range(len(world)):
            if world[i][j] is not None:
                if world[i][j].type == cellTypes['Bacteria']:
                    Bacteria += 1
                    if world[i][j].state == cellStates['mutated']:
                        stableMutated += 1
                    elif world[i][j].state == cellStates['damaged']:
                        stableDamaged += 1
                    elif world[i][j].state == cellStates['Healthy']:
                        BacteriaHealthy += 1
                else:  # The whole mutator section can be replaced
                    mutatorCells += 1
                    if world[i][j].state == cellStates['mutated']:
                        mutatorMutated += 1
                    elif world[i][j].state == cellStates['damaged']:
                        mutatorDamaged += 1
                    elif world[i][j].state == cellStates['Healthy']:
                        mutatorHealthy += 1

    prevStats['Bacteria']['total'].append(Bacteria)
    prevStats['Bacteria']['Healthy'].append(BacteriaHealthy)
    prevStats['Bacteria']['damaged'].append(stableDamaged)
    prevStats['Bacteria']['mutated'].append(stableMutated)

    prevStats['Unused']['total'].append(mutatorCells)
    prevStats['Unused']['Healthy'].append(mutatorHealthy)
    prevStats['Unused']['damaged'].append(mutatorDamaged)
    prevStats['Unused']['mutated'].append(mutatorMutated)

    prevStats['BacteriaVictories'] = 1 if prevStats['Bacteria']['total'] > prevStats['Unused']['total'] \
        else 0
    prevStats['UnusedVictories'] = int(not prevStats['UnusedVictories'])

    return prevStats


def update_mean_stats(stats, meanSimulationStats, iterations):
    # update mean statistics
    for key, meanVal in meanSimulationStats.items():
        if isinstance(meanVal, dict):
            for cellCountType in stats[key].keys():
                updatedList = meanSimulationStats[key][cellCountType]
                updatedList = [meanEpochVal + round(epochVal / iterations, 2)
                               for meanEpochVal, epochVal in zip(updatedList,
                                                                 stats[key][cellCountType])]
                meanSimulationStats[key][cellCountType] = updatedList
        else:
            # total victories of cell types
            meanSimulationStats[key] += stats[key]

    return meanSimulationStats


def simulation_skeleton(args, rng, forward_func, worldSize, iterations, totalPopDensity, BacterialDensity,
                        animate, epochs, forward_func_args: dict):

    """ Simulation skeleton. 
        The forward function must have the signature used in this source.

    :param args: argparse object with the simulation parameterss
    :param rng: np.random.default_rng object
    :param forward_func: function that advances the simulation 1 epoch
    :param int worldSize: width and height of the grid
    :param int iterations: number of times the simulation will be repeated
    :param float totalPopDensity:
            [0, 1] density of the total initial cell population
    :param float BacterialDensity:
            [0, 1] relative density of the stable cell population.
            Unstable cells will have (1-BacterialDensity) as initial density.
    :param bool animate: Create an animation with the evolution of the world? 
            Ignored if iterations > 1
    :param int epochs: Number of world epochs per iteration.
    :param dict forward_func_args: additional keyword arguments for forward_func
    """

    expName = f'experiment_{datetime.now().strftime("%d%m%Y_%H%M%S")}'
    currentPath = pathlib.Path(__file__).parent.resolve()

    figurePath = currentPath.parent.resolve() / 'experiment_results/'
    if not figurePath.exists():
        os.mkdir(figurePath)

    os.mkdir(figurePath / expName)

    # save parameters of the experiment
    with open(figurePath / expName / "params.txt", "w") as file:
        file.write(parse_all_params(args))

    world = [[None for _ in range(worldSize)] for _ in range(worldSize)]

    mooreDomain = precompute_moore_domain(world)

    stats = {'Bacteria': {'total': [], 'Healthy': [], 'damaged': [], 'mutated': []},
             'Unused': {'total': [], 'Healthy': [], 'damaged': [], 'mutated': []},
             'BacteriaVictories': 0, 'UnusedVictories': 0}
    if iterations == 1:
        # animation only available when 1 iteration is performed.
        init_world(world, totalPopDensity, BacterialDensity, args, rng)

        if animate:
            fig = plt.figure()
            data = np.zeros((worldSize, worldSize, 3))
            im = plt.imshow(data)

            plt.tick_params(axis='both',  # changes apply to both
                            which='both',  # both major and minor ticks are affected
                            bottom=False,  # ticks along the bottom edge are off
                            top=False,  # ticks along the top edge are off
                            left=False,
                            labelleft=False,
                            labelbottom=False)

            def init():
                im.set_data(np.zeros((worldSize, worldSize, 3)))
                return im

            def animate_frame(_, prevStats):
                _ = forward_func(world=world, mooreDomain=mooreDomain, prevStats=prevStats, rng=rng,
                                 **forward_func_args)
                im.set_data(color_world(world))
                return im

            anim = animation.FuncAnimation(fig, animate_frame, init_func=init, frames=epochs - 1 if epochs > 1 else 1,
                                           fargs=[stats])
            anim.save(f'{figurePath}/{expName}/system_evolution.gif',
                      fps=min(max(10, epochs / 20), 24))
            stats = forward_func(world=world, mooreDomain=mooreDomain,
                                 prevStats=stats, rng=rng,
                                 **forward_func_args)
        else:
            for i in range(epochs):
                stats = forward_func(world=world, mooreDomain=mooreDomain,
                                     prevStats=stats, rng=rng,
                                     **forward_func_args)

        gen_save_plots(epochs, stats, figurePath / expName, iterations)

    elif iterations > 1:
        zeros = [0 for _ in range(epochs)]
        meanSimulationStats = {'Bacteria': {'total': zeros, 'Healthy': zeros,
                                            'damaged': zeros, 'mutated': zeros},
                               'Unused': {'total': zeros, 'Healthy': zeros,
                                          'damaged': zeros, 'mutated': zeros},
                               'BacteriaVictories': 0, 'UnusedVictories': 0}
        for i in range(iterations):
            new_seed = rng.integers(10000000)
            new_rng = np.random.default_rng(new_seed)
            init_world(world, totalPopDensity, BacterialDensity, args, new_rng)

            for _ in range(epochs):
                stats = forward_func(world=world, mooreDomain=mooreDomain, prevStats=stats, rng=rng,
                                     **forward_func_args)

            # meanSimulationStats = update_mean_stats(stats, meanSimulationStats, iterations)

        gen_save_plots(epochs, meanSimulationStats, figurePath / expName, iterations)


def color_world(world: List[List[Cell]]):
    # translate the world into a np.ndarray with the specified color values
    coloredWorld = [[None for _ in range(len(world))] for _ in range(len(world))]

    for i in range(len(world)):
        for j in range(len(world)):
            cell = world[i][j]
            if cell is None:
                coloredWorld[i][j] = colors['empty']
            else:
                coloredWorld[i][j] = colors[cell.type][cell.state]

    return coloredWorld


def clear_plots():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def gen_double_linechart(x, y1, y2, line1Legend, line2Legend, xlabel, ylabel, plotTitle,
                         filepath, filename, dpi=300):
    """ linechart with two lines.  """

    plt.plot(x, y1, label=line1Legend)
    plt.plot(x, y2, label=line2Legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plotTitle)
    plt.legend()
    plt.savefig(f'{filepath}/{filename}.png', dpi=dpi)


def gen_save_plots(epochs, stats, path, iterations):
    clear_plots()

    x = list(range(epochs))

    meanSim = '' if iterations == 1 else '(Mean) '
    # total population numbers
    gen_double_linechart(x=x, y1=stats['Bacteria']['total'], y2=stats['Unused']
                         ['total'], line1Legend='Bacteria cells',
                         line2Legend='Unused cells', xlabel='Epoch',
                         ylabel=f'{meanSim}Number of cells',
                         plotTitle=f'{meanSim}Total population evolution',
                         filepath=path, filename='total_population_evolution')

    clear_plots()
    # healthy cells
    gen_double_linechart(x=x, y1=stats['Bacteria']['Healthy'], y2=stats['Unused']['Healthy'], line1Legend='Stable cells',
                         line2Legend='Unused cells', xlabel='Epoch', ylabel=f'{meanSim}Number of healthy cells',
                         plotTitle=f'{meanSim}Healthy population evolution', filepath=path,
                         filename='Healthy_population_evolution')
    clear_plots()
    # mutated cells
    gen_double_linechart(x=x, y1=stats['Bacteria']['mutated'], y2=stats['Unused']['mutated'], line1Legend='Stable cells',
                         line2Legend='Unused cells', xlabel='Epoch', ylabel=f'{meanSim}Number of mutated cells',
                         plotTitle=f'{meanSim}Mutated population evolution', filepath=path,
                         filename='mutated_population_evolution')
    # damaged cells
    clear_plots()
    gen_double_linechart(x=x, y1=stats['Bacteria']['damaged'], y2=stats['Unused']['damaged'], line1Legend='Stable cells',
                         line2Legend='Unused cells', xlabel='Epoch', ylabel=f'{meanSim}Number of damaged cells',
                         plotTitle=f'{meanSim}Damaged population evolution', filepath=path,
                         filename='damaged_population_evolution')

    # barplot with number of cell type victories
    if iterations > 1:
        clear_plots()
        plt.bar([0, 1], [stats['BacteriaVictories'], stats['UnusedVictories']])
        plt.xlabel('Cell type')
        plt.ylabel('Number of iterations won')
        plt.title('Iterations won by each cell type')
        plt.xticks([0, 1], ['BacteriaVictories', 'UnusedVictories'])
        plt.savefig(f'{path}/generations_won.png', dpi=300)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_all_params(args):
    params = str(vars(args)).split(',')  # ["{'a': 1", ..., "'b': 2}",]
    params = [p[:p.find(':')] + '\n' + p[p.find(':'):] for p in params]
    params = ['--' + param for param in params]  # ["--{'a': \n 1", ..., "--'b': \n 2}"]
    params = '\n'.join(params)  # "--{'a': \n 1 \n ... \n --'b': \n 2}"
    params = params.replace('{', '').replace('}', '')  # "--'a': \n 1 \n ... \n --'b': \n 2"
    params = params.replace("'", '').replace(':', '')  # "--a \n 1 \n ... \n --b \n 2"
    return params.replace(' ', '')
