# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 06:48:45 2024

@author: Le Roy
Utility variables, methods and classes

"""

import argparse
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm


def init_world(world, BacterialDensity, BacterialQuantity, radius, PhageConcentration,
               PhageQuantity, worldSize, NutrienceDensity):
    """ Initialise lattice with Bacteria.
        There are density * len(world) Bacterial cells placed. """
    center = worldSize // 2

    # Create a mask for positions within the circle
    y, x = np.ogrid[:worldSize, :worldSize]
    mask = (x - center)**2 + (y - center)**2 <= radius**2

    # Bacteria Layer: Set the Bacteria Density
    BacterialLayer = world[:, :, 0]
    num_values = int(worldSize * worldSize * BacterialDensity)
    flat_indices = np.random.choice(worldSize * worldSize, num_values, replace=False)
    indices = np.unravel_index(flat_indices, (worldSize, worldSize))
    BacterialLayer[indices] = BacterialQuantity
    world[:, :, 0] = BacterialLayer

    # Phage Layer: Generate random values and apply the mask
    phage_layer = world[:, :, 2]
    random_values = np.random.rand(worldSize, worldSize)
    phage_positions = random_values <= PhageConcentration
    phage_layer[mask & phage_positions] = PhageQuantity


    return


def update_Nutrience(prevStats, totalNutrience):
    percentageNutrience = 1 / totalNutrience * prevStats['Nutrience']['Available']
    return percentageNutrience


def find_circle_radius(world):
    world_size = world.shape[0]
    center = world_size // 2

    for i in range(center):
        if world[center, center + i, 0] > 1:
            return i  # return the radius from the center

    return -1  # Return -1 if no circle is found (error case)


def save_stats_to_file(stats, file_path, timesteps):
    with open(file_path, 'w') as file:
        file.write('TimeStep\tTotalBacteria\tHealthyBacteria\tInfectedBacteria\tPhages\tNutrience\n')
        for i in range(timesteps):
            total_bacteria = stats['Bacteria']['total'][i]
            healthy_bacteria = stats['Bacteria']['Healthy'][i]
            infected_bacteria = stats['Bacteria']['Infected'][i]
            phages = stats['P']['Infected'][i]  # Adjust as per your actual data structure
            nutrience = stats['Nutrience']['Available'][i] if 'Available' in stats['Nutrience'] else 'N/A'
            file.write(f'{i+1}\t{total_bacteria}\t{healthy_bacteria}\t{infected_bacteria}\t{phages}\t{nutrience}\n')


def simulation_skeleton(args, Forward_Function):

    expName = f'experiment_{datetime.now().strftime("%d%m%Y_%H%M%S")}'
    currentPath = pathlib.Path(__file__).parent.resolve()
    figurePath = currentPath.parent.resolve() / 'experiment_results/'
    if not figurePath.exists():
        os.mkdir(figurePath)
    os.mkdir(figurePath / expName)

    # save parameters of the experiment
    with open(figurePath / expName / "params.txt", "w") as file:
        file.write(parse_all_params(args))

    # create world, then will with bacteria and phages
    world = np.zeros((args.worldSize, args.worldSize, 3))
    init_world(world, args.BacterialDensity, args.BacterialQuantity, args.Phageradius,
               args.PhageConcentration, args.PhageQuantity, args.worldSize, args.NutrienceDensity)

    # Set nutrience for the system
    totalNutrience = (args.worldSize ** 2) * args.NutrienceDensity

    # initiate empty stat list
    stats = {'Bacteria': {'total': [], 'Healthy': [], 'Infected': []},
             'Phages': {'Free': []},
             'Nutrience': {'Available': totalNutrience}}

    for i in tqdm(range(args.timesteps)):
        stats = Forward_Function(world, args.worldSize, args.replicationRate, args.InfectionRate,
                                 args.LatencyPeriod, args.BurstSize, args.BactDiffusionRate,
                                 args.PhageDiffusionRate, totalNutrience,
                                 stats)

    # TODO: phage plane should not be limited to 1200
    for Plane, Limit in [(2, None), (1, 100), (0, 100)]:
        plot_layer(world, Plane, Limit, figurePath / expName)

    gen_save_plots(args.timesteps, stats, figurePath / expName)

    # save final radius to file
    final_radius = find_circle_radius(world)
    with open(figurePath / expName / "final_radius.txt", "w") as file:
        file.write(f"{expName}\t{final_radius}\n")

    with open(figurePath / expName / 'out.txt', 'w', newline='') as file:
        file.write("Healty\tInfected\tPhages\n")
        for i in range(args.timesteps):
            file.write(f"{stats['Bacteria']['Healthy'][i]}\t{stats['Bacteria']['Infected'][i]}\t{stats['Phages']['Free'][i]}\n")

    return final_radius


def clear_plots():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def gen_double_linechart(x, y1, y2, line1Legend, line2Legend, xlabel, ylabel,
                         plotTitle, filepath, filename, dpi=300):
    """ linechart with two lines.  """

    # Adjust the length of y1 and y2 to match the length of x
    y1_adjusted = y1[:len(x)] if len(y1) >= len(x) else np.pad(y1, (0, len(x) - len(y1)),
                                                               mode='constant')
    y2_adjusted = y2[:len(x)] if len(y2) >= len(x) else np.pad(y2, (0, len(x) - len(y2)),
                                                               mode='constant')

    plt.plot(x, y1_adjusted, label=line1Legend)
    plt.plot(x, y2_adjusted, label=line2Legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plotTitle)
    plt.legend()
    plt.savefig(f'{filepath}/{filename}.png', dpi=dpi)


def gen_single_linechart(x, y1, line1Legend, xlabel, ylabel, plotTitle,
                         filepath, filename, dpi=300):
    """ linechart with two lines.  """

    plt.plot(x, y1, label=line1Legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plotTitle)
    plt.legend()
    plt.savefig(f'{filepath}/{filename}.png', dpi=dpi)


def gen_save_plots(timesteps, stats, path):
    clear_plots()

    x = list(range(timesteps))

    # total population numbers
    gen_single_linechart(x=x, y1=stats['Bacteria']['total'],
                         line1Legend='Bacteria cells',
                         xlabel='timesteps',
                         ylabel='Number of cells',
                         plotTitle='Total population evolution',
                         filepath=path, filename='total_population_evolution')

    clear_plots()
    # healthy cells
    gen_double_linechart(x=x, y1=stats['Bacteria']['Healthy'],
                         y2=stats['Bacteria']['Infected'],
                         line1Legend='Healthy bacterial cells',
                         line2Legend='Infected bacterial cells',
                         xlabel='timesteps',
                         ylabel='Number of healthy cells',
                         plotTitle='Healthy population evolution',
                         filepath=path,
                         filename='Bacterial_population_evolution')


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
    # ["--{'a': \n 1", ..., "--'b': \n 2}"]
    params = ['--' + param for param in params]
    params = '\n'.join(params)  # "--{'a': \n 1 \n ... \n --'b': \n 2}"
    # "--'a': \n 1 \n ... \n --'b': \n 2"
    params = params.replace('{', '').replace('}', '')
    # "--a \n 1 \n ... \n --b \n 2"
    params = params.replace("'", '').replace(':', '')
    return params.replace(' ', '')


def plot_layer(world, plane, Limit, path, dpi=300):
    """Plot the phage layer of the world as a heatmap."""
    Names = ('Healthy bacteria', 'Infected bacteria', 'Phages', 'Nutrience')
    filename = Names[plane]
    phage_layer = world[:, :, plane]

    plt.figure(figsize=(10, 10))
    plt.imshow(phage_layer, cmap='viridis', origin='lower', vmin=0, vmax=Limit)
    plt.colorbar(label=f'{filename} Presence')
    plt.title(f'{filename} Layer Heatmap')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')

    plt.savefig(f'{path}/{filename}.png', dpi=dpi)
    plt.show()
