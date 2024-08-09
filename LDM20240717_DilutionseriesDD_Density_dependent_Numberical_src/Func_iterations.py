# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 03:18:26 2024

@author: Le Roy
"""

import numpy as np
import scipy.stats as stats


def autoreplicate(world, worldSize, replicationRate, percentageNutrience, K=93):
    """ Autoreplication step. The healthy and infected bacterial layers are replicated according to
    the replication rate. replication rate is limited by carrying capacity K, and the percentage
    nutrients available in the specific cell.
    """

    BacterialLayer = world[:, :, 0]
    InfectedLayer = world[:, :, 1]

    NewBacterialLayer = (
        replicationRate * BacterialLayer *
        ((K - (BacterialLayer + InfectedLayer))/K) *
        (percentageNutrience)
    )

    NewInfectedLayer = (
        0.4 * replicationRate * InfectedLayer *
        ((K - (BacterialLayer + InfectedLayer))/K) *
        (percentageNutrience)
    )

    NewBacteria = np.sum(NewInfectedLayer) + np.sum(NewBacterialLayer)
    world[:, :, 0] += NewBacterialLayer
    world[:, :, 1] += NewInfectedLayer
    return world, NewBacteria


def Moore_diffuse_layer(world, diffusionRate, Plane):
    """Diffusion of cells and phages occures to their serounding 8 positions. replication occures
    at each time step and splits part of the center value in 8 (dependent on individual diffusion
    rate).
    """
    # Create shifted versions of the layer for all 8 neighbors
    layer = world[:, :, Plane]
    north = np.roll(layer, -1, axis=0)
    south = np.roll(layer, 1, axis=0)
    west = np.roll(layer, -1, axis=1)
    east = np.roll(layer, 1, axis=1)
    north_west = np.roll(north, -1, axis=1)
    north_east = np.roll(north, 1, axis=1)
    south_west = np.roll(south, -1, axis=1)
    south_east = np.roll(south, 1, axis=1)

    # Calculate the spread amount
    spread_amount = np.divide(np.multiply(diffusionRate, layer), 8)
    # Update the layer with the spread amount
    layer -= spread_amount * 4 + spread_amount * 4 / np.sqrt(2)
    surrounding = (north + south + west + east + (north_west + north_east + south_west + south_east)/np.sqrt(3))

    layer += np.divide(np.multiply(diffusionRate, surrounding), 8)

    world[:, :, Plane] = layer



def sigmoid(x):
    # y = (1 + np.exp(-30 * x)) ** (-10)
    y = 1 - x ** 3
    # y = (1-x)^2
    # y = np.exp(-5 * x)
    return y


def Infection_DD(world, InfectionRate, percentageNutrience, K=93):
    '''
    Infection of healty cells is dictated by the number of healty cells, the number of phages,
    the nutrient availability and the infection rate. number of phages that will infect is chosen,
    the percentage cells that will be uneffected at the specific MOI is then calculated.
    the infecting pahges are taken away from the phage plane, the infected cells are taken from the
    healthy plane and placed in the infected plain.
    '''
    PhageLayer = world[:, :, 2]
    BacterialLayer = world[:, :, 0]

    BacterialDensity = (1/K*BacterialLayer)
    a = sigmoid(BacterialDensity)

    InfectingPhages = PhageLayer * InfectionRate
    SuccesfullInfection = InfectingPhages * (percentageNutrience) * a
    MOI = np.divide(SuccesfullInfection, BacterialLayer, out=np.zeros_like(PhageLayer),
                    where=BacterialLayer != 0)
    infection_mask = MOI != 0

    Uninfected = stats.poisson.pmf(0, MOI)

    Infected = 1 - Uninfected

    world[:, :, 2][infection_mask] = np.subtract(PhageLayer[infection_mask],
                                                 InfectingPhages[infection_mask])
    world[:, :, 1] += np.multiply(BacterialLayer, Infected)
    world[:, :, 0] = np.multiply(world[:, :, 0], Uninfected)


def Infection_NDD(world, InfectionRate, percentageNutrience, K=93):
    '''
    Infection of healty cells is dictated by the number of healty cells, the number of phages,
    the nutrient availability and the infection rate. number of phages that will infect is chosen,
    the percentage cells that will be uneffected at the specific MOI is then calculated.
    the infecting pahges are taken away from the phage plane, the infected cells are taken from the
    healthy plane and placed in the infected plain.
    '''
    PhageLayer = world[:, :, 2]
    BacterialLayer = world[:, :, 0]

    # BacterialDensity = (1/K*BacterialLayer)
    # a = 1 - sigmoid(BacterialDensity)

    InfectingPhages = PhageLayer * InfectionRate
    SuccesfullInfection = InfectingPhages * (percentageNutrience) # * a
    MOI = np.divide(SuccesfullInfection, BacterialLayer, out=np.zeros_like(PhageLayer),
                    where=BacterialLayer != 0)
    infection_mask = MOI != 0

    Uninfected = stats.poisson.pmf(0, MOI)

    Infected = 1 - Uninfected

    world[:, :, 2][infection_mask] = np.subtract(PhageLayer[infection_mask],
                                                 InfectingPhages[infection_mask])
    world[:, :, 1] += np.multiply(BacterialLayer, Infected)
    world[:, :, 0] = np.multiply(world[:, :, 0], Uninfected)


def Lysation(world, LatencyPeriod, BurstSize, percentageNutrience):
    '''
    Lasis of infected cells happends on average after "LatencyPeriod". meaning in each step 1/Lp
    will be lysated. Lysed bacteria are removed from the phage layer and add to the phage plane
    dependent on the burst size, which scales with nutrient availability
    '''
    InfectedLayer = world[:, :, 1]
    LysatedInStep = InfectedLayer * LatencyPeriod

    PhagesInStep = LysatedInStep * (BurstSize * percentageNutrience)
    world[:, :, 1] -= LysatedInStep
    world[:, :, 2] += PhagesInStep


def update_statistics(world, stats, NewBacteria):
    """ Updates a dict with statistics of the current world """

    Bacteria = 0
    BacteriaHealthy = 0
    BacteriaInfected = 0
    Phages = 0
    BacteriaHealthy, BacteriaInfected, Phages = np.sum(world, (0, 1))
    Bacteria = BacteriaHealthy + BacteriaInfected

    # Append statistics to prevStats
    stats['Bacteria']['total'].append(Bacteria)
    stats['Bacteria']['Healthy'].append(BacteriaHealthy)
    stats['Bacteria']['Infected'].append(BacteriaInfected)
    stats['Phages']['Free'].append(Phages)
    stats['Nutrience']['Available'] = max(stats['Nutrience']['Available']-NewBacteria, 0)
    return stats


def update_Nutrience(prevStats, totalNutrience):
    percentageNutrience = 1 / totalNutrience * prevStats['Nutrience']['Available']
    return percentageNutrience


def Rounding_world(world):
    thresh = 0.00001
    super_threshold_indices = world < thresh
    world[super_threshold_indices] = 0


def Forward_Function(world, worldSize, replicationRate, InfectionRate, LatencyPeriod, BurstSize,
                     BactDiffusionRate, PhageDiffusionRate, totalNutrience, stats):

    percentageNutrience = update_Nutrience(stats, totalNutrience)

    world, NewBacteria = autoreplicate(world, worldSize, replicationRate, percentageNutrience)
    Infection_NDD(world, InfectionRate, percentageNutrience)
    Lysation(world, LatencyPeriod, BurstSize, percentageNutrience)

    Moore_diffuse_layer(world, BactDiffusionRate, slice(0, 1))
    Moore_diffuse_layer(world, PhageDiffusionRate, Plane=2)
    Rounding_world(world)

    stats = update_statistics(world, stats, NewBacteria)
    return stats
