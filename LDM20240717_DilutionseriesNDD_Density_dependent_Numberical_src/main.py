# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:26:43 2024

@author: Le Roy
Main code as addapted from Source

Density dependent variation
"""
# importing packages and other files
import argparse
import time

from Skeleton_Function import simulation_skeleton
from Func_iterations import Forward_Function

BacterialDensity = 0.0849  # 0.00416
timesteps = 1440
worldSize = 3500
replicationRate = 0.01  # Replication rate for bacteria
BactDiffusionRate = 0.01
PhageDiffusionRate = 0.006
LatencyPeriod = 1/180
BurstSize = 200
InfectionRate = 0.75
PhageConcentration = 68 / 7272205  # 0.00006012 * 0.2
Phageradius = 1650
BacterialQuantity = 1
PhageQuantity = 1
NutrienceDensity = 60000

NumberofRuns = 1

# temporary loop
DilutionSeries = True


def calculate_phage_concentrations():
    return [(68000 * 10**-x) / 7272205 for x in range(1, 7)]
# Main function, starts timer, sets parameters in 'parser',Specifies range for in case range = True


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser('Launch simulation', fromfile_prefix_chars="@")

    stateCaseHelp = """ Initial state of the cell population.
    'sparse' only has a few stable cells and unstable cells,
    while 'dense' has the board full of stable cells with a small cluster
    of already established mutator cells. """
    parser.add_argument('--simulationType', choices=['caseA', 'caseB'],
                        default='caseB', help=stateCaseHelp)

    BacterialDensityHelp = """ Bacterial population density, between 0 and 1."""

    parser.add_argument('--BacterialDensity', type=float, default=BacterialDensity,
                        help=BacterialDensityHelp)  # Corrected help parameter

    parser.add_argument('--timesteps', type=int, default=timesteps,
                        help='Number of generations to simulate.')

    parser.add_argument('--worldSize', type=int, default=worldSize,
                        help='Width and height of the grid.')

    parser.add_argument('--replicationRate', type=float, default=replicationRate,
                        help='Replication rate of stable cells.')

    parser.add_argument('--BactDiffusionRate', type=float, default=BactDiffusionRate,
                        help='Bacterial diffusion rate.')

    parser.add_argument('--PhageDiffusionRate', type=float, default=PhageDiffusionRate,
                        help='Phage diffusion rate.')

    parser.add_argument('--LatencyPeriod', type=float, default=LatencyPeriod,
                        help='Latency period.')

    parser.add_argument('--BurstSize', type=float, default=BurstSize,
                        help='Burst size.')

    parser.add_argument('--InfectionRate', type=float, default=InfectionRate,
                        help='Infection rate.')

    parser.add_argument('--PhageConcentration', type=float, default=PhageConcentration,
                        help='Phage Concentration.')

    parser.add_argument('--Phageradius', type=float, default=Phageradius,
                        help='Radius of phage spread.')

    parser.add_argument('--BacterialQuantity', type=float, default=BacterialQuantity,
                        help='''Phage concentration calculated by dividing number of phages
                        over area of phage circle.''')

    parser.add_argument('--NutrienceDensity', type=float, default=NutrienceDensity,
                        help='Number of nutrients on each cell in the plate.')

    parser.add_argument('--PhageQuantity', type=float, default=PhageQuantity,
                        help='The first parameter to be ranged.')

    args = parser.parse_args()

    final_radius_list = []
    if DilutionSeries:
        phage_concentrations = calculate_phage_concentrations()
        for phage_concentration in phage_concentrations:
            args.PhageConcentration = phage_concentration
            for _ in range(NumberofRuns):
                final_radius = simulation_skeleton(args, Forward_Function)
                final_radius_list.append((phage_concentration, final_radius))
    else:
        for _ in range(NumberofRuns):
            final_radius = simulation_skeleton(args, Forward_Function)
            final_radius_list.append((phage_concentration, final_radius))

    print("final radius is:", final_radius_list)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("run time was: {:.2f} seconds".format(elapsed_time))

    return


if __name__ == "__main__":
    main()
