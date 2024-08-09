# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:12:23 2024

@author: Le Roy, chatGTP version 17/05/2024.
original prompt:
"Create a python script that plots the poisson distribution based on input parameters.
the system of interest is about bacteria and bacteriophages.
I want the distribution of infected cells for a specific ratio between phages and bacteria (moi).
If possible add another layer so I can play with the likely infection after a certain time frame
based on infection speed. "
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def poisson_distribution(mois, infection_rate, time_factor, max_cells=20):
    """
    Plot the Poisson distributions for a list of MOIs and a given time factor.
    Also, print the total number of infected cells excluding zero infections for each MOI.
    
    Parameters:
    - mois: List of multiplicities of infection (phages per bacterium)
    - time_factor: Factor by which infection likelihood increases over time
    - max_cells: Maximum number of cells to consider in the plot
    """
    # Define colors for different MOIs
    colors = ['skyblue', 'salmon', 'lightgreen', 'lightcoral']

    # Plot the distributions
    plt.figure(figsize=(10, 6))
    for i, moi in enumerate(mois):
        # Calculate adjusted MOI based on time factor
        adjusted_moi = moi * infection_rate * time_factor

        # Generate the range of number of infected cells
        k = np.arange(0, max_cells + 1)

        # Calculate the Poisson probabilities
        poisson_probabilities = stats.poisson.pmf(k, adjusted_moi)

        # Calculate the expected number of infected cells excluding zero infections
        expected_infected_cells_excluding_zero = sum(poisson_probabilities[1:])

        # Print the total number of infected cells excluding zero infections
        print(f'Total number of infected cells (excluding zero infections) for MOI = {moi}: {expected_infected_cells_excluding_zero:.2f}')

        # Plot the distribution
        plt.bar(k, poisson_probabilities, color=colors[i], alpha=0.6, label=f'MOI = {moi}')

    plt.title(f'Poisson Distribution for MOIs = {mois}, Infection Rate = {infection_rate}, and Time Factor = {time_factor}')
    plt.xlabel('Infections per Cell')
    plt.ylabel('Probability')
    plt.xticks(k)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage
    mois = [1, 2.5, 5, 10]
    infection_rate = float(input("Enter the infection rate (between 0 and 1): "))
    time_factor = float(input("Enter the time factor (e.g., 1.0 for initial, 2.0 for doubled infection rate): "))
    poisson_distribution(mois, infection_rate, time_factor)


