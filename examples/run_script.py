#!/usr/bin/env python3
import argparse
from latent_ensembles_detector import (
    compute_spike_matrix, 
    estimate_ensembles_number, 
    perform_fastICA, 
    find_principal_neurons, 
    load_data, 
    save_data, 
    plot_principal_cells_heatmap
)

# Load data 
spike_times = load_data("data/spike_times.npy")
spike_clusters = load_data("data/spike_clusters.npy")
time = load_data("data/time.npy")

# Define experimental parameters
start_time = 0  # in seconds
end_time = 600  # in seconds
sampling_rate = 30000  # in Hz

spike_matrix, _ =  compute_spike_matrix (spikeTimes = spike_times, spikeClusters = spike_clusters, time = time, 
                                         start_time = start_time,  end_time = end_time, sampling_rate = sampling_rate)

n_ensembles =  estimate_ensembles_number(spike_matrix = spike_matrix)
    
weights, _ =  perform_fastICA(n_ensembles = n_ensembles, spike_matrix = spike_matrix)

principalCells = find_principal_neurons (weights = weights)

# Save results
save_data("results/spike_matrix.npy", spike_matrix)
save_data("results/weights.npy", weights)
save_data("results/principal_cells.json", principalCells)