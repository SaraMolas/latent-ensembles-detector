"""
Functions required for performing FastICA
"""
import numpy as np
import math
from typing import Tuple, Optional
import scipy.stats as stats
from sklearn.decomposition import FastICA

def compute_spike_matrix (spikeTimes: np.ndarray, spikeClusters: np.ndarray, time: np.ndarray, start_time: float,
                           end_time: float, sampling_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spike matrix from spike times and clusters"""

    timestamps = np.arange(time[0],math.ceil(time[-1]), 0.025) # get timestamps in 25 ms bins
    cell_IDs = np.unique(spikeClusters) # get number of cells
    spike_matrix = np.zeros((len(cell_IDs), len(timestamps))) # create matrix with zeros
    n_bins = len(timestamps) 

    # Fill matrix with spike count for each neuron per timebin
    # then iterate through number of place cells in place cells list
    for counter, cell in enumerate(cell_IDs):

        cell_indexes = np.where(spikeClusters == cell) # find the indices of the spikes of that cell
        spike_times_OE = spikeTimes[cell_indexes] # extract times at which that cell fired

        # need to convert times from Openephys times to seconds or milliseconds
        spike_times = spike_times_OE / sampling_rate # divide by sampling rate to get seconds

        # Extract spikes from the recording session (only matters if there is multiple sessions in one recording)
        spike_times_session = spike_times[(spike_times >= start_time) & (spike_times <= end_time)] # extract spikes from that session
        
        # then allocate 1s in the matrix in the timebins when it fired - use histogram function
        binnedSpikes, _= np.histogram(spike_times_session, bins = n_bins)
        zBinnedSpikes = stats.zscore(binnedSpikes) # z-score the binned spikes
        spike_matrix[counter, :] = zBinnedSpikes # fill the matrix with the z-scored binned spikes

    # Remove rows with all NaNs
    spike_matrix_clean = spike_matrix
    cell_IDs_clean = cell_IDs
    row = 0
    for it in range(spike_matrix.shape[0]):
        if np.isnan(spike_matrix_clean[row,:]).all() == True:
            spike_matrix_clean = np.delete(spike_matrix_clean, row, 0)
            cell_IDs_clean = np.delete(cell_IDs_clean , row, 0)
            row = row
        else:
            row = row + 1
        
    return spike_matrix_clean, cell_IDs_clean

def estimate_ensembles_number(spike_matrix: np.ndarray) -> int:
    """Estimate number of ensembles from spike matrix using Marcenko-Pastur distribution"""
    
    #  Create covariance (= correlation) matrix of the spike matrix.
    cov_matrix = np.matmul(spike_matrix,np.transpose(spike_matrix)) / spike_matrix.shape[1]
    
    # remove rows with all NaNs
    cov_matrix_clean = cov_matrix
    row = 0
    for it in range(cov_matrix.shape[0]):
        if np.isnan(cov_matrix[row,:]).all() == True:   
            cov_matrix_clean = np.delete(cov_matrix, row, 0)
            cov_matrix_clean = np.delete(cov_matrix, row, 1)
            row = row
        else: 
            row = row + 1

    # Find number of total eigenvalues
    eigvals = np.linalg.eigvalsh(cov_matrix)

    # Obtain number of significant eigenvalues based on Marcenko-Pastur distribution
    q = spike_matrix.shape[1] / spike_matrix.shape[0] # compute number of columns divided by number of rows
    s = 1 # set variance of matrix, should be 1 bc of normalization
    boundMax = s * ((1 + math.sqrt(1/q))**2) # find upper bound
    sig_eig_vals = np.where(eigvals > boundMax) # find eigenvalues above upper bound
    n_ensembles = len(sig_eig_vals[0])
    
    if n_ensembles == 0: # if there are no coactivation patterns, return error
        raise ValueError("No neural ensembles found")
    else: 
        print("number of estimated assemblies :" + str(n_ensembles) + " out of " + str(len(eigvals)) + " eigenvalues")
        return n_ensembles
    
def perform_fastICA(n_ensembles: int, spike_matrix: np.ndarray, max_iter: Optional[int]= 2000) -> Tuple[np.ndarray, np.ndarray]:

    # then compute the independent components through the fastICA algorithm
    ICA = FastICA(n_components=n_ensembles, max_iter=max_iter)
    weights = ICA.fit_transform(spike_matrix) 

    # this step is exclusive to Van de Ven 2016: they scale to unit length the weight vector and set the set sign of vector so that highest absolute weight is positive
    for pat in range(weights.shape[1]):
        weights[:,pat] = weights[:,pat] / np.linalg.norm(weights[:,pat])
        if np.max(np.absolute(weights[:,pat])) == np.max(weights[:,pat]):
            weights[:,pat] = weights[:,pat] # keep the sign of vector if highest absolute weight is positive
        else:
            weights[:,pat] = -(weights[:,pat]) # change the sign of vector if highest absolute weight is negative

    return weights, ICA.mixing_

def find_principal_neurons (weights: np.ndarray) -> list:
    """Find principal neurons for each ensemble based on weights"""
    # iterate over weight vectors
    principalCells = [] # create empty list
    for pat in range(weights.shape[1]):
        # find the mean and std of those weights
        meanWeights = np.mean(weights[:,pat])
        stdWeights = np.std(weights[:,pat])
        
        # find the index of cells with weights 2 std above the mean = principal cells
        principalCells.append(np.where(weights[:,pat] > (meanWeights + 2 * stdWeights)))
    
    return principalCells

