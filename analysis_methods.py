import numpy as np
from ptu import PTU
import matplotlib.pyplot as plt

def pca(aug_samples):
    # Do PCA by hand since we want to save off the covariance matrices
    cov_matrix = np.cov(aug_samples.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    sort_indices = np.argsort(eig_vals)

    # Sort by largest to smallest eigenvalue
    eig_vals = eig_vals[sort_indices][::-1]
    eig_vecs = eig_vecs[sort_indices][::-1]
    variance_percents = np.abs(eig_vals) / np.sum(np.abs(eig_vals))
    subspace_dict = {
        'cov_matrix': cov_matrix,
        'eig_vals': eig_vals,
        'eig_vecs': eig_vecs,
        'variance_percents': variance_percents
    }
    return subspace_dict

def dist_distribution(aug_samples):
    aug_dists = np.sqrt(np.sum(aug_samples ** 2, -1))
    aug_dist_hist = plt.hist(aug_dists, bins=50)
    plt.title('Distribution of distances by augmentation')
    plt.show()
    plt.clf()

def ptu_analysis(aug_samples):
    # Why does this give a linear histogram???
    model = PTU(
        aug_samples,
        n_neighbors=20,
        geod_n_neighbors=30,
        embedding_dim=2
    )
    embedded = model.fit()

    # Get matrix of pairwise distances
    original_dists = model.ptu_dists
    embedded_dists = np.expand_dims(embedded, 0) - np.expand_dims(embedded, 1)
    embedded_dists = np.sqrt(np.sum(np.square(embedded_dists), -1))

    # remove diagonal elements
    original_dists = original_dists[
        ~np.eye(original_dists.shape[0],dtype=bool)
    ].reshape(original_dists.shape[0],-1)
    embedded_dists = embedded_dists[
        ~np.eye(embedded_dists.shape[0],dtype=bool)
    ].reshape(embedded_dists.shape[0],-1)

    # Look at histogram of difference between
    # indices of sorted geodesic distances and
    # sorted embedded euclidean distances
    original_shape = list(original_dists.shape)
    original_dists = np.reshape(original_dists, -1)
    embedded_dists = np.reshape(embedded_dists, -1)

    original_sort_indices = np.argsort(original_dists)
    embedded_sort_indices = np.argsort(embedded_dists)
    index_delta = np.abs(original_sort_indices - embedded_sort_indices)
    max_index_delta = len(aug_samples)**2
    num_bins = 50
    bins = [i * max_index_delta / num_bins for i in range(num_bins)]
    index_delta = plt.hist(index_delta, bins=bins)
    plt.title('Absolute change in sort indices')
    plt.show()
    plt.clf()
    plt.scatter(embedded[:, 0], embedded[:, 1])
    plt.show()
