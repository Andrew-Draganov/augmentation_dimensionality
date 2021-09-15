import numpy as np
import random

def enumerate2(xs, start=0, step=1):
    i = start
    for count in range(len(xs)):
        try:
            yield (count, xs[i])
        except:
            break
        i += step

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

# Augmentation kwarg generator
def get_rotation_kwargs(epoch, epochs_per_aug):
    scalar = float(180) / epochs_per_aug
    min_rotation = (epoch - 1) * scalar
    max_rotation = epoch * scalar
    rotation_sample = random.choice([-1, 1]) * random.uniform(min_rotation, max_rotation)
    return {'angle': rotation_sample}

def get_blur_kwargs(epoch, epochs_per_aug):
    if epoch < 3:
        epoch += 2 # Size 1 kernel does nothing, so add 2 to get size 3 kernel
    if epoch % 2 == 1:
        return {'kernel_size': epoch}

    # Use half the default std. dev. and keep kernel same size as last time
    sigma = (0.3 * ((epoch) * 0.5 - 1) + 0.8) / 2
    return {'kernel_size': epoch - 1, 'sigma': sigma}

def get_gamma_kwargs(epoch, epochs_per_aug):
    return {'gamma': epoch}

def get_crop_kwargs(epoch, epochs_per_aug):
    return {'output_size': 32 - 2 * epoch}

