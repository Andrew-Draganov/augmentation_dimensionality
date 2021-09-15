import json
from tqdm import tqdm
import pdb
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA

training_data = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=None
)

# Cifar normalization
normalize = T.Compose([
	T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# MNIST normalization
# normalize = T.Compose([
#     T.ToTensor(),
#     T.Normalize([0.1307], [0.3081])
# ])

def get_rotation_kwargs(epoch):
    scalar = float(180) / epochs_per_aug
    min_rotation = (epoch - 1) * scalar
    max_rotation = epoch * scalar
    rotation_sample = random.choice([-1, 1]) * random.uniform(min_rotation, max_rotation)
    return {'angle': rotation_sample}

def get_blur_kwargs(epoch):
    if epoch < 3:
        epoch += 2 # Size 1 kernel does nothing, so add 2 to get size 3 kernel
    if epoch % 2 == 1:
        return {'kernel_size': epoch}

    # Use half the default std. dev. and keep kernel same size as last time
    sigma = (0.3 * ((epoch) * 0.5 - 1) + 0.8) / 2
    return {'kernel_size': epoch - 1, 'sigma': sigma}

def get_gamma_kwargs(epoch):
    return {'gamma': epoch}

def get_crop_kwargs(epoch):
    return {'output_size': 32 - 2 * epoch}

epochs_per_aug = 10
num_samples = len(training_data)
n_components = 10
augmentation_list = {
    'center_crop': T.functional.center_crop,
    'gamma': T.functional.adjust_gamma,
    'gaussian_blur': T.functional.gaussian_blur,
    'rotation': T.functional.rotate,
}
kwargs_dict = {
    'center_crop': get_crop_kwargs,
    'gamma': get_gamma_kwargs,
    'gaussian_blur': get_blur_kwargs,
    'rotation': get_rotation_kwargs,
}

augmentation_subspaces = {}

# Sweep intensities or each augmentation, applying each intensity value to the entire dataset
# Then look at the subspace induced by the augmentation across its intensities
# How linear is it?
# Also would be interesting to look at the dimensionality of the low-dim manifold induced by aug
for aug_i, (aug_name, aug_func) in enumerate(augmentation_list.items()):
    aug_samples = np.zeros([num_samples * epochs_per_aug, 3072])
    pbar = tqdm(range(1, epochs_per_aug))
    for epoch in pbar:
        pbar.set_description('Epochs for %s' % aug_name)
        def augment(**kwargs):
            return aug_func(**kwargs)

        # Get keyword arguments for this augmentation at this epoch
        kwargs = kwargs_dict[aug_name](epoch)

        pbar2 = tqdm(enumerate(training_data), leave=False, total=num_samples)
        pbar2.set_description('Epoch %d' % epoch)
        for sample_i, (img, label) in pbar2:
            img = normalize(img)
            kwargs['img'] = img
            aug_img = augment(**kwargs)
            if list(aug_img.shape) != list(img.shape):
                aug_img = T.functional.resize(aug_img, list(img.shape)[1:])
            img, aug_img = np.array(img), np.array(aug_img)

            # Subtract original image from augmented image
            delta = np.reshape(img - aug_img, -1)
            index = num_samples * epoch + sample_i
            aug_samples[index, :] = delta

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
    augmentation_subspaces[aug_name] = subspace_dict
    print(aug_name, ' : ', subspace_dict['variance_percents'][:n_components])

filename = 'augmentation_subspaces.npy'
np.save(filename, augmentation_subspaces)
# Load with `d = np.load(filename, allow_pickle=True)`
# Access with `d = d[()]`
