import json
from tqdm import tqdm
import pdb
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from ptu import PTU
from utils import enumerate2, pca, get_rotation_kwargs, \
        get_blur_kwargs, get_gamma_kwargs, get_crop_kwargs

# MNIST dataset
training_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=None
)
normalize = T.Compose([
    T.ToTensor(),
    T.Normalize([0.1307], [0.3081])
])

# # CIFAR dataset
# training_data = torchvision.datasets.MNIST(
#     root='./data',
#     train=True,
#     download=True,
#     transform=None
# )
# # Cifar normalization
# normalize = T.Compose([
# 	T.ToTensor(),
#     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

epochs_per_aug = 1
stride = 100
num_samples = int(len(training_data) / stride)
n_components = 10
augmentation_list = {
    'rotation': T.functional.rotate,
    'center_crop': T.functional.center_crop,
    'gamma': T.functional.adjust_gamma,
    'gaussian_blur': T.functional.gaussian_blur,
}
kwargs_dict = {
    'rotation': get_rotation_kwargs,
    'center_crop': get_crop_kwargs,
    'gamma': get_gamma_kwargs,
    'gaussian_blur': get_blur_kwargs,
}
augmentation_subspaces = {}

# Sweep intensities or each augmentation, applying each intensity value to the entire dataset
# Then look at the subspace induced by the augmentation across its intensities
# How linear is it?
# Also would be interesting to look at the dimensionality of the low-dim manifold induced by aug
for aug_i, (aug_name, aug_func) in enumerate(augmentation_list.items()):
    # aug_samples = np.zeros([num_samples * epochs_per_aug, 3072]) # CIFAR-10
    aug_samples = np.zeros([num_samples * epochs_per_aug, 784]) # MNIST
    pbar = tqdm(range(epochs_per_aug))
    for epoch in pbar:
        pbar.set_description('Epochs for %s' % aug_name)
        def augment(**kwargs):
            return aug_func(**kwargs)

        pbar2 = tqdm(
            enumerate2(training_data, step=stride), 
            leave=False, 
            total=num_samples
        )
        pbar2.set_description('Epoch %d' % epoch)
        for sample_i, (img, label) in pbar2:
            img = normalize(img)

            # Get and apply keyword arguments for this augmentation
            kwargs = kwargs_dict[aug_name](epoch + 1, epochs_per_aug=epochs_per_aug)
            kwargs['img'] = img
            aug_img = augment(**kwargs)

            if list(aug_img.shape) != list(img.shape):
                aug_img = T.functional.resize(aug_img, list(img.shape)[1:])
            img, aug_img = np.array(img), np.array(aug_img)

            # Subtract original image from augmented image
            delta = np.reshape(img - aug_img, -1)
            index = num_samples * epoch + sample_i
            aug_samples[index, :] = delta

    # Perform PCA
    # subspace_dict = pca(aug_samples)
    # augmentation_subspaces[aug_name] = subspace_dict
    # print(aug_name, ' : ', subspace_dict['variance_percents'][:n_components])

    ptu = PTU(
        aug_samples,
        n_neighbors=10,
        geod_n_neighbors=15,
        embedding_dim=10
    )
    embedding = ptu.fit()
    original_dists = ptu.ptu_dists
    embedding_dists = np.expand_dims(embedding, 0) - np.expand_dims(embedding, 1)
    embedding_dists = np.sqrt(np.sum(np.square(embedding_dists), -1))

    # remove diagonal elements
    original_dists = original_dists[
        ~np.eye(original_dists.shape[0],dtype=bool)
    ].reshape(original_dists.shape[0],-1)
    embedding_dists = embedding_dists[
        ~np.eye(embedding_dists.shape[0],dtype=bool)
    ].reshape(embedding_dists.shape[0],-1)

    # Get ratio of embedding to geodesic distances
    dist_ratio = original_dists / embedding_dists
    zero_min = dist_ratio - np.min(dist_ratio)
    span = np.max(dist_ratio) - np.min(dist_ratio)
    dist_ratio = zero_min / span
    dist_ratio = (dist_ratio * 255).astype(np.uint8)
    img = Image.fromarray(dist_ratio, 'L')
    img.save('dist_ratio.png')
    img.show()
    print(np.mean(dist_ratio))
    print(np.var(dist_ratio))
    print(np.median(dist_ratio))

filename = 'augmentation_subspaces.npy'
np.save(filename, augmentation_subspaces)
# Load with `d = np.load(filename, allow_pickle=True)`
# Access with `d = d[()]`
