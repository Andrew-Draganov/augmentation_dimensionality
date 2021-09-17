from tqdm import tqdm
import pdb
import numpy as np
import torchvision
import torchvision.transforms as T
from PIL import Image
from utils import enumerate2, get_rotation_kwargs, \
        get_blur_kwargs, get_gamma_kwargs, get_crop_kwargs
from analysis_methods import pca, dist_distribution, ptu_analysis

# MNIST dataset
# training_data = torchvision.datasets.MNIST(
#     root='./data',
#     train=True,
#     download=True,
#     transform=None
# )
# normalize = T.Compose([
#     T.ToTensor(),
#     T.Normalize([0.1307], [0.3081])
# ])

# CIFAR dataset
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

epochs_per_aug = 10
stride = 50
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

# Sweep intensities for each augmentation, applying each intensity value to the entire dataset
# Then look at the subspace induced by the augmentation across its intensities
for aug_i, (aug_name, aug_func) in enumerate(augmentation_list.items()):
    # aug_samples = np.zeros([num_samples * epochs_per_aug, 784]) # MNIST
    aug_samples = np.zeros([num_samples * epochs_per_aug, 3072]) # CIFAR-10

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
    subspace_dict = pca(aug_samples)
    augmentation_subspaces[aug_name] = subspace_dict
    print(aug_name, ' : ', subspace_dict['variance_percents'][:n_components])

    # Look at distribution of distances of augmented image from original image
    dist_distribution(aug_samples)

    # Look at geodesic distances in augmentation space
    ptu_analysis(aug_samples)

filename = 'augmentation_subspaces.npy'
np.save(filename, augmentation_subspaces)
# Load with `d = np.load(filename, allow_pickle=True)`
# Access with `d = d[()]`
