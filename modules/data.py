import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from PIL import ImageFilter, ImageOps
from sklearn.preprocessing import MinMaxScaler


def load_dataset(dataset, data_path, transform=None):
    """
    Loads "CIFAR10", "CIFAR100", or "STL10" datasets from torchvision.
    Any other arguments result in an exception.
    """
    if dataset == "CIFAR10":
        train_data = CIFAR10(data_path, train=True, transform=transform, download=True)
        test_data = CIFAR10(data_path, train=False, transform=transform, download=True)
        complete_data = torch.utils.data.ConcatDataset([train_data, test_data])
    elif dataset == "CIFAR100":
        def to_coarse(targets):
            coarse_labels = np.array([4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                      3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                      6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                                      0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                      5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                      16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                      10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                                      2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                      16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                      18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
            return coarse_labels[targets]
        superclasses = ["aquatic mammals", "fish", "flowers", "food containers",
                        "fruit and vegetables", "household electrical devices",
                        "household furniture", "insects", "large carnivores",
                        "large man-made outdoor things", "large natural outdoor scenes",
                        "large omnivores and herbivores", "medium-sized mammals",
                        "non-insect invertebrates", "people", "reptiles", "small mammals",
                        "trees", "vehicles 1", "vehicles 2"]
        train_data = CIFAR100(data_path, train=True, transform=transform, download=True)
        test_data = CIFAR100(data_path, train=False, transform=transform, download=True)
        train_data.targets = to_coarse(train_data.targets)
        test_data.targets = to_coarse(test_data.targets)
        train_data.classes = superclasses
        test_data.classes = superclasses
        complete_data = torch.utils.data.ConcatDataset([train_data, test_data])
    elif dataset == "STL10":
        train_data = STL10(data_path, split="train", transform=transform, download=True)
        test_data = STL10(data_path, split="test", transform=transform, download=True)
        unlabeled_data = STL10(data_path, split="unlabeled", transform=transform, download=True)
        complete_data = torch.utils.data.ConcatDataset([train_data, test_data, unlabeled_data])
    else:
        raise NotImplementedError(f"dataset {dataset} not supported by this function")
    class_labels = train_data.classes
    return complete_data, class_labels


def display_images(images, grid, path, labels=None):
    """
    Displays images in a grid with optional labels.
    """
    r, c = grid
    fig = plt.figure(figsize=(3*c, 3*r))
    for i, im in enumerate(images):
        ax = plt.subplot(r, c, i+1)
        plt.imshow(im.astype("uint8"))
        if labels:
            plt.title(labels[i])
        plt.axis("off")
    plt.savefig(path, bbox_inches='tight', pad_inches = 0)


def plot_embedding(X, y, path, title=None):
    _, ax = plt.subplots()
    X = MinMaxScaler().fit_transform(X)
    y_unique = np.unique(y)
    for i in y_unique:
        ax.scatter(
            *X[y == i].T,
            marker=f"${i}$",
            s=30,
            color=plt.cm.turbo(i / len(y_unique)),
            alpha=0.425,
            zorder=2,
        )
    ax.set_title(title)
    ax.axis("off")
    plt.savefig(path)


#### Random Augmentation Transform


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    def __call__(self, x):
        return ImageOps.solarize(x)


class Augmentation:
    def __init__(self, size, test=False):
        self.train_transform_1 = [
            torchvision.transforms.RandomResizedCrop(size=size, scale=(0.08, 1.)),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                                               p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.8),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]

        self.train_transform_2 = [
            torchvision.transforms.RandomResizedCrop(size=size, scale=(0.08, 1.)),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                                               p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.8),
            torchvision.transforms.RandomApply([Solarize()], p=0.2),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]

        self.test_transform = [
            torchvision.transforms.Resize(size=(size, size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]

        self.train_transform_1 = torchvision.transforms.Compose(self.train_transform_1)
        self.train_transform_2 = torchvision.transforms.Compose(self.train_transform_2)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)
        self.to_tensor = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])

        self.test = test

    def __call__(self, x):
        if self.test:
            return self.test_transform(x), self.to_tensor(x)

        return self.train_transform_1(x), self.train_transform_2(x)
