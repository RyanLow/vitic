import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from modules.model import ViTIC
from modules.data import load_dataset, display_images, plot_embedding, Augmentation
from modules.metrics import NMI, ARI, FMI, ACC
from modules.utils import yaml_config_hook


def eval():
    parser = argparse.ArgumentParser()
    config = yaml_config_hook(sys.argv[1])
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    a = parser.parse_args([])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset, class_labels = load_dataset(a.dataset, a.data_path, Augmentation(a.image_size, test=True))
    num_classes = len(class_labels)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=a.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=a.num_workers,
    )

    model = ViTIC(a.feature_dim, num_classes, a.projector_dim, a.image_size)
    checkpoint = torch.load(a.model_path, map_location=device.type)
    model_id = re.search(r'/([^/]+).tar', a.model_path).group(1)
    model.load_state_dict(checkpoint['weights'])
    model.to(device)

    model.eval()
    clusters_vector = []
    labels_vector = []
    vit_outputs = []
    instance_proj_outputs = []
    for step, ((x, _), y) in tqdm(enumerate(data_loader)):
        x = x.to(device)
        with torch.no_grad():
            h, z, c = model.evaluate(x)
        clusters_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        vit_outputs.extend(h.cpu().detach().numpy())
        instance_proj_outputs.extend(z.cpu().detach().numpy())
    clusters_vector = np.array(clusters_vector)
    labels_vector = np.array(labels_vector)
    vit_outputs = np.array(vit_outputs)
    instance_proj_outputs = np.array(instance_proj_outputs)

    if not os.path.exists(f'{a.results_path}/{a.dataset}/{model_id}'):
        os.makedirs(f'{a.results_path}/{a.dataset}/{model_id}')

    plt.plot(checkpoint['loss_hist'])
    plt.title('Training loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{a.results_path}/{a.dataset}/{model_id}/loss.png')

    images = []
    labels = []
    for c in range(num_classes):
        cluster_idx = np.flatnonzero((clusters_vector == c) & (labels_vector != -1))
        sample = np.random.choice(cluster_idx, a.num_images, replace=False)
        images.extend([np.moveaxis(np.array(dataset[i][0][1]), 0, -1) for i in sample])
        labels.extend([class_labels[labels_vector[i]] for i in sample])
    display_images(images, (num_classes, a.num_images), f'{a.results_path}/{a.dataset}/{model_id}/examples.png', labels)

    plt_samples = np.random.choice(np.flatnonzero(labels_vector != -1), a.num_embeddings, replace=False)

    pca_vit = PCA(n_components=2).fit_transform(vit_outputs)
    plot_embedding(pca_vit[plt_samples], clusters_vector[plt_samples], f'{a.results_path}/{a.dataset}/{model_id}/pca_vit.png')

    pca_inst = PCA(n_components=2).fit_transform(instance_proj_outputs)
    plot_embedding(pca_inst[plt_samples], clusters_vector[plt_samples], f'{a.results_path}/{a.dataset}/{model_id}/pca_inst.png')

    tsne_vit = TSNE(init="pca", n_jobs=2, random_state=0).fit_transform(vit_outputs)
    plot_embedding(tsne_vit[plt_samples], clusters_vector[plt_samples], f'{a.results_path}/{a.dataset}/{model_id}/tsne_vit.png')

    tsne_inst = TSNE(init="pca", n_jobs=2, random_state=0).fit_transform(instance_proj_outputs)
    plot_embedding(tsne_inst[plt_samples], clusters_vector[plt_samples], f'{a.results_path}/{a.dataset}/{model_id}/tsne_inst.png')

    loss = pd.DataFrame({'loss': checkpoint['loss_hist']})
    loss.to_csv(f'{a.results_path}/{a.dataset}/{model_id}/loss.csv')

    if a.dataset == 'STL10':
        clusters_vector = clusters_vector[labels_vector != -1]
        labels_vector = labels_vector[labels_vector != -1]
    nmi = NMI(labels_vector, clusters_vector)
    ari = ARI(labels_vector, clusters_vector)
    fmi = FMI(labels_vector, clusters_vector)
    acc = ACC(labels_vector, clusters_vector)
    scores = pd.DataFrame([[nmi, ari, fmi, acc]], columns=['NMI', 'ARI', 'FMI', 'ACC'])
    scores.to_csv(f'{a.results_path}/{a.dataset}/{model_id}/scores.csv')


if __name__ == '__main__':
    eval()
