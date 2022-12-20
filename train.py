import os
import sys
import numpy as np
import torch
import argparse
from tqdm import tqdm

from modules.model import ViTIC
from modules.objective import InstanceLoss, ClusterLoss
from modules.data import load_dataset, Augmentation
from modules.utils import save_model, yaml_config_hook


def train():
    parser = argparse.ArgumentParser()
    config = yaml_config_hook(sys.argv[1])
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    a = parser.parse_args([])

    if a.seed:
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        torch.cuda.manual_seed(a.seed)
        np.random.seed(a.seed)

    if not os.path.exists(f'{a.save_path}/{a.dataset}'):
        os.makedirs(f'{a.save_path}/{a.dataset}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset, class_labels = load_dataset(a.dataset, a.data_path, Augmentation(a.image_size))
    num_classes = len(class_labels)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=a.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=a.num_workers,
    )

    model = ViTIC(a.feature_dim, num_classes, a.projector_dim, a.image_size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)

    if a.model_path:
        checkpoint = torch.load(a.model_path)
        model.load_state_dict(checkpoint['weights'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    instance_loss = InstanceLoss(a.batch_size, a.instance_temp, device).to(device)
    cluster_loss = ClusterLoss(num_classes, a.cluster_temp, device).to(device)

    loss_hist = checkpoint['loss_hist'] if a.model_path else []
    for epoch in tqdm(range((checkpoint['epoch'] + 1) if a.model_path else 0, a.epochs)):
        loss_epoch = 0

        for step, ((x_i, x_j), _) in enumerate(data_loader):
            optimizer.zero_grad()
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            z_i, z_j, c_i, c_j = model(x_i, x_j)
            loss_instance = instance_loss(z_i, z_j)
            loss_cluster = cluster_loss(c_i, c_j)
            loss = loss_instance + loss_cluster
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

            if step % 50 == 0:
                print(f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")

        loss_hist.append(loss_epoch)

        if epoch % a.save_freq == 0:
            path = f'{a.save_path}/{a.dataset}/{epoch}_{loss_epoch}.tar'
            save_model(path, model, optimizer, epoch, loss_hist)
    
    path = f'{a.save_path}/{a.dataset}/{epoch}_{loss_epoch}_final.tar'
    save_model(path, model, optimizer, a.epochs, loss_hist)


if __name__ == "__main__":
    train()
