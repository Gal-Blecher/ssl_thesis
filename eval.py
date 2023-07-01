import torch
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import datasets
import numpy as np


def plot_data_latent(model, data, show=True):
    z_points = torch.tensor([])
    z_labels = torch.tensor([])
    model.eval().cpu()
    for d in data:
        model(d[0])
        z_points = torch.cat((z_points, model.z))
        z_labels = torch.cat((z_labels, d[1]))
    z_points = z_points.detach().numpy()
    z_labels = z_labels.detach().numpy()

    if z_points.shape[1] > 2:
        z_points = TSNE(n_components=2, learning_rate='auto', init='random', random_state=11).fit_transform(z_points)

    points_df = pd.DataFrame({'x': z_points[:, 0],
                              'y': z_points[:, 1],
                              'label': z_labels})


    if (show == True):
        sns.scatterplot(data=points_df, x='x', y='y', hue='label', palette='Paired', edgecolor="black",
                        s=40, legend='full').set(title='1 Expert Latent Representation')
        plt.show()

def plot_input_reconstruction(x, x_hat):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # Transpose the tensor dimensions (C, H, W) to (H, W, C) for visualization
    x = np.transpose(x.detach().cpu().numpy(), (1, 2, 0))
    x_hat = np.transpose(x_hat.detach().cpu().numpy(), (1, 2, 0))

    # Plot input image
    axes[0].imshow(x)
    axes[0].set_title('Input')
    axes[0].axis('off')

    # Plot reconstruction image without specifying cmap
    axes[1].imshow(x_hat)
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model = torch.load('/Users/galblecher/Desktop/Thesis_out/ssl_cifar/unsupervised only/model.pkl', map_location=torch.device('cpu'))
    data = datasets.get_dataset()
    # plot_data_latent(model, data['test_loader'])
    batch = next(iter(data['test_loader']))
    model(batch[0])
    x_hat = model.x_hat
    plot_input_reconstruction(batch[0][0], x_hat[0])

