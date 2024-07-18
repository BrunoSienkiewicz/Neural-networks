import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from scipy import linalg


class Evaluator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes=10):
        super(Evaluator, self).__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, 50)
        self.fc_out  = nn.Linear(50, n_classes)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def get_features(self, x):
        x = torch.flatten(x, 1)
        x = self.LeakyReLU(self.fc_1(x))
        x = self.LeakyReLU(self.fc_2(x))
        return x
    

    def forward(self, x):
        x = self.get_features(x)
        x = self.fc_out(x)
        return x

    def evaluate(self, model, orig_data, device, latent_dim=32, n_gen=100):
        with torch.no_grad():
            fixed_noise = torch.randn(n_gen, latent_dim, device=device)
            generations = model(fixed_noise)
            dist_orig_data = self.get_features(orig_data[:n_gen].to(device)).cpu()
            dist_gen = self.get_features(generations.to(device)).cpu()
            return calculate_frechet_distance(dist_orig_data.numpy(), dist_gen.numpy())

def calculate_frechet_distance(distribution_1, distribution_2, eps=1e-6):
    mu1 = np.mean(distribution_1, axis=0)
    sigma1 = np.cov(distribution_1, rowvar=False)

    mu2 = np.mean(distribution_2, axis=0)
    sigma2 = np.cov(distribution_2, rowvar=False)

    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def visualize_reconstructions(model, input_imgs, device):
    torch.stack([input_imgs[i][0] for i in range(len(input_imgs))], dim=0)
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs, means, log_var = model(input_imgs.to(device))
    reconst_imgs = reconst_imgs.cpu()
    
    # Plotting
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=False)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(10,10))
    plt.title(f"Reconstructions")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

def generate_images(model, n_imgs, device, latent_dim=32):
    # Generate images
    model.eval()
    with torch.no_grad():
        generated_imgs = model(torch.randn([n_imgs, latent_dim]).to(device))
    generated_imgs = generated_imgs.cpu()
    
    grid = torchvision.utils.make_grid(generated_imgs, nrow=4, normalize=False)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(10,10))
    plt.title(f"Generations")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

def generate_images_unet(net, n_gen=8, n_steps=50, device='cuda'):
    pred_hist = []
    step_hist = []
    with torch.no_grad():
        fixed_noise = torch.randn(n_gen, 3, 32, 32).to(device)
        x = fixed_noise
        for i in range(n_steps):
            step_hist.append(x.detach().cpu()) 
            pred = net(x)
            pred_hist.append(pred.detach().cpu())
            mix_factor = 1 / (n_steps - i)  
            x = x * (1 - mix_factor) + pred * mix_factor  
        return pred, pred_hist, step_hist

def evaluate_unet(eval, model, orig_data, device, n_gen=100, n_steps=50):
    with torch.no_grad():
        fixed_noise = torch.randn(n_gen, 3, 32, 32).to(device)
        x = model(fixed_noise)
        for i in range(n_steps):
            pred = model(x)
            mix_factor = 1 / (n_steps - i) 
            x = x * (1 - mix_factor) + pred * mix_factor  
        dist_orig_data = eval.get_features(orig_data.to(device)).cpu()
        dist_gen = eval.get_features(pred.to(device)).cpu()
        return calculate_frechet_distance(dist_orig_data.numpy(), dist_gen.numpy())
    