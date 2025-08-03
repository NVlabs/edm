# your_sampler.py

import torch
import numpy as np
import tqdm
import pickle
import PIL.Image
import dnnlib
import os
import datetime

@torch.no_grad()
def eta_constant(t):
    return torch.zeros_like(t)

@torch.no_grad()
def eta_discrete(u, alpha_s, alpha_t, sigma_s, sigma_t):
    gamma_s = (alpha_s**2 / sigma_s**2) * (sigma_t**2 / alpha_t**2)

    numerator = gamma_s - 1
    denominator = torch.sqrt(gamma_s * u**2) - torch.sqrt(u**2 + 1 - gamma_s)
    return numerator / denominator

@torch.no_grad()
def eta_continous(net, s):
    lambda_s_prime = net._d_lambda(s)
    eta_s = net.u_constant - torch.sqrt(net.u_constant**2 + lambda_s_prime)
    return eta_s


@torch.no_grad()
def sample_images_from_model(
    network_pkl,
    dest_path,
    seed=0,
    gridw=8,
    gridh=8,
    num_steps=1000,
    device=torch.device('cuda')
):
    t_min = torch.tensor(5e-3)
    t_max = torch.tensor(1 - 5e-3)

    class_index = 4
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    # # jeżeli chcesz same jelenie to zostaw te linijki, jeżeli wszytsko to zakomentuj
    # class_labels[:, :] = 0
    # class_labels[:, class_index] = 1

    z_t = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    ts = torch.linspace(t_max, t_min, steps=num_steps + 1, device=device)

    for i in tqdm.trange(num_steps, desc="Sampling"):
        t = ts[i].unsqueeze(0)
        s = ts[i + 1].unsqueeze(0)

        alpha_t = net.alpha(t)
        sigma_t = net.sigma(t)
        alpha_s = net.alpha(s)
        sigma_s = net.sigma(s)

        u_s = net.u(s)
        u_t = net.u(t)

        # 1. eta = 1
        # 2. eta z pliku not_main.pdf
        # 3. eta z pliku Pokarowski_Heidelberg2025.pdf
        # eta_s = eta_constant(t)
        eta_s = eta_discrete(u_s, alpha_t, alpha_s, sigma_s, sigma_t)
        # eta_s = torch.tensor(0).to(device)
        # eta_s = eta_continous(net, s)

        eps_scaled = net(z_t, t, class_labels)
        eps_pred = eps_scaled / u_t.reshape(-1, 1, 1, 1)

        coeff_eps = sigma_s * torch.sqrt(1 - eta_s**2) - alpha_s * (sigma_t / alpha_t)
        coeff_z = alpha_s / alpha_t

        eps = torch.randn_like(z_t) if i < num_steps - 1 else 0

        z_t = coeff_eps.reshape(-1,1,1,1) * eps_pred + coeff_z.reshape(-1,1,1,1) * z_t + sigma_s.reshape(-1,1,1,1) * eta_s.reshape(-1,1,1,1) * eps
    
    save_image_grid(z_t, dest_path, gridw, gridh)

def save_image_grid(images, path, gridw, gridh):
    images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    C, H, W = images.shape[1:]
    images = images.view(gridh, gridw, C, H, W).permute(0, 3, 1, 4, 2).reshape(gridh * H, gridw * W, C)
    image = PIL.Image.fromarray(images.cpu().numpy(), 'RGB')
    image.save(path)


def main():
    network_path = 'model/ueps.pkl'  
    output_path = f'output/generated_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    sample_images_from_model(
        network_pkl=network_path,
        dest_path=output_path,
        seed=42,
        gridw=8,
        gridh=8,
        num_steps=18,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

if __name__ == "__main__":
    main()
