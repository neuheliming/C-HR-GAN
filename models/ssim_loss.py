
import torch
import torch.nn.functional as F

def ssim_loss(img1, img2, window_size=11, size_average=True):
    def gaussian(window_size, sigma):
        sigma_tensor = torch.tensor(sigma)
        gauss = torch.Tensor([torch.exp(-((x - window_size // 2)**2) / (2 * sigma_tensor**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    (_, channel, height, width) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2/2
    epsilon = 1e-5
    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = (((2 * mu1_mu2 + C1) ** 5) * (sigma12 + C3)) * \
           ((2 * torch.sqrt(torch.clamp(sigma1_sq, min=epsilon)) * torch.sqrt(torch.clamp(sigma2_sq, min=epsilon)) + C2) ** 3) / \
           (((mu1_sq + mu2_sq + C1) ** 5) * ((sigma1_sq + sigma2_sq + C2) ** 3) * \
           (torch.sqrt(torch.clamp(sigma1_sq, min=epsilon)) * torch.sqrt(torch.clamp(sigma2_sq, min=epsilon)) + C3))
    # ssim_map = (((2 * mu1_mu2 + C1) ** 5) * (sigma12 + C3)) * ((2 * torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C2) ** 3) / (
    #         ((mu1_sq + mu2_sq + C1) ** 5) * ((sigma1_sq + sigma2_sq + C2) ** 3) * ((torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C3)))

    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean(1).mean(1).mean(1)
