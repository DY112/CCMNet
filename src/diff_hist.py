import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

class DifferentiableHistogram(nn.Module):
    def __init__(self, nbins=64, hist_boundary=(-2.85, 2.85)):
        super(DifferentiableHistogram, self).__init__()
        self.default_nbins = nbins
        self.default_hist_boundary = hist_boundary

    def rgb_to_uv(self, rgb):
        """ Converts RGB to log-chroma space. """
        log_rgb = torch.log(rgb + EPS)
        u = log_rgb[:, 1] - log_rgb[:, 0]
        v = log_rgb[:, 1] - log_rgb[:, 2]
        return torch.stack([u, v], dim=-1)

    def get_hist_colors(self, img):
        """ Gets valid chroma and color values for histogram computation. """
        img_r = img.view(-1, 3)
        img_chroma = self.rgb_to_uv(img_r)
        valid_pixels = torch.sum(img_r, dim=1) > EPS  # exclude any zero pixels
        valid_chroma = img_chroma[valid_pixels, :]
        valid_colors = img_r[valid_pixels, :]
        return valid_chroma, valid_colors

    def forward(self, img, nbins=None, hist_boundary=None):
        # Use default values if not provided
        if nbins is None:
            nbins = self.default_nbins
        if hist_boundary is None:
            hist_boundary = self.default_hist_boundary

        # Calculate eps and bin centers
        eps = (hist_boundary[1] - hist_boundary[0]) / (nbins - 1)
        A_u = torch.arange(hist_boundary[0], hist_boundary[1] + eps / 2, eps, device=img.device)
        A_v = torch.flip(A_u, dims=[0])

        batch_size, channels, height, width = img.shape
        img = img.permute(0, 2, 3, 1).contiguous()  # Change to (batch_size, height, width, channels)
        img = img.view(batch_size, -1, channels)  # Flatten the spatial dimensions

        # Initialize histogram
        histogram = torch.zeros((batch_size, nbins, nbins), device=img.device)

        for b in range(batch_size):
            # Get valid chroma and color values
            chroma_input, rgb_input = self.get_hist_colors(img[b])

            # Calculate Iy
            Iy = torch.sqrt(torch.sum(rgb_input ** 2, dim=1))

            # Calculate differences in log_U space
            diff_u = torch.abs(chroma_input[:, 0].unsqueeze(1) - A_u.unsqueeze(0))

            # Calculate differences in log_V space
            diff_v = torch.abs(chroma_input[:, 1].unsqueeze(1) - A_v.unsqueeze(0))

            # Apply threshold
            diff_u[diff_u > eps] = 0
            diff_u[diff_u != 0] = 1

            diff_v[diff_v > eps] = 0
            diff_v[diff_v != 0] = 1

            # Calculate histogram
            Iy_diff_v = Iy.unsqueeze(1) * diff_v
            N = torch.matmul(Iy_diff_v.transpose(0, 1), diff_u)
            
            # add small epsilon to avoid zero values from N
            N = N + 1e-8

            # Normalize histogram
            norm_factor = torch.sum(N) + 1e-8
            N = torch.sqrt(N / norm_factor)

            histogram[b] = N
            
        return histogram

# Example usage
if __name__ == "__main__":
    # Create a random image tensor with shape (batch_size, channels, height, width)
    img = torch.rand((2, 3, 256, 256))  # Example with batch size of 2
    
    # Create the histogram layer
    hist_layer = DifferentiableHistogram()
    
    # Compute the histogram with custom nbins and hist_boundary
    histogram = hist_layer(img, nbins=64, hist_boundary=(-2.85, 2.85))
    
    print(histogram.shape)  # Should print torch.Size([2, 64, 64])