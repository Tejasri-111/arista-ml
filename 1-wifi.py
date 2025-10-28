import numpy as np
import matplotlib.pyplot as plt


def generate_gaussian_noise_grid(mean=0, variance=1):
    """Create a 2D grid (21x101) with Gaussian noise."""
    y = np.arange(0, 21, 1)
    x = np.arange(0, 101, 1)
    grid = np.zeros((len(y), len(x)))
    noise = np.random.normal(mean, np.sqrt(variance), grid.shape)
    return x, y, grid + noise


def add_wifi_clients(grid, x, y, n_clients=3, width=10, gap=5, amp_mean=5, amp_var=1):
    """
    Add N Wi-Fi-like client blobs on the noise grid with Gaussian-random amplitudes.

    Args:
        grid (np.ndarray): 2D noise grid.
        x, y (arrays): coordinate axes.
        n_clients (int): number of clients to place.
        width (float): width of each Wi-Fi block (Hz scale simulated).
        gap (int): gap between client blocks along x-axis.
        amp_mean (float): mean of Gaussian amplitude.
        amp_var (float): variance of Gaussian amplitude.

    Returns:
        np.ndarray: Modified grid with Wi-Fi client power added.
    """
    grid = grid.copy()
    nx = len(x)
    ny = len(y)
    total_width = n_clients * width + (n_clients - 1) * gap
    start_x = (nx - total_width) // 2  # center clients along x-axis

    for i in range(n_clients):
        # Random amplitude from Gaussian distribution
        amp = np.random.normal(amp_mean, np.sqrt(amp_var))

        # x region for this client
        x_start = int(start_x + i * (width + gap))
        x_end = int(min(x_start + width, nx))
        
        # Center frequency (y-center)
        fc = 10
        
        # y array for sinc² envelope
        yy = np.arange(ny)
        sinc_profile = np.sinc((yy - fc) / 5)**2  # width control = 5
        sinc_profile = sinc_profile / sinc_profile.max()  # normalize to [0,1]

        # Apply the profile
        for xi in range(x_start, x_end):
            grid[:, xi] += amp * sinc_profile

    # Normalize intensity values to [0, 255]
    grid = grid - grid.min()
    grid = (grid / grid.max()) * 255
    grid = np.clip(grid, 0, 255)

    return grid.astype(np.uint8)


def plot_noise_grid(x, y, grid, title):
    """Plot the 2D grid using matplotlib."""
    plt.figure(figsize=(8, 4))
    plt.imshow(grid, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=255)
    plt.colorbar(label='Power Intensity (0–255)')
    plt.title(title)
    plt.xlabel('X-axis (0–100)')
    plt.ylabel('Y-axis (0–20)')
    plt.show()


def main():
    # Noise parameters
    mean = float(input("Enter mean of Gaussian noise: "))
    variance = float(input("Enter variance of Gaussian noise: "))

    x, y, noise_grid = generate_gaussian_noise_grid(mean, variance)

    # Wi-Fi clients
    n_clients = int(input("Enter number of Wi-Fi clients: "))
    amp_mean = float(input("Enter mean amplitude for clients: "))
    amp_var = float(input("Enter variance of amplitude for clients: "))

    full_grid = add_wifi_clients(
        noise_grid, x, y,
        n_clients=n_clients,
        width=10, gap=5,
        amp_mean=amp_mean, amp_var=amp_var
    )

    plot_noise_grid(x, y, full_grid, f"Wi-Fi Clients ({n_clients}) + Gaussian Noise")


if __name__ == "__main__":
    main()
