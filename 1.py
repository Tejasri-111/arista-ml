import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_noise_grid(mean=0, variance=1):
    """
    Create a 2D grid (21x101) with Gaussian noise.
    
    Args:
        mean (float): Mean of Gaussian noise.
        variance (float): Variance of Gaussian noise.
    
    Returns:
        tuple: (x, y, noisy_grid)
    """
    # Axes
    y = np.arange(0, 21, 1)   # 0 → 20 inclusive
    x = np.arange(0, 101, 1)  # 0 → 100 inclusive
    
    # Initialize grid
    grid = np.zeros((len(y), len(x)))
    
    # Gaussian noise
    noise = np.random.normal(mean, np.sqrt(variance), grid.shape)
    
    # Add noise to grid
    noisy_grid = grid + noise
    
    return x, y, noisy_grid


def plot_noise_grid(x, y, noise_grid, mean, variance):
    """
    Plot the 2D Gaussian noise grid using matplotlib.
    """
    plt.figure(figsize=(8, 4))
    plt.imshow(noise_grid, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Power (a.u.)')
    plt.title(f'2D Gaussian Noise (mean={mean}, var={variance})')
    plt.xlabel('X-axis (0–100)')
    plt.ylabel('Y-axis (0–20)')
    plt.show()


def main():
    # Input parameters
    mean = float(input("Enter mean of Gaussian noise: "))
    variance = float(input("Enter variance of Gaussian noise: "))
    
    # Generate noise grid
    x, y, noise_grid = generate_gaussian_noise_grid(mean, variance)
    
    # Plot result
    plot_noise_grid(x, y, noise_grid, mean, variance)


if __name__ == "__main__":
    main()
