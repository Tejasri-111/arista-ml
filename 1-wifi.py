import numpy as np
import matplotlib.pyplot as plt


def generate_gaussian_noise_grid(x,y,mean=0, variance=1):
    grid = np.zeros((len(y), len(x)))
    noise = np.random.normal(mean, np.sqrt(variance), grid.shape)
    return grid + noise


def add_wifi_clients(grid, x, y, n_clients=3, width=10, gap=5, amp_mean=5, amp_var=1):

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

        y_start = max(int(fc - 3), 0)     # roughly a few widths around center
        y_end   = min(int(fc + 3), ny)

        # Apply the profile
        for xi in range(x_start, x_end):
            grid[y_start:y_end, xi] += amp * sinc_profile[y_start:y_end]

    return grid

def add_zigbee_clients(grid, x, y, n_clients=5, width=3, amp_mean=2, amp_var=0.5,sigma_y = 1.5,bursty= True, duty_cycle= 0.2):
    """
    Add Zigbee-like narrowband peaks on the noise grid.
    """
    grid = grid.copy().astype(float)
    nx = len(x)
    ny = len(y)
    
    for _ in range(n_clients):
        # Random amplitude
        amp = np.random.normal(amp_mean, np.sqrt(amp_var))

        # Random x (horizontal center position)
        x_center = np.random.randint(0, nx)
        x_start = max(x_center - width // 2, 0)
        x_end = min(x_center + width // 2, nx)
        
        # Random y (center frequency)
        fc = np.random.randint(5, ny - 5)

        # y array for Gaussian profile
        yy = np.arange(ny)
        gauss_profile = np.exp(-0.5 * ((yy - fc) / sigma_y)**2)
        gauss_profile = gauss_profile / gauss_profile.max()  # normalize

        # Limit the vertical extent (like Wi-Fi)
        y_start = max(int(fc - 3), 0)
        y_end = min(int(fc + 3), ny)

        for xi in range(x_start, x_end):
            if bursty and (np.random.rand() > duty_cycle):
                # skip inactive columns
                continue
            grid[y_start:y_end, xi] += amp * gauss_profile[y_start:y_end]
    
    return grid



def add_cordless_phone_clients(grid, x, y, n_clients=2, width=6, gap=8,
                               amp_mean=3.0, amp_var=0.5, sigma_y=2.5,
                               bursty=False, duty_cycle=0.2):
    
    #Add cordless-phone-like medium-band signals to the grid and normalize result.
    grid = grid.copy().astype(float)
    nx = len(x)
    ny = len(y)

    total_width = n_clients * width + (n_clients - 1) * gap
    

    for i in range(n_clients):
        amp = np.random.normal(amp_mean, np.sqrt(amp_var))

        # Random x (horizontal center position)
        x_center = np.random.randint(0, nx)
        x_start = max(x_center - width // 2, 0)
        x_end = min(x_center + width // 2, nx)

        # random vertical center (fc)
        fc = np.random.randint(3, ny - 3)

        # Gaussian vertical profile
        yy = np.arange(ny)
        gauss_profile = np.exp(-0.5 * ((yy - fc) / sigma_y) ** 2)
        gauss_profile /= gauss_profile.max()

        # Limit vertical region
        y_start = max(int(fc - 3 ), 0)
        y_end = min(int(fc + 3 ) + 1, ny)

        # Apply signal per column
        for xi in range(x_start, x_end):
            if bursty and (np.random.rand() > duty_cycle):
                continue
            grid[y_start:y_end, xi] += amp * gauss_profile[y_start:y_end]

    # 🔹 Normalize grid to [0, 255]
    grid -= grid.min()
    if grid.max() > 0:   # avoid divide-by-zero
        grid = (grid / grid.max()) * 255
    grid = np.clip(grid, 0, 255)

    return grid


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


def add_bluetooth_clients(grid, x, y, n_clients=20, width=1, height_sigma=0.8,
                          amp_mean=60, amp_var=10, duty_cycle=0.3):
    """
    Add Bluetooth-like narrowband hopping bursts to the grid.

    Each 'client' is a short-duration, narrow-frequency burst (FHSS behavior).
    """
    grid = grid.copy().astype(float)
    nx = len(x)
    ny = len(y)

    for _ in range(n_clients):
        # Random amplitude
        amp = np.random.normal(amp_mean, np.sqrt(amp_var))

        # Random position (time and frequency)
        x_center = np.random.randint(0, nx)
        y_center = np.random.randint(0, ny)

        # Horizontal range (short duration)
        x_start = max(x_center - width // 2, 0)
        x_end = min(x_center + width // 2 + 1, nx)

        # Gaussian frequency envelope (narrow)
        yy = np.arange(ny)
        gauss_profile = np.exp(-0.5 * ((yy - y_center) / height_sigma)**2)
        gauss_profile /= gauss_profile.max()

        # Only active with some probability (duty cycle)
        if np.random.rand() > duty_cycle:
            continue

        for xi in range(x_start, x_end):
            grid[:, xi] += amp * gauss_profile

    return grid


def main():
    #step 1: only noise
    """Create a 2D grid (21x101) with Gaussian noise."""
    y = np.arange(0, 21, 1)
    x = np.arange(0, 101, 1)
    # Noise parameters
    mean = 10 #float(input("Enter mean of Gaussian noise: "))
    variance = 2 #float(input("Enter variance of Gaussian noise: "))

    noise_grid = generate_gaussian_noise_grid(x,y,mean, variance).astype(float)

    #step 2: wifi is added + (noise exists)
    # Wi-Fi clients
    n_wifi_clients = int(input("Enter number of Wi-Fi clients: "))
    amp_mean =180 #float(input("Enter mean amplitude for clients: "))
    amp_var =25  #float(input("Enter variance of amplitude for clients: "))

    only_wifi_grid = add_wifi_clients(
        noise_grid, x, y,
        n_clients=n_wifi_clients,
        width=10, gap=5,
        amp_mean=amp_mean, amp_var=amp_var
    )

    #plot_noise_grid(x, y, only_wifi_grid, f"Wi-Fi + Gaussian Noise")

    #step 3:  adding zigbee clients
    n_zigbee = int(input("Enter number of Zigbee clients: "))
    zigbee_amp_mean =80 #float(input("Enter mean amplitude for Zigbee: "))
    zigbee_amp_var =15 #float(input("Enter variance for Zigbee: "))

    zigbee_grid = add_zigbee_clients(
        only_wifi_grid, x, y, n_clients=n_zigbee,
        width=3, amp_mean=zigbee_amp_mean, amp_var=zigbee_amp_var,bursty = False, duty_cycle= 0.2
    )

    #plot_noise_grid(x, y, zigbee_grid, f"Wi-Fi + Zigbee + Noise")

    #step 4: adding cordless phone clients
    n_cordless_phone = int(input("Enter number of Cordless Phone clients: "))
    amp_mean_phone = 120 #float(input("Enter mean amplitude for Cordless Phones: "))
    amp_var_phone = 20#float(input("Enter variance of amplitude for Cordless Phones: "))

    cordless_grid = add_cordless_phone_clients(
        zigbee_grid, x, y,
        n_clients=n_cordless_phone,
        width=6, gap=8,
        amp_mean=amp_mean_phone, amp_var=amp_var_phone,
        sigma_y=2.5,
        bursty=False, duty_cycle=0.8
    )

   # Step 5: Add Bluetooth clients
    n_bt = int(input("Enter number of Bluetooth clients: "))
    bt_amp_mean = 60
    bt_amp_var = 10

    bluetooth_grid = add_bluetooth_clients(
        cordless_grid, x, y,
        n_clients=n_bt,
        width=1,
        height_sigma=0.8,
        amp_mean=bt_amp_mean,
        amp_var=bt_amp_var,
        duty_cycle=0.3
    )

    grid = bluetooth_grid
    # Normalize to [0, 255]
    grid = grid - grid.min()
    grid = (grid / grid.max()) * 255
    grid = np.clip(grid, 0, 255)
    plot_noise_grid(x, y, grid,
                    f"Bluetooth + Cordless Phones + Zigbee + Wi-Fi + Noise")


if __name__ == "__main__":
    main()
