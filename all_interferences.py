import numpy as np
import matplotlib.pyplot as plt


def generate_gaussian_noise_grid(x, y, mean=0, variance=1):
    grid = np.zeros((len(y), len(x)))
    noise = np.random.normal(mean, np.sqrt(variance), grid.shape)
    return grid + noise


# ==================== WIFI ====================
def add_wifi_clients(noise_grid, grid, x, y, n_clients=3, width=10, x_gap=10,
                     amp_mean=180, amp_var=25, positions_log=None):
    """
    Add multiple Wi-Fi-like signals spaced across the y-axis (frequency),
    each occupying a distinct vertical band with controlled gaps.
    """
    grid = grid.copy().astype(float)
    nx, ny = len(x), len(y)

    # --- Compute vertical spacing parameters ---
    gap = max(1, int(0.05 * ny))  # 5% of height as gap
    usable_height = ny - (n_clients + 1) * gap
    if usable_height <= 0:
        raise ValueError("Too many Wi-Fi clients for given Y range")

    bw_per_client = max(3, usable_height // n_clients)

    # --- Generate clients ---
    for i in range(n_clients):
        y_start = gap + i * (bw_per_client + gap)
        y_end = min(y_start + bw_per_client, ny)
        fc = int((y_start + y_end) / 2)  # center frequency index

        yy = np.arange(ny)
        sinc_profile = np.sinc((yy - fc) / 5)**2
        sinc_profile /= sinc_profile.max()

        # --- Horizontal duplication across x-axis ---
        x_start = 1
        while x_start + width < nx:
            x_end = int(min(x_start + width, nx))
            amp = np.random.normal(amp_mean, np.sqrt(amp_var))
            for xi in range(x_start, x_end):
                grid[y_start:y_end, xi] += amp * sinc_profile[y_start:y_end]
            x_start += width + x_gap

        # --- Log the generated client ---
        if positions_log is not None:
            positions_log.append({
                "type": "Wi-Fi",
                "center_x": int(nx // 2),
                "center_y": int(fc),
                "bandwidth_px": y_end - y_start,
                "horizontal_width" : width,
                "amplitude": float(amp),
                "duty_cycle": 1.0,
                "noise_floor": float(np.mean(noise_grid)),
            })

    return grid


# ==================== ZIGBEE ====================
def add_zigbee_clients(noise_grid,grid, x, y, n_clients=5, width=3, amp_mean=2, amp_var=0.5,
                       sigma_y=1.5, bursty=False, duty_cycle=0.2, positions_log=None):
    grid = grid.copy().astype(float)
    nx, ny = len(x), len(y)

    for _ in range(n_clients):
        amp = np.random.normal(amp_mean, np.sqrt(amp_var))
        x_center = np.random.randint(0, nx)
        x_start = max(x_center - width // 2, 0)
        x_end = min(x_center + width // 2, nx)
        fc = np.random.randint(5, ny - 5)

        yy = np.arange(ny)
        gauss_profile = np.exp(-0.5 * ((yy - fc) / sigma_y)**2)
        gauss_profile /= gauss_profile.max()
        y_start, y_end = max(int(fc - 1), 0), min(int(fc + 1), ny)

        for xi in range(x_start, x_end):
            if bursty and (np.random.rand() > duty_cycle):
                continue
            grid[y_start:y_end, xi] += amp * gauss_profile[y_start:y_end]

        if positions_log is not None:
            positions_log.append({
                "type": "Zigbee",
                "center_x": int(x_center),
                "center_y": int(fc),
                "bandwidth_px": y_end- y_start,
                "horizontal_width": width,
                "amplitude": float(amp),
                "duty_cycle":duty_cycle if bursty else 1,
                "noise_floor": np.mean(noise_grid),
            })

    return grid


# ==================== CORDLESS PHONE ====================
def add_cordless_phone_clients(noise_grid,grid, x, y, n_clients=2, width=6, gap=8,
                               amp_mean=3.0, amp_var=0.5, sigma_y=2.5,
                               bursty=False, duty_cycle=0.2, positions_log=None):

    grid = grid.copy().astype(float)
    nx, ny = len(x), len(y)

    for _ in range(n_clients):
        amp = np.random.normal(amp_mean, np.sqrt(amp_var))
        x_center = np.random.randint(0, nx)
        x_start = max(x_center - width // 2, 0)
        x_end = min(x_center + width // 2, nx)
        fc = np.random.randint(3, ny - 3)

        yy = np.arange(ny)
        gauss_profile = np.exp(-0.5 * ((yy - fc) / sigma_y)**2)
        gauss_profile /= gauss_profile.max()
        y_start, y_end = max(int(fc - 2), 0), min(int(fc + 2) + 1, ny)

        for xi in range(x_start, x_end):
            if bursty and (np.random.rand() > duty_cycle):
                continue
            grid[y_start:y_end, xi] += amp * gauss_profile[y_start:y_end]

        if positions_log is not None:
            positions_log.append({
                "type": "Cordless-phone",
                "center_x": int(x_center),
                "center_y": int(fc),
                "bandwidth_px": y_end- y_start,
                "horizontal_width": width,
                "amplitude": float(amp),
                "duty_cycle": duty_cycle if bursty else 1,
                "noise_floor": np.mean(noise_grid),
            })

    return grid


# ==================== BLUETOOTH ====================
def add_bluetooth_clients(noise_grid,grid, x, y, n_clients=20, width=1, height_sigma=0.8,
                          amp_mean=60, amp_var=10, duty_cycle=0.3, positions_log=None):
    grid = grid.copy().astype(float)
    nx, ny = len(x), len(y)

    for _ in range(n_clients):
        amp = np.random.normal(amp_mean, np.sqrt(amp_var))
        x_center = np.random.randint(0, nx)
        y_center = np.random.randint(0, ny)
        x_start = max(x_center - width // 2, 0)
        x_end = min(x_center + width // 2 + 1, nx)
        y_start, y_end = max(int(fc - 1), 0), min(int(fc + 1) + 1, ny)

        yy = np.arange(ny)
        gauss_profile = np.exp(-0.5 * ((yy - y_center) / height_sigma)**2)
        gauss_profile /= gauss_profile.max()

        if np.random.rand() > duty_cycle:
            continue

        for xi in range(x_start, x_end):
            grid[y_start:y_end, xi] += amp * gauss_profile

        if positions_log is not None:
            positions_log.append({
                "type": "Cordless-phone",
                "center_x": int(x_center),
                "center_y": int(y_center),
                "bandwidth_px": y_end - y_start,
                "horizontal_width": width,
                "amplitude": float(amp),
                "duty_cycle": duty_cycle ,
                "noise_floor": np.mean(noise_grid),
            })
    return grid


# ==================== PLOTTER ====================
def plot_noise_grid(x, y, grid, title):
    plt.figure(figsize=(8, 4))
    plt.imshow(grid, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=255)
    plt.colorbar(label='Power Intensity (0–255)')
    plt.title(title)
    plt.xlabel('X-axis (0–100)')
    plt.ylabel('Y-axis (0–20)')
    plt.show()


# ==================== MAIN ====================
def main():
    y = np.arange(0, 21, 1)
    x = np.arange(0, 101, 1)

    mean, variance = 10, 2
    noise_grid = generate_gaussian_noise_grid(x, y, mean, variance).astype(float)

    # Store (type, x, y) for all clients
    positions_log = []

    # --- Wi-Fi ---
    n_wifi_clients = int(input("Enter number of Wi-Fi clients: "))
    wifi_grid = add_wifi_clients(noise_grid,noise_grid, x, y,
                                 n_clients=n_wifi_clients, width=10, x_gap=10,
                                 amp_mean=180, amp_var=25,
                                 positions_log=positions_log)

    # --- Zigbee ---
    n_zigbee = int(input("Enter number of Zigbee clients: "))
    zigbee_grid = add_zigbee_clients(noise_grid,wifi_grid, x, y, n_clients=n_zigbee,
                                     width=3, amp_mean=80, amp_var=15,
                                     bursty=False, duty_cycle=0,
                                     positions_log=positions_log)

    # --- Cordless Phones ---
    n_cordless = int(input("Enter number of Cordless Phone clients: "))
    cordless_grid = add_cordless_phone_clients(noise_grid,zigbee_grid, x, y,
                                               n_clients=n_cordless,
                                               width=6, gap=8,
                                               amp_mean=120, amp_var=20,
                                               sigma_y=2.5,
                                               bursty=False, duty_cycle=0.8,
                                               positions_log=positions_log)

    # --- Bluetooth ---
    n_bt = int(input("Enter number of Bluetooth clients: "))
    bluetooth_grid = add_bluetooth_clients(noise_grid,cordless_grid, x, y,
                                           n_clients=n_bt, width=1,
                                           height_sigma=0.8,
                                           amp_mean=70, amp_var=5,
                                           duty_cycle=1,
                                           positions_log=positions_log)

    # --- Final normalization and plot ---
    grid = bluetooth_grid
    grid -= grid.min()
    grid = (grid / grid.max()) * 255
    grid = np.clip(grid, 0, 255)


    plot_noise_grid(x, y, grid,
                    f"Bluetooth + Cordless + Zigbee + Wi-Fi + Noise")

  


if __name__ == "__main__":
    main()
