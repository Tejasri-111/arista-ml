import numpy as np

class Pulse:
    def __init__(self, t, l, r, vals):
        self.t_start = t
        self.t_end = t
        self.l = l
        self.r = r
        self.kc = (l + r) / 2
        self.bw = r - l + 1
        self.pvals = [vals]
        self.peak = np.max(vals)
    def extend(self, t, l, r, vals):
        self.t_end = t
        self.pvals.append(vals)
        self.peak = max(self.peak, np.max(vals))

def detect_pulses_by_edges(
    grid, 
    rise_thresh= 10, 
    fall_thresh=0.05, 
    min_intensity=0.2,
    merge_gap=1
):
    """
    Detect pulses by identifying rising/falling edges in frequency intensity.
    Adds absolute intensity thresholding.
    """
    arr = np.asarray(grid, float)
    T, N = arr.shape
    active, completed = [], []

    for t in range(T):
        row = arr[t]
        diff = np.diff(row)

        # Identify edge indices
        rising_edges = np.where(diff > rise_thresh)[0]
        falling_edges = np.where(diff < -fall_thresh)[0]

        # Pair up rising and falling edges
        pulses = []
        i = 0
        while i < len(rising_edges):
            l = rising_edges[i]
            r_candidates = falling_edges[falling_edges > l]
            if len(r_candidates) == 0:
                break
            r = r_candidates[0]

            # ✅ NEW: absolute intensity check
            if np.max(row[l:r+1]) >= min_intensity and r - l > 1:
                pulses.append((l, r))
            i += 1

        # Merge nearby regions
        merged = []
        for l, r in pulses:
            if not merged or l - merged[-1][1] > merge_gap:
                merged.append([l, r])
            else:
                merged[-1][1] = r
        pulses = [(int(l), int(r)) for l, r in merged]

        # Create Pulse objects
        new_pulses = []
        for l, r in pulses:
            vals = row[l:r+1]
            new_pulses.append((l, r, vals))

        # --- Match to active pulses (same as before) ---
        matched_active = set()
        for (l, r, vals) in new_pulses:
            matched = False
            if active:
                kc_new = (l + r) / 2
                bw_new = r - l + 1
                kc_act = np.array([a.kc for a in active])
                bw_act = np.array([a.bw for a in active])

                overlap = np.maximum(0, np.minimum([a.r for a in active], r) - np.maximum([a.l for a in active], l))
                union = np.maximum([a.r for a in active], r) - np.minimum([a.l for a in active], l)
                FOR = overlap / (union + 1e-12)

                cont_match = (FOR > 0.6) & (np.round(kc_act) == np.round(kc_new))
                if np.any(cont_match):
                    idx = np.where(cont_match)[0][0]
                    active[idx].extend(t, l, r, vals)
                    matched = True
                    matched_active.add(idx)
            if not matched:
                active.append(Pulse(t, l, r, vals))

        # Finalize unmatched actives
        new_active = [act for i, act in enumerate(active) if i in matched_active]
        for i, act in enumerate(active):
            if i not in matched_active:
                completed.append(act)
        active = new_active

    completed.extend(active)
    return completed


if __name__ == "__main__":
    grid = grid_loaded = np.loadtxt("../results/interference_grid.txt")
    print(grid.shape)
    completed = detect_pulses_by_edges(grid)