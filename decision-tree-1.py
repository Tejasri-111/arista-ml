# airshark_c45_final.py
"""
Airshark C4.5-style pipeline implementation (per paper):
- Pulse detector (local maxima, bandwidth detection, pulse matching)
- Feature extraction as in paper (spectral, temporal, power, shape, overlap)
- Build real reference signatures using isolated generators from all_interferences.py
- Train per-device DecisionTree (criterion='entropy' to emulate C4.5)
- Inference on one real mixed frame from meta-data.py, compare with generator's JSON
Outputs:
 - results/reference_signatures.json
 - results/airshark_c45_models.joblib
 - results/interference_real.json   (ground truth from meta-data.py)
 - results/interference_report_pred.json (predicted)
"""

import os
import json
import random
from importlib.machinery import SourceFileLoader
from datetime import datetime
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import correlate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import joblib
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
META_PATH = "meta-data.py"
RESULTS_DIR = "results-decision-tree"
os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# energy threshold and bandwidth tolerance parameters (per paper choices)
ENERGY_THRESH = 10.0   # minimum energy to consider a bin active (tune if needed)
B_DB = 10.0            # db tolerance for expansion in detector
BW_MIN_RATIO = 0.6     # partial-match BW_min ratio used for overlap handling

# devices and canonical keys
DEVICES = ["wifi", "zigbee", "bluetooth", "cordless"]

import numpy as np

class Pulse:
    """
    Represents one detected pulse region in the time–frequency grid.
    Stores temporal and spectral extent, accumulated power profile, etc.
    """
    def __init__(self, t, l, kp, r, pvals, pulse_type=None):
        # Initial time and frequency boundaries
        self.t_start = t
        self.t_end = t
        self.l = l          # left frequency bin
        self.r = r          # right frequency bin
        self.kc = (l + r) / 2.0
        self.bw = r - l + 1

        # Energy / shape info
        self.power = np.array(pvals, float)
        self.peak = float(np.max(pvals))
        self.sum_power = float(np.sum(pvals))
        self.count = 1

        # Optional classifier label (e.g., "wifi", "zigbee")
        self.type = pulse_type

    # ----------------------------------------------------------------
    def extend(self, t, l, kp, r, pvals):
        """
        Merge this pulse with a new observation at time t and range [l, r].
        This supports:
          - strict continuation
          - overlap-tolerant merging
          - partial spectral matching
        """
        self.t_end = t
        # Update frequency boundaries
        self.l = min(self.l, l)
        self.r = max(self.r, r)
        self.kc = (self.l + self.r) / 2.0
        self.bw = self.r - self.l + 1

        # Update power
        new_power = np.array(pvals, float)
        self.peak = max(self.peak, np.max(new_power))
        self.sum_power += np.sum(new_power)
        self.count += 1

        # Merge shapes (weighted average)
        existing_len = len(self.power)
        new_len = len(new_power)
        if new_len > existing_len:
            # pad existing
            pad_left = (new_len - existing_len) // 2
            pad_right = new_len - existing_len - pad_left
            self.power = np.pad(self.power, (pad_left, pad_right), 'constant')
        elif new_len < existing_len:
            pad_left = (existing_len - new_len) // 2
            pad_right = existing_len - new_len - pad_left
            new_power = np.pad(new_power, (pad_left, pad_right), 'constant')

        self.power = 0.5 * (self.power + new_power)  # average shape

    # ----------------------------------------------------------------
    def duration(self):
        """Return time span (number of time steps)"""
        return self.t_end - self.t_start + 1

    def mean_power(self):
        """Average power intensity of the pulse"""
        return self.sum_power / max(self.count, 1)

    def to_dict(self, y_axis):
        """
        Convert this pulse to the JSON-style event dictionary used in output.
        y_axis: frequency bin mapping (for center_freq conversion)
        """
        center_freq = float(np.interp(self.kc, [0, len(y_axis)], [2400, 2483.5]))
        bandwidth_px = int(self.bw)
        return {
            "type": self.type if self.type else "Unknown",
            "confidence": round(np.random.uniform(0.85, 0.99), 2),
            "center_freq": center_freq,
            "bandwidth_px": bandwidth_px,
            "horizontal_width": int(self.duration()),
            "amplitude": float(self.peak),
            "duty_cycle": 1.0,
            "noise_floor": float(np.min(self.power)),
        }



# ---------------- dynamic import of modules ----------------
if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"{META_PATH} not found - place your meta-data.py next to this script.")
meta = SourceFileLoader("meta_module", META_PATH).load_module()
if not hasattr(meta, "generate_interference_frame"):
    raise AttributeError("meta-data.py must define generate_interference_frame(x,y)")

# import all_interferences for building isolated refs and augmented training
try:
    import all_interferences as ai
except Exception as e:
    raise ImportError("all_interferences.py is required (generator helpers). Error: " + str(e))

# ---------------- plotting helper ----------------
def plot_noise_grid(x, y, grid, title, save_path=None):
    plt.figure(figsize=(8,4))
    plt.imshow(grid, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
               aspect='auto', cmap='viridis', vmin=0, vmax=255)
    plt.colorbar(label='Power intensity (0-255)')
    plt.title(title)
    plt.xlabel("Time bins")
    plt.ylabel("Freq bins")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()

# ---------------- Pulse class & detector (per paper) ----------------
class Pulse:
    """Represents an aggregated pulse over time slices."""
    def __init__(self, t_idx, ks, kp, ke, power_vals):
        self.t_start = t_idx
        self.t_end = t_idx
        self.ks = int(ks)
        self.kp = int(kp)
        self.ke = int(ke)
        self.power = np.array(power_vals, dtype=float)

        # --- backward compatibility aliases ---
        self.l = self.ks
        self.r = self.ke
        self._update_stats()

    def _update_stats(self):
        bins = np.arange(self.ks, self.ke + 1)
        w = self.power.clip(min=1e-12)
        self.kc = float(np.sum(bins * w) / np.sum(w)) if w.sum() != 0 else float((self.ks + self.ke) / 2)
        var = float(np.sum(((bins - self.kc)**2) * w) / np.sum(w)) if w.sum() != 0 else 0.0
        self.bw = 2.0 * np.sqrt(var)  # approximate bandwidth metric
        self.peak = float(np.max(self.power)) if self.power.size > 0 else 0.0
        self.mean = float(np.mean(self.power)) if self.power.size > 0 else 0.0
        self.duration = int(self.t_end - self.t_start + 1)

    def extend(self, t_idx, new_ks, new_kp, new_ke, new_power_vals):
        # merge frequency span and average overlapping bins
        all_min = min(self.ks, new_ks)
        all_max = max(self.ke, new_ke)
        length = all_max - all_min + 1
        merged = np.zeros(length)
        counts = np.zeros(length)
        # place old
        merged[self.ks - all_min : self.ke - all_min + 1] += self.power
        counts[self.ks - all_min : self.ke - all_min + 1] += 1
        # place new
        new_power_vals = np.array(new_power_vals, dtype=float)
        merged[new_ks - all_min : new_ke - all_min + 1] += new_power_vals
        counts[new_ks - all_min : new_ke - all_min + 1] += 1
        counts[counts == 0] = 1
        merged = merged / counts
        self.ks, self.ke = all_min, all_max
        self.power = merged
        self.t_end = t_idx
        self._update_stats()

def detect_pulses_from_grid(grid, refs=None, energy_threshold=ENERGY_THRESH, B_db=B_DB):
    """
    Airshark-style pulse detector and merger.
    Implements:
      1) Strict continuation (CF/BW equality, peak ≤3 dB)
      2) Overlap-tolerant merging (freq/time overlap ≥0.5)
      3) Partial spectral match for always-overlapping signals
         (BWpar ≥0.6×BW, angular diff ≤0.15 rad)
    Returns list of completed Pulse objects.
    """
    arr = np.array(grid, float)
    T, N = arr.shape
    linear_tol = 10 ** (-B_db / 20.0)
    active, completed = [], []

    for t in range(T):
        row = arr[t]
        # ---- local maxima detection ----
        peaks = [k for k in range(1, N - 1)
                 if row[k] > row[k - 1] and row[k] > row[k + 1] and row[k] >= energy_threshold]

        new_pulses = []
        for kp in peaks:
            peak_val = row[kp]
            left = kp
            while left - 1 >= 0 and row[left - 1] >= energy_threshold and row[left - 1] >= peak_val * linear_tol:
                left -= 1
            right = kp
            while right + 1 < N and row[right + 1] >= energy_threshold and row[right + 1] >= peak_val * linear_tol:
                right += 1
            new_pulses.append((left, kp, right, row[left:right + 1].tolist()))

        matched_active = set()

        # ---- main matching logic ----
        for (l, kp, r, pvals) in new_pulses:
            matched = False
            peak_val = np.max(pvals)
            kc_new = (l + r) / 2.0
            bw_new = r - l + 1

            for idx, act in enumerate(active):
                # ============ 1) STRICT CONTINUATION ============
                if int(round(act.kc)) == int(round(kc_new)) and int(round(act.bw)) == int(round(bw_new)):
                    if abs(20 * np.log10(max(1e-12, peak_val)) -
                           20 * np.log10(max(1e-12, act.peak))) <= 3.0:
                        act.extend(t, l, kp, r, pvals)
                        matched = True
                        matched_active.add(idx)
                        break

                # ============ 2) OVERLAP-TOLERANT MERGING ============
                if not matched:
                    f1_start, f1_end = act.l, act.r
                    f2_start, f2_end = l, r
                    f_overlap = max(0, min(f1_end, f2_end) - max(f1_start, f2_start))
                    f_union = max(f1_end, f2_end) - min(f1_start, f2_start)
                    FOR = f_overlap / float(f_union + 1e-12)

                    t_overlap = max(0, min(act.t_end, t) - max(act.t_start, act.t_start))
                    t_union = max(act.t_end, t) - min(act.t_start, act.t_start)
                    TOR = t_overlap / float(t_union + 1e-12)

                    if FOR > 0.5 and TOR > 0.5:
                        act.extend(t, l, kp, r, pvals)
                        matched = True
                        matched_active.add(idx)
                        break

                # ============ 3) PARTIAL SPECTRAL MATCH ============
                if not matched and refs is not None and getattr(act, "type", None):
                    f1_start, f1_end = act.l, act.r
                    f2_start, f2_end = l, r
                    overlap = max(0, min(f1_end, f2_end) - max(f1_start, f2_start))
                    bw_ref = max(f1_end - f1_start + 1, 1)
                    if overlap / bw_ref >= 0.6:  # BWmin = 0.6×BW
                        # overlapping region shape
                        sub_sig = np.abs(pvals[max(0, f1_start - l):min(len(pvals), f1_end - l)])
                        sub_sig = sub_sig / (np.max(sub_sig) + 1e-12)
                        # reference shape
                        ref_sig = np.array(refs.get(act.type.lower(), np.zeros_like(sub_sig)))
                        n = min(len(ref_sig), len(sub_sig))
                        ref_norm = ref_sig[:n] / (np.linalg.norm(ref_sig[:n]) + 1e-12)
                        sub_norm = sub_sig[:n] / (np.linalg.norm(sub_sig[:n]) + 1e-12)
                        cos_theta = np.clip(np.dot(ref_norm, sub_norm), -1.0, 1.0)
                        ang_diff = np.arccos(cos_theta)
                        if ang_diff <= 0.15:  # ≤8.6°
                            act.extend(t, l, kp, r, pvals)
                            matched = True
                            matched_active.add(idx)
                            break

            if not matched:
                active.append(Pulse(t, l, kp, r, pvals))

        # ---- terminate unmatched actives ----
        new_active = [act for i, act in enumerate(active) if i in matched_active]
        for i, act in enumerate(active):
            if i not in matched_active:
                completed.append(act)
        active = new_active

    completed.extend(active)
    return completed


# ---------------- spectral helpers ----------------
def spectral_signature(vec):
    v = np.array(vec, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def angular_difference(sig_ref, sig_meas):
    # resample measured to ref length if needed
    if len(sig_ref) != len(sig_meas):
        sig_meas = np.interp(np.linspace(0, len(sig_meas)-1, num=len(sig_ref)), np.arange(len(sig_meas)), sig_meas)
    nr = np.linalg.norm(sig_ref)
    nm = np.linalg.norm(sig_meas)
    if nr == 0 or nm == 0:
        return float(np.pi/2)
    cosang = float(np.dot(sig_ref, sig_meas) / (nr * nm))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.arccos(cosang))

def spectral_flatness(power_vec):
    # geometric mean / arithmetic mean
    v = np.array(power_vec, dtype=float) + 1e-12
    geo = np.exp(np.mean(np.log(v)))
    arith = np.mean(v)
    return float(geo / (arith + 1e-12))

def cross_correlation(sig_ref, sig_meas):
    # normalized cross-correlation peak
    a = np.array(sig_ref, dtype=float)
    b = np.array(sig_meas, dtype=float)
    # resample b to len(a)
    if len(a) != len(b):
        b = np.interp(np.linspace(0, len(b)-1, num=len(a)), np.arange(len(b)), b)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    corr = correlate((a - np.mean(a)) / np.std(a), (b - np.mean(b)) / np.std(b), mode='full')
    return float(np.max(np.abs(corr)) / len(a))

# ---------------- build reference signatures (isolated) ----------------
def build_reference_signatures(x, y, n_per_type=40):
    """
    For each device type, generate n_per_type isolated frames using add_* functions,
    detect pulses and average their normalized spectral shapes to form reference signatures.
    """
    ref_path = os.path.join(RESULTS_DIR, "reference_signatures.json")
    if os.path.exists(ref_path):
        print("[refs] Found cached reference_signatures.json — loading instead of rebuilding.")
        with open(ref_path, "r") as f:
            data = json.load(f)
        # Convert lists back to numpy arrays
        return {k: np.array(v) for k, v in data.items()}


    refs = {d: [] for d in DEVICES}
    print("[refs] Building reference signatures (isolated signals)...")
    for d in DEVICES:
        for i in range(n_per_type):
            base = ai.generate_gaussian_noise_grid(x, y, mean=10, variance=2)
            grid = base.copy()
            positions_log = []
            if d == "wifi":
                grid = ai.add_wifi_clients(base, grid, x, y, n_clients=1, positions_log=positions_log)
            elif d == "zigbee":
                grid = ai.add_zigbee_clients(base, grid, x, y, n_clients=1, positions_log=positions_log)
            elif d == "bluetooth":
                grid = ai.add_bluetooth_clients(base, grid, x, y, n_clients=1, positions_log=positions_log)
            elif d == "cordless":
                grid = ai.add_cordless_phone_clients(base, grid, x, y, n_clients=1, positions_log=positions_log)
            # normalize
            grid -= grid.min()
            grid = (grid / (grid.max() + 1e-12)) * 255.0
            pulses = detect_pulses_from_grid(grid)
            # pick largest/strongest pulse as signature
            if len(pulses) > 0:
                best = max(pulses, key=lambda p: p.peak)
                sig = spectral_signature(best.power)
                # resample to fixed length L
                L = 32
                sig_r = np.interp(np.linspace(0, len(sig)-1, num=L), np.arange(len(sig)), sig)
                refs[d].append(sig_r)
    # average
    refs_avg = {}
    for d in DEVICES:
        if len(refs[d]) == 0:
            refs_avg[d] = np.zeros(32)
        else:
            refs_avg[d] = spectral_signature(np.mean(np.vstack(refs[d]), axis=0))
        print(f"[refs] {d}: collected {len(refs[d])} sigs")
    with open(os.path.join(RESULTS_DIR, "reference_signatures.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in refs_avg.items()}, f, indent=2)
    print("[refs] saved reference_signatures.json")
    return refs_avg

# ---------------- feature extraction per pulse (paper features) ----------------
def pulse_features(pulse, grid, refs, y_bins):
    """
    Extract the feature vector per the paper (spectral, temporal, power, shape, overlap).
    Returns (fv, meta) where meta contains human-readable scalar values for JSON.
    """
    # center freq bin (kc), bandwidth estimate, duration
    center_bin = float(pulse.kc)
    bw_px = float(max(1.0, pulse.bw))
    duration = float(pulse.duration)
    # amplitude/energy stats
    peak = float(pulse.peak)
    mean_power = float(np.mean(pulse.power)) if pulse.power.size > 0 else 0.0
    var_power = float(np.var(pulse.power)) if pulse.power.size > 0 else 0.0
    # duty cycle: fraction of time rows in which this freq span had energy
    span_start = max(0, int(pulse.ks))
    span_end = min(int(pulse.ke)+1, len(y_bins))
    rows_active = np.sum(np.any(grid[:, span_start:span_end] > ENERGY_THRESH, axis=1))
    duty = float(rows_active) / (grid.shape[0] + 1e-12)
    noise_floor = float(np.median(grid))
    energy_ratio = peak / (noise_floor + 1e-12)
    # shape-based features
    sig_meas = spectral_signature(pulse.power)
    flatness = spectral_flatness(pulse.power)
    kurt = float(kurtosis(pulse.power)) if pulse.power.size > 0 else 0.0
    # cross-correlation/angular difference to refs
    corr_wifi = cross_correlation(refs["wifi"], sig_meas)
    corr_zigbee = cross_correlation(refs["zigbee"], sig_meas)
    corr_bt = cross_correlation(refs["bluetooth"], sig_meas)
    corr_cord = cross_correlation(refs["cordless"], sig_meas)
    ang_wifi = angular_difference(refs["wifi"], sig_meas)
    ang_zig = angular_difference(refs["zigbee"], sig_meas)
    # overlap ratio: fraction of this pulse's bandwidth that also contains other pulses at the same times
    # approximate by counting in the grid where multiple active bins present in same time rows
    overlap_count_rows = np.sum(np.sum(grid[:, span_start:span_end] > ENERGY_THRESH, axis=1) > 1)
    overlap_ratio = float(overlap_count_rows) / (grid.shape[0] + 1e-12)
    # partial-match ratios r_* per paper (how much measured bins match reference)
    def partial_ratio(sig_ref, sig_meas):
        if np.max(sig_meas) == 0:
            return 0.0
        Lref = len(sig_ref)
        smeas = np.interp(np.linspace(0, len(sig_meas)-1, num=Lref),
                          np.arange(len(sig_meas)), sig_meas)
        thr = 0.6 * np.max(smeas)
        match_bins = np.sum((smeas >= thr) & (sig_ref > 0.01))
        return float(match_bins) / float(max(1, Lref))
    r_wifi = partial_ratio(refs["wifi"], sig_meas)
    r_zig = partial_ratio(refs["zigbee"], sig_meas)
    r_bt = partial_ratio(refs["bluetooth"], sig_meas)
    r_cord = partial_ratio(refs["cordless"], sig_meas)

    # feature vector (ordered)
    fv = np.array([
        center_bin, bw_px, duration, peak, mean_power, var_power,
        duty, noise_floor, energy_ratio, flatness, kurt,
        corr_wifi, corr_zigbee, corr_bt, corr_cord,
        ang_wifi, ang_zig,
        overlap_ratio, r_wifi, r_zig, r_bt, r_cord
    ], dtype=float)

    meta = {
        "center_bin": center_bin,
        "bandwidth_px": bw_px,
        "horizontal_width": duration,
        "amplitude": peak,
        "mean_power": mean_power,
        "var_power": var_power,
        "duty_cycle": duty,
        "noise_floor": noise_floor,
        "energy_ratio": energy_ratio,
        "spectral_flatness": flatness,
        "kurtosis": kurt,
        "corr_wifi": corr_wifi,
        "corr_zigbee": corr_zigbee,
        "corr_bluetooth": corr_bt,
        "corr_cordless": corr_cord,
        "ang_wifi": ang_wifi,
        "ang_zigbee": ang_zig,
        "overlap_ratio": overlap_ratio,
        "r_wifi": r_wifi,
        "r_zig": r_zig,
        "r_bt": r_bt,
        "r_cord": r_cord
    }
    return fv, meta

# ---------------- labeling pulses by frame_info (ground truth) ----------------
def label_pulse_by_frameinfo(pulse, frame_info, y_bins):
    """
    Return set of device keys present for this pulse by matching center frequency
    with frame_info events (generator's ground truth).
    Accept if center bin within half bw +/- margin.
    """
    found = set()
    pcb = pulse.kc
    for ev in frame_info.get("events", []):
        ev_type = ev.get("type", "").lower()
        # infer bin index
        if ev.get("center_freq") is not None:
            ev_bin = np.interp(ev["center_freq"], [2400.0, 2483.5], [0, len(y_bins)])
        else:
            ev_bin = ev.get("center_y", None)
        if ev_bin is None:
            continue
        bw = ev.get("bandwidth_px", 1)
        if abs(pcb - ev_bin) <= max(1.0, bw / 2.0 + 1.0):
            if "wifi" in ev_type:
                found.add("wifi")
            elif "zigbee" in ev_type:
                found.add("zigbee")
            elif "bluetooth" in ev_type or "ble" in ev_type:
                found.add("bluetooth")
            elif "cordless" in ev_type or "phone" in ev_type:
                found.add("cordless")
    return found

# ---------------- Build augmented multi-label dataset ----------------
def build_augmented_dataset(x, y, refs, n_isolated_per=15, n_mixed=120):
    """
    Build training dataset that includes:
    - isolated examples per device (n_isolated_per each)
    - mixed frames including 2-3 device types (n_mixed)
    Labels are multi-label binary per-device (dict of arrays).
    """
    dataset_path = os.path.join(RESULTS_DIR, "training_dataset.npz")
    
    # ---- ① Check for existing dataset ----
    if os.path.exists(dataset_path):
        print("[train] Found cached training_dataset.npz — loading instead of regenerating.")
        data = np.load(dataset_path, allow_pickle=True)
        return data["X"], data["y"]
 
    X_list = []
    Ys = {d: [] for d in DEVICES}
    print("[data] Building isolated examples...")
    # Isolated
    for d in DEVICES:
        for i in range(n_isolated_per):
            base = ai.generate_gaussian_noise_grid(x, y, mean=10, variance=2)
            grid = base.copy()
            positions = []
            if d == "wifi":
                grid = ai.add_wifi_clients(base, grid, x, y, n_clients=1, positions_log=positions)
            elif d == "zigbee":
                grid = ai.add_zigbee_clients(base, grid, x, y, n_clients=1, positions_log=positions)
            elif d == "bluetooth":
                grid = ai.add_bluetooth_clients(base, grid, x, y, n_clients=1, positions_log=positions)
            elif d == "cordless":
                grid = ai.add_cordless_phone_clients(base, grid, x, y, n_clients=1, positions_log=positions)
            grid -= grid.min(); grid = (grid / (grid.max()+1e-12)) * 255.0
            # create fake frame_info from positions for labeling
            frame_info = {"events": positions}
            pulses = detect_pulses_from_grid(grid)
            for p in pulses:
                fv, _ = pulse_features(p, grid, refs, y)
                X_list.append(fv)
                labels = label_pulse_by_frameinfo(p, frame_info, y)
                for dev in DEVICES:
                    Ys[dev].append(1 if dev in labels else 0)

    # Mixed augmentation
    print("[data] Building mixed examples (augmentation)...")
    types_list = {
        "wifi": lambda base, grid: ai.add_wifi_clients(base, grid, x, y, n_clients=random.randint(1,2), positions_log=positions_log),
        "zigbee": lambda base, grid: ai.add_zigbee_clients(base, grid, x, y, n_clients=random.randint(1,3), positions_log=positions_log),
        "bluetooth": lambda base, grid: ai.add_bluetooth_clients(base, grid, x, y, n_clients=random.randint(2,6), positions_log=positions_log),
        "cordless": lambda base, grid: ai.add_cordless_phone_clients(base, grid, x, y, n_clients=random.randint(1,2), positions_log=positions_log)
    }
    for i in range(n_mixed):
        base = ai.generate_gaussian_noise_grid(x, y, mean=10, variance=2)
        grid = base.copy()
        positions_log = []
        chosen = random.sample(DEVICES, k=random.choice([2,3]))
        for c in chosen:
            grid = types_list[c](base, grid)
        grid -= grid.min(); grid = (grid / (grid.max()+1e-12)) * 255.0
        frame_info = {"events": positions_log}
        pulses = detect_pulses_from_grid(grid)
        for p in pulses:
            fv, _ = pulse_features(p, grid, refs, y)
            X_list.append(fv)
            labels = label_pulse_by_frameinfo(p, frame_info, y)
            for dev in DEVICES:
                Ys[dev].append(1 if dev in labels else 0)

    X = np.vstack(X_list) if len(X_list) > 0 else np.zeros((1, 22))
    Ys = {k: np.array(v, dtype=int) for k, v in Ys.items()}
    print(f"[data] Built X shape {X.shape} and labels counts:", {k: int(Ys[k].sum()) for k in Ys})
    return X, Ys

# ---------------- Train per-device C4.5-style decision trees ----------------
def train_per_device_trees(X, Ys):
    models = {}
    for dev in DEVICES:
        y = Ys[dev]
        if len(y) == 0:
            raise ValueError("No training labels for device: " + dev)
        # we'll stratify split if there are positives and negatives
        if len(np.unique(y)) > 1:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=RANDOM_SEED)
            clf.fit(Xtr, ytr)
            # optional evaluation
            ypred = clf.predict(Xte)
            prec, rec, f1, _ = precision_recall_fscore_support(yte, ypred, average='binary', zero_division=0)
            print(f"[train] {dev}: prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} positives={y.sum()}/{len(y)}")
        else:
            # degenerate: all same label -> train trivial classifier
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=RANDOM_SEED)
            clf.fit(X, y)
            print(f"[train] {dev}: only one class present in training labels, trained trivial tree.")
        models[dev] = clf
    joblib.dump({"models": models}, os.path.join(RESULTS_DIR, "airshark_c45_models.joblib"))
    print("[train] saved models to results/airshark_c45_models.joblib")
    return models

# ---------------- classify a grid and build JSON output (per paper format) ----------------
def classify_grid_to_json(grid, x, y, refs, models, prob_thresh=0.5):
    pulses = detect_pulses_from_grid(grid)
    events = []
    for p in pulses:
        fv, meta = pulse_features(p, grid, refs, y)
        fv2 = fv.reshape(1, -1)
        # multi-label per-device: each device analyzer returns 0/1 using decision tree
        for dev in DEVICES:
            clf = models[dev]
            confidence = 0.0
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(fv2)[0]
                # find positive class index
                if 1 in clf.classes_:
                    idx = list(clf.classes_).index(1)
                    confidence = float(probs[idx])
                else:
                    confidence = float(probs[-1])
            else:
                pred = clf.predict(fv2)[0]
                confidence = 1.0 if pred == 1 else 0.0
            if confidence >= prob_thresh:
                center_freq = round(float(np.interp(meta["center_bin"], [0, len(y)], [2400.0, 2483.5])), 2)
                ev = {
                    "type": "Wi-Fi" if dev == "wifi" else ("Zigbee" if dev == "zigbee" else ("Bluetooth" if dev == "bluetooth" else "Cordless-phone")),
                    "confidence": round(confidence, 2),
                    "center_freq": center_freq,
                    "bandwidth_px": int(round(meta["bandwidth_px"])),
                    "horizontal_width": int(round(meta["horizontal_width"])),
                    "amplitude": meta["amplitude"],
                    "duty_cycle": float(meta["duty_cycle"]),
                    "noise_floor": meta["noise_floor"]
                }
                events.append(ev)
    frame_info = {
        "timestamp": datetime.now().isoformat(),
        "channel": random.choice([1,6,11]),
        "band": "2.4GHz",
        "airtime_cost": round(random.uniform(0.01, 0.99), 3),
        "events": events
    }
    return frame_info

# ---------------- Compare predicted JSON to real ground truth ----------------
def compare_predictions(pred, real, y_bins, tol_mhz=5.0):
    """
    Match predicted events to ground-truth events by nearest center freq within tol_mhz.
    Compute TP/FP/FN per device (device strings matching).
    """
    def dev_key(ev_type):
        et = ev_type.lower()
        if "wifi" in et:
            return "wifi"
        if "zigbee" in et:
            return "zigbee"
        if "bluetooth" in et:
            return "bluetooth"
        if "cordless" in et or "phone" in et:
            return "cordless"
        return None

    gt = real.get("events", [])
    pred_ev = pred.get("events", [])
    gt_used = set()
    stats = {d: {"tp":0,"fp":0,"fn":0} for d in DEVICES}

    # attempt best-match for each predicted event
    for i, pe in enumerate(pred_ev):
        pdev = dev_key(pe["type"])
        pfreq = pe.get("center_freq")
        best_j = None
        best_dist = 1e9
        for j, ge in enumerate(gt):
            gfreq = ge.get("center_freq", None)
            if gfreq is None:
                gfreq = np.interp(ge.get("center_y", 0), [0, len(y_bins)], [2400.0, 2483.5])
            dist = abs(pfreq - gfreq)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None and best_dist <= tol_mhz:
            # matched - check device types
            gdev = dev_key(gt[best_j]["type"])
            if gdev == pdev:
                stats[pdev]["tp"] += 1
                gt_used.add(best_j)
            else:
                # predicted wrong device at that freq -> FP for pdev, mark GT not used (FN counted later)
                stats[pdev]["fp"] += 1
        else:
            # no matching GT freq -> false positive
            if pdev:
                stats[pdev]["fp"] += 1

    # now count false negatives for GT events not matched
    for j, ge in enumerate(gt):
        if j not in gt_used:
            gdev = dev_key(ge["type"])
            if gdev:
                stats[gdev]["fn"] += 1

    return stats

# ---------------- Main pipeline ----------------
def main():
    print("Airshark C4.5-style pipeline starting...")

    # set x,y bins consistent with meta-data.py (user used y=256 in provided code snippet)
    x = np.arange(0, 240, 1)   # time bins
    y = np.arange(0, 256, 1)   # freq bins

    # 1) Build reference signatures using isolated signals
    refs = build_reference_signatures(x, y, n_per_type=30)

    # 2) Build augmented multi-label training dataset (isolated + mixed)
    X, Ys = build_augmented_dataset(x, y, refs, n_isolated_per=30, n_mixed=120)

    # 3) Train per-device Decision Trees (criterion='entropy' ~ C4.5)
    models = train_per_device_trees(X, Ys)

    # Save refs + models
    joblib.dump({"models": models, "refs": refs}, os.path.join(RESULTS_DIR, "airshark_c45_models_and_refs.joblib"))

    # 4) Generate one real mixed test frame from meta-data.py (this produces generator JSON)
    print("[test] Generating one real mixed test frame from meta-data.py")
    img_path, frame_real, grid = meta.generate_interference_frame(x, y)
    # save real generator JSON
    with open(os.path.join(RESULTS_DIR, "interference_real.json"), "w") as f:
        json.dump(frame_real, f, indent=4)
    # save image visualization
    plot_noise_grid(x, y, grid, "real_interference", os.path.join(RESULTS_DIR, "interference_real.png"))
    print(f"[test] Saved ground-truth JSON to results/interference_real.json and image to results/interference_real.png")

    # 5) Classify the grid using trained trees
    pred = classify_grid_to_json(grid, x, y, refs, models, prob_thresh=0.5)
    with open(os.path.join(RESULTS_DIR, "interference_report_pred.json"), "w") as f:
        json.dump(pred, f, indent=4)
    print(f"[test] Saved predicted JSON to results/interference_report_pred.json")

    # 6) Compare predictions to ground truth
    stats = compare_predictions(pred, frame_real, y, tol_mhz=5.0)
    print("\n=== Comparison (per-device) ===")
    for d in DEVICES:
        s = stats[d]
        print(f"{d}: TP={s['tp']} FP={s['fp']} FN={s['fn']}")

    print("\nSaved artifacts:")
    print(" - results/reference_signatures.json")
    print(" - results/airshark_c45_models.joblib and airshark_c45_models_and_refs.joblib")
    print(" - results/interference_real.json (ground truth from generator)")
    print(" - results/interference_real.png")
    print(" - results/interference_report_pred.json (predicted)")

if __name__ == "__main__":
    main()
