# airshark_final.py
"""
Airshark-style pipeline:
- Uses meta-data.py's generate_interference_frame(x,y) to get image, ground-truth, and grid.
- Builds real reference spectral signatures from generated frames.
- Detects pulses robustly (local maxima, expand by dB, match across time).
- Extracts robust features (incl. partial-bandwidth ratio and angular diffs).
- Trains one DecisionTreeClassifier per device (wifi, bluetooth, zigbee, cordless).
- Classifies pulses in a demo frame and writes results in JSON format similar to user's example.
"""

import os
import json
import random
import numpy as np
from datetime import datetime
from importlib.machinery import SourceFileLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------------------
META_PATH = "meta-data.py"        # your generator module (must define generate_interference_frame)
RESULTS_DIR = "results"
N_REF_FRAMES = 60                 # frames to build reference signatures
N_TRAIN_FRAMES = 300              # frames to build training set
MIN_PULSE_ENERGY = 10.0
B_DB = 10.0
PROB_THRESH = {"wifi": 0.5, "bluetooth": 0.5, "zigbee": 0.5, "cordless": 0.5}
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------- import meta-data.py dynamically --------------
if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"{META_PATH} not found. Place meta-data.py in same folder.")
meta = SourceFileLoader("meta_module", META_PATH).load_module()
if not hasattr(meta, "generate_interference_frame"):
    raise AttributeError("meta-data.py must define generate_interference_frame(x,y) returning (img_path, frame_info, grid)")

# -------------------- helpers: plotting ------------------------------
def plot_noise_grid(x, y, grid, title, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.imshow(grid, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=255)
    plt.colorbar(label='Power Intensity (0–255)')
    plt.title(title)
    plt.xlabel('X-axis (0–100)')
    plt.ylabel('Y-axis (0–20)')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# -------------------- Pulse class + detector -----------------------
class Pulse:
    def __init__(self, t_idx, ks, kp, ke, power_vals):
        self.t_start = t_idx
        self.t_end = t_idx
        self.ks = ks
        self.kp = kp
        self.ke = ke
        self.power = np.array(power_vals, dtype=float)
        self._update_stats()

    def _update_stats(self):
        bins = np.arange(self.ks, self.ke + 1)
        w = self.power.clip(min=1e-12)
        self.kc = (np.sum(bins * w) / np.sum(w)) if w.sum() != 0 else (self.ks + self.ke) / 2.0
        var = (np.sum(((bins - self.kc) ** 2) * w) / np.sum(w)) if w.sum() != 0 else 0.0
        self.bw = 2.0 * np.sqrt(var)
        self.peak_power = float(np.max(self.power)) if self.power.size else 0.0

    def extend(self, new_t_idx, new_ks, new_kp, new_ke, new_power_vals):
        all_min = min(self.ks, new_ks)
        all_max = max(self.ke, new_ke)
        length = all_max - all_min + 1
        merged = np.zeros(length)
        counts = np.zeros(length)
        merged[self.ks - all_min : self.ke - all_min + 1] += self.power
        counts[self.ks - all_min : self.ke - all_min + 1] += 1
        merged[new_ks - all_min : new_ke - all_min + 1] += new_power_vals
        counts[new_ks - all_min : new_ke - all_min + 1] += 1
        counts[counts == 0] = 1
        merged = merged / counts
        self.ks, self.ke = all_min, all_max
        self.power = merged
        self.t_end = new_t_idx
        self._update_stats()

def detect_pulses_from_grid(grid, energy_threshold=MIN_PULSE_ENERGY, B_db=B_DB):
    arr = np.array(grid).astype(float)
    time_steps, n_bins = arr.shape
    active = []
    completed = []
    linear_tol = 10 ** (-B_db / 20.0)
    min_energy = energy_threshold

    for t in range(time_steps):
        sample = arr[t]
        peaks = []
        # strict local maxima detection
        for k in range(1, n_bins-1):
            if sample[k] > sample[k-1] and sample[k] > sample[k+1] and sample[k] >= min_energy:
                peaks.append(k)
        new_pulses = []
        for kp in peaks:
            peak_power = sample[kp]
            left = kp
            while left-1 >=0 and sample[left-1] >= min_energy and sample[left-1] >= peak_power * linear_tol:
                left -= 1
            right = kp
            while right+1 < n_bins and sample[right+1] >= min_energy and sample[right+1] >= peak_power * linear_tol:
                right += 1
            new_pulses.append((left, kp, right, sample[left:right+1].tolist()))
        matched_active = set()
        for (l, kp, r, pvals) in new_pulses:
            matched = False
            for idx, act in enumerate(active):
                overlap = not (r < act.ks or l > act.ke)
                if overlap:
                    peak_diff = abs(np.max(pvals) - act.peak_power)
                    if peak_diff <= max(1.0, 0.5*act.peak_power):
                        act.extend(t, l, kp, r, pvals)
                        matched = True
                        matched_active.add(idx)
                        break
            if not matched:
                active.append(Pulse(t, l, kp, r, pvals))
        # terminate unmatched actives
        new_active = []
        for idx, act in enumerate(active):
            if idx in matched_active:
                new_active.append(act)
            else:
                completed.append(act)
        active = new_active
    # add remaining
    completed.extend(active)
    return completed

# -------------------- spectral helpers & signatures ----------------
def spectral_signature(vec):
    v = np.array(vec, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def angular_difference(sig_ref, sig_meas):
    if len(sig_ref) != len(sig_meas):
        sig_meas = np.interp(np.linspace(0, len(sig_meas)-1, num=len(sig_ref)), np.arange(len(sig_meas)), sig_meas)
    if np.linalg.norm(sig_ref) == 0 or np.linalg.norm(sig_meas) == 0:
        return float(np.pi/2)
    cosang = np.dot(sig_ref, sig_meas) / (np.linalg.norm(sig_ref) * np.linalg.norm(sig_meas))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.arccos(cosang))

# -------------------- build real reference signatures ----------------
def build_reference_signatures(n_frames, x, y_bins):
    # collect per-device measured signature vectors
    collected = {"wifi": [], "zigbee": [], "bluetooth": [], "cordless": []}
    print(f"[refs] building reference signatures from {n_frames} frames...")
    for i in range(n_frames):
        img_path, frame_info, grid = meta.generate_interference_frame(x, y_bins)
        pulses = detect_pulses_from_grid(grid)
        # match each ground-truth event to nearest detected pulse and store its normalized signature
        for ev in frame_info.get("events", []):
            t = ev.get("type", "").lower()
            if "wifi" in t:
                key = "wifi"
            elif "zigbee" in t:
                key = "zigbee"
            elif "bluetooth" in t or "ble" in t:
                key = "bluetooth"
            elif "cordless" in t or "phone" in t:
                key = "cordless"
            else:
                continue
            # pick center bin either from center_freq (if present) or center_y
            center_bin = None
            if ev.get("center_freq") is not None:
                center_bin = np.interp(ev["center_freq"], [2400.0, 2483.5], [0, len(y_bins)])
            elif ev.get("center_y") is not None:
                center_bin = ev["center_y"]
            if center_bin is None:
                continue
            # find pulse nearest to center_bin
            best = None
            bestd = 1e9
            for p in pulses:
                d = abs(p.kc - center_bin)
                if d < bestd:
                    bestd = d
                    best = p
            if best is not None:
                sig = spectral_signature(best.power)
                # resample to fixed length (32) for reference averaging
                L = 32
                sig_r = np.interp(np.linspace(0, len(sig)-1, num=L), np.arange(len(sig)), sig)
                collected[key].append(sig_r)
    # average per key
    refs = {}
    for k, vecs in collected.items():
        if len(vecs) == 0:
            refs[k] = np.zeros(32)
        else:
            refs[k] = spectral_signature(np.mean(np.vstack(vecs), axis=0))
        print(f"[refs] {k}: collected {len(vecs)} sigs")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "reference_signatures.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in refs.items()}, f, indent=2)
    print("[refs] saved reference_signatures.json")
    return refs

# -------------------- pulse -> feature vector -----------------------
def partial_ratio(sig_ref, sig_meas):
    # resample measured to reference length
    if len(sig_meas) == 0 or np.max(sig_meas) == 0:
        return 0.0
    Lref = len(sig_ref)
    smeas = np.interp(np.linspace(0, len(sig_meas)-1, num=Lref), np.arange(len(sig_meas)), sig_meas)
    thr = 0.6 * np.max(smeas)
    matches = np.sum((smeas >= thr) & (sig_ref > 0.01))
    return float(matches) / float(max(1, Lref))

def pulse_feature_vector(pulse, grid, refs, y_bins):
    amp = float(pulse.peak_power)
    bw_px = float(pulse.bw)
    horiz = float(pulse.t_end - pulse.t_start + 1)
    center_bin = float(pulse.kc)
    span = slice(max(0, int(pulse.ks)), min(len(y_bins), int(pulse.ke)+1))
    rows_with_energy = np.sum(np.any(grid[:, span] > MIN_PULSE_ENERGY, axis=1))
    duty = float(rows_with_energy) / (grid.shape[0] + 1e-9)
    noise_floor = float(np.median(grid))
    energy_ratio = amp / (noise_floor + 1e-9)
    sig_meas = spectral_signature(pulse.power)
    ang_z = angular_difference(refs["zigbee"], sig_meas)
    ang_b = angular_difference(refs["bluetooth"], sig_meas)
    ang_c = angular_difference(refs["cordless"], sig_meas)
    overlap_count = int(np.sum(np.sum(grid[:, span] > MIN_PULSE_ENERGY, axis=1) > 1))
    r_z = partial_ratio(refs["zigbee"], sig_meas)
    r_b = partial_ratio(refs["bluetooth"], sig_meas)
    r_c = partial_ratio(refs["cordless"], sig_meas)
    features = {
        "amplitude": amp,
        "bandwidth_px": bw_px,
        "horizontal_width": horiz,
        "center_bin": center_bin,
        "duty": duty,
        "noise_floor": noise_floor,
        "energy_ratio": energy_ratio,
        "ang_zigbee": ang_z,
        "ang_bluetooth": ang_b,
        "ang_cordless": ang_c,
        "overlap_count": overlap_count,
        "r_z": r_z,
        "r_b": r_b,
        "r_c": r_c
    }
    fv = np.array([
        features["amplitude"],
        features["bandwidth_px"],
        features["horizontal_width"],
        features["center_bin"],
        features["duty"],
        features["noise_floor"],
        features["energy_ratio"],
        features["ang_zigbee"],
        features["ang_bluetooth"],
        features["ang_cordless"],
        features["overlap_count"],
        features["r_z"],
        features["r_b"],
        features["r_c"]
    ], dtype=float)
    return fv, features

# -------------------- label pulses using meta-data frame_info ----------
def label_pulse_by_frameinfo(pulse, frame_info, y_bins):
    found = set()
    pcb = pulse.kc
    for ev in frame_info.get("events", []):
        # center bin from center_freq or center_y
        if ev.get("center_freq") is not None:
            ev_bin = np.interp(ev["center_freq"], [2400.0, 2483.5], [0, len(y_bins)])
        else:
            ev_bin = ev.get("center_y", None)
        if ev_bin is None:
            continue
        bw_px = ev.get("bandwidth_px", 1)
        if abs(pcb - ev_bin) <= max(1.0, bw_px / 2.0 + 1.0):
            t = ev.get("type", "").lower()
            if "wifi" in t:
                found.add("wifi")
            elif "zigbee" in t:
                found.add("zigbee")
            elif "bluetooth" in t or "ble" in t:
                found.add("bluetooth")
            elif "cordless" in t or "phone" in t:
                found.add("cordless")
    return found

# -------------------- build per-pulse training dataset using meta-data gen -------
def build_pulse_dataset(n_frames, x, y_bins, refs):
    X = []
    Ys = {"wifi": [], "bluetooth": [], "zigbee": [], "cordless": []}
    print(f"[train] building dataset from {n_frames} frames...")
    for i in range(n_frames):
        img_path, frame_info, grid = meta.generate_interference_frame(x, y_bins)
        pulses = detect_pulses_from_grid(grid)
        for p in pulses:
            fv, meta_feat = pulse_feature_vector(p, grid, refs, y_bins)
            X.append(fv)
            found = label_pulse_by_frameinfo(p, frame_info, y_bins)
            Ys["wifi"].append(1 if "wifi" in found else 0)
            Ys["bluetooth"].append(1 if "bluetooth" in found else 0)
            Ys["zigbee"].append(1 if "zigbee" in found else 0)
            Ys["cordless"].append(1 if "cordless" in found else 0)
    X = np.vstack(X) if len(X) > 0 else np.zeros((1, 14))
    Ys = {k: np.array(v, dtype=int) if len(v) > 0 else np.zeros((X.shape[0],), dtype=int) for k, v in Ys.items()}
    print(f"[train] pulses: {X.shape[0]}")
    return X, Ys

# -------------------- train per-device DecisionTree classifiers ----------
def train_per_device_classifiers(X, Ys):
    models = {}
    for k, y in Ys.items():
        print(f"[train] {k}: positives={np.sum(y)} / {len(y)}")
        if len(np.unique(y)) == 1:
            clf = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)
            clf.fit(X, y)
        else:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
            clf = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED)
            clf.fit(Xtr, ytr)
            ypred = clf.predict(Xte)
            print(f"=== report ({k}) ===")
            print(classification_report(yte, ypred, zero_division=0))
        models[k] = clf
    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump({"models": models}, os.path.join(RESULTS_DIR, "airshark_models.joblib"))
    print("[train] models saved to results/airshark_models.joblib")
    return models

# -------------------- classify frame and produce JSON ---------------------
def classify_grid_and_build_json(grid, x, y_bins, refs, models, prob_thresholds=PROB_THRESH):
    pulses = detect_pulses_from_grid(grid)
    events = []
    for p in pulses:
        fv, meta = pulse_feature_vector(p, grid, refs, y_bins)
        fv2 = fv.reshape(1, -1)
        for dev, clf in models.items():
            confidence = 0.0
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(fv2)[0]
                # find index of positive class
                if 1 in clf.classes_:
                    idx_pos = list(clf.classes_).index(1)
                    confidence = float(probs[idx_pos])
                else:
                    confidence = float(probs[-1])
            else:
                pred = clf.predict(fv2)[0]
                confidence = 1.0 if pred == 1 else 0.0
            if confidence >= prob_thresholds.get(dev, 0.5):
                center_freq = round(float(np.interp(meta["center_bin"], [0, len(y_bins)], [2400.0, 2483.5])), 2)
                ev = {
                    "type": "Wi-Fi" if dev == "wifi" else ("Bluetooth" if dev == "bluetooth" else ("Zigbee" if dev == "zigbee" else "Cordless-phone")),
                    "confidence": round(float(confidence), 2),
                    "center_freq": center_freq,
                    "bandwidth_px": int(round(meta["bandwidth_px"])),
                    "horizontal_width": int(round(meta["horizontal_width"])),
                    "amplitude": float(meta["amplitude"]),
                    "duty_cycle": float(meta["duty"]),        # keep float like sample
                    "noise_floor": float(meta["noise_floor"])
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

# -------------------- main pipeline -----------------------------------
def main():
    x = np.arange(0, 240, 1)
    # choose y bins consistent with meta-data; try 56 as a default (matches many generator configs)
    y = np.arange(0, 56, 1)

    # Step 1: build references
    refs = build_reference_signatures(N_REF_FRAMES, x, y)

    # Step 2: build training set with labels from meta-data frames
    X, Ys = build_pulse_dataset(N_TRAIN_FRAMES, x, y, refs)

    # Step 3: train per-device decision trees
    models = train_per_device_classifiers(X, Ys)

    # Step 4: generate one demo frame (meta-data provides ground-truth) and classify it
    img_path, frame_info_gt, grid = meta.generate_interference_frame(x, y)

    # Ensure results dir exists, save image
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_noise_grid(x, y, grid, "interference", os.path.join(RESULTS_DIR, "interference.png"))

    # classify
    frame_out = classify_grid_and_build_json(grid, x, y, refs, models, prob_thresholds=PROB_THRESH)

    # Save JSON to results/interference_report.json
    json_out = os.path.join(RESULTS_DIR, "interference_report.json")
    with open(json_out, "w") as f:
        json.dump(frame_out, f, indent=4)

    # Print predicted JSON and ground-truth for inspection
    print("\n=== PREDICTED JSON ===")
    print(json.dumps(frame_out, indent=4))
    print("\n=== GROUND TRUTH (meta-data) ===")
    print(json.dumps(frame_info_gt, indent=4))

    # Save models+refs for reuse
    joblib.dump({"models": models, "refs": refs}, os.path.join(RESULTS_DIR, "airshark_models_and_refs.joblib"))
    print(f"\nSaved outputs:\n - {json_out}\n - {os.path.join(RESULTS_DIR, 'interference.png')}\n - {os.path.join(RESULTS_DIR, 'reference_signatures.json')}\n - {os.path.join(RESULTS_DIR, 'airshark_models_and_refs.joblib')}")

if __name__ == "__main__":
    main()
