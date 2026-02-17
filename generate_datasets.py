"""
generate_datasets.py
Run once locally to create 6 compressed .npz files for the reproducibility lab.
Each file contains a functional connectivity tensor (upper triangle) and behavioral data.

Usage:
    python generate_datasets.py
"""

import numpy as np
import os

# ---------------------------------------------------------------------------
# 1. Load atlas labels and build network lookup
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_FILE = os.path.join(SCRIPT_DIR, "atlas_labels.txt")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

with open(LABEL_FILE) as f:
    ROI_NAMES = [line.strip() for line in f if line.strip()]

N_ROIS = len(ROI_NAMES)
assert N_ROIS == 216, f"Expected 216 ROIs, got {N_ROIS}"

SUBCORTICAL = {"HIP", "AMY", "pTHA", "aTHA", "NAc", "GP", "PUT", "CAU"}


def get_network(label):
    """Return the network name for a given ROI label."""
    # Subcortical: e.g. 'HIP-rh', 'AMY-lh'
    prefix = label.split("-")[0]
    if prefix in SUBCORTICAL:
        return "Subcortical"
    # Cortical: e.g. '17Networks_LH_SalVentAttnA_Ins_1'
    parts = label.split("_")
    if len(parts) >= 3:
        return parts[2]  # e.g. 'SalVentAttnA', 'DefaultA', 'VisCent'
    return "Unknown"


NETWORK_MAP = [get_network(name) for name in ROI_NAMES]
UNIQUE_NETWORKS = sorted(set(NETWORK_MAP))

# Build boolean masks for same-network edges (for within-network boost)
NETWORK_INDICES = {}
for net in UNIQUE_NETWORKS:
    NETWORK_INDICES[net] = [i for i, n in enumerate(NETWORK_MAP) if n == net]


def roi_index(substring):
    """Find the index of the first ROI whose name contains the substring."""
    for i, name in enumerate(ROI_NAMES):
        if substring in name:
            return i
    raise ValueError(f"No ROI found matching '{substring}'")


# ---------------------------------------------------------------------------
# 2. Topic configuration
# ---------------------------------------------------------------------------

N_SUBJECTS = 200

TOPICS = {
    "pain": {
        "outcome": {"name": "Pain_VAS", "mean": 4.5, "std": 2.0, "min": 0, "max": 10},
        "seed": 42,
        # Real effects: bilateral NAc <-> DefaultA PFCm
        "real_effects": [
            {"roi_a": "NAc-rh", "roi_b": "17Networks_RH_DefaultA_PFCm_1",
             "r": 0.21, "seeds": {"discovery": 341, "validation": 4}},
            {"roi_a": "NAc-lh", "roi_b": "17Networks_LH_DefaultA_PFCm_1",
             "r": 0.21, "seeds": {"discovery": 342, "validation": 5}},
        ],
        "variables": [
            {"name": "Age", "mean": 35, "std": 12, "min": 18, "max": 70, "r": 0.20},
            {"name": "Sex", "type": "binary", "p": 0.52},
            {"name": "BMI", "mean": 26, "std": 5, "min": 15, "max": 45, "r": 0.0},
            {"name": "HRV", "mean": 55, "std": 15, "min": 10, "max": 120, "r": 0.0},
            {"name": "Sleep_Quality", "mean": 7, "std": 3, "min": 0, "max": 21, "r": 0.20},
            {"name": "Physical_Activity", "mean": 150, "std": 60, "min": 0, "max": 400, "r": 0.0},
            {"name": "Caffeine_mg", "mean": 200, "std": 100, "min": 0, "max": 600, "r": 0.0},
            {"name": "Stress_Level", "mean": 16, "std": 7, "min": 0, "max": 40, "r": 0.38},
            {"name": "GAD7", "mean": 7, "std": 4, "min": 0, "max": 21, "r": 0.35},
            {"name": "Catastrophizing", "mean": 22, "std": 10, "min": 0, "max": 52, "r": 0.42},
            {"name": "Fear_Avoidance", "mean": 38, "std": 14, "min": 0, "max": 96, "r": 0.35},
            {"name": "Pain_Duration_mo", "mean": 18, "std": 12, "min": 0, "max": 120, "r": 0.22},
            {"name": "Neuroticism", "mean": 25, "std": 8, "min": 0, "max": 48, "r": 0.20},
            {"name": "Alcohol_drinks", "mean": 5, "std": 4, "min": 0, "max": 25, "r": 0.0},
            {"name": "Screen_Time", "mean": 6, "std": 3, "min": 0, "max": 16, "r": 0.0},
            {"name": "Social_Media", "mean": 90, "std": 45, "min": 0, "max": 300, "r": 0.0},
            {"name": "Education_yrs", "mean": 14, "std": 2.5, "min": 10, "max": 22, "r": 0.0},
            {"name": "Income_k", "mean": 52, "std": 25, "min": 10, "max": 200, "r": 0.0},
        ],
    },
    "depression": {
        "outcome": {"name": "PHQ9", "mean": 8.0, "std": 5.5, "min": 0, "max": 27},
        "seed": 123,
        # Real effects: bilateral SalVentAttnA Insula <-> FrMed
        "real_effects": [
            {"roi_a": "17Networks_LH_SalVentAttnA_Ins_1", "roi_b": "17Networks_RH_SalVentAttnA_FrMed_1",
             "r": 0.21, "seeds": {"discovery": 1018, "validation": 6}},
            {"roi_a": "17Networks_RH_SalVentAttnA_Ins_1", "roi_b": "17Networks_LH_SalVentAttnA_FrMed_1",
             "r": 0.21, "seeds": {"discovery": 1019, "validation": 7}},
        ],
        "variables": [
            {"name": "Age", "mean": 38, "std": 14, "min": 18, "max": 70, "r": 0.20},
            {"name": "Sex", "type": "binary", "p": 0.55},
            {"name": "BMI", "mean": 27, "std": 5, "min": 15, "max": 45, "r": 0.0},
            {"name": "HRV", "mean": 52, "std": 14, "min": 10, "max": 120, "r": 0.0},
            {"name": "Sleep_Quality", "mean": 10, "std": 4, "min": 0, "max": 21, "r": 0.38},
            {"name": "Physical_Activity", "mean": 130, "std": 55, "min": 0, "max": 400, "r": -0.20},
            {"name": "Caffeine_mg", "mean": 220, "std": 110, "min": 0, "max": 600, "r": 0.0},
            {"name": "Stress_Level", "mean": 20, "std": 8, "min": 0, "max": 40, "r": 0.40},
            {"name": "Rumination", "mean": 45, "std": 12, "min": 22, "max": 88, "r": 0.42},
            {"name": "Loneliness", "mean": 42, "std": 12, "min": 20, "max": 80, "r": 0.38},
            {"name": "Social_Support", "mean": 28, "std": 8, "min": 12, "max": 60, "r": -0.22},
            {"name": "Neuroticism", "mean": 28, "std": 9, "min": 0, "max": 48, "r": 0.40},
            {"name": "Self_Esteem", "mean": 18, "std": 6, "min": 0, "max": 30, "r": -0.22},
            {"name": "Alcohol_drinks", "mean": 6, "std": 5, "min": 0, "max": 25, "r": 0.18},
            {"name": "Screen_Time", "mean": 7, "std": 3, "min": 0, "max": 16, "r": 0.18},
            {"name": "Social_Media", "mean": 110, "std": 50, "min": 0, "max": 300, "r": 0.20},
            {"name": "Education_yrs", "mean": 14, "std": 2.5, "min": 10, "max": 22, "r": 0.0},
            {"name": "Income_k", "mean": 48, "std": 22, "min": 10, "max": 200, "r": 0.0},
        ],
    },
    "anxiety": {
        "outcome": {"name": "GAD7", "mean": 7.0, "std": 4.5, "min": 0, "max": 21},
        "seed": 7,
        # Real effect: AMY-rh <-> SalVentAttnA FrMed LH
        "real_effects": [
            {"roi_a": "AMY-rh", "roi_b": "17Networks_LH_SalVentAttnA_FrMed_1",
             "r": 0.21, "seeds": {"discovery": 2022, "validation": 7}},
        ],
        "variables": [
            {"name": "Age", "mean": 32, "std": 11, "min": 18, "max": 70, "r": 0.20},
            {"name": "Sex", "type": "binary", "p": 0.58},
            {"name": "BMI", "mean": 25, "std": 4.5, "min": 15, "max": 45, "r": 0.0},
            {"name": "HRV", "mean": 58, "std": 16, "min": 10, "max": 120, "r": 0.0},
            {"name": "Sleep_Quality", "mean": 8, "std": 3, "min": 0, "max": 21, "r": 0.22},
            {"name": "Physical_Activity", "mean": 160, "std": 65, "min": 0, "max": 400, "r": 0.0},
            {"name": "Caffeine_mg", "mean": 190, "std": 90, "min": 0, "max": 600, "r": 0.18},
            {"name": "Stress_Level", "mean": 22, "std": 8, "min": 0, "max": 40, "r": 0.40},
            {"name": "Social_Support", "mean": 30, "std": 10, "min": 12, "max": 60, "r": -0.22},
            {"name": "Worry", "mean": 48, "std": 14, "min": 16, "max": 80, "r": 0.42},
            {"name": "Intolerance_Uncertainty", "mean": 65, "std": 18, "min": 27, "max": 135, "r": 0.38},
            {"name": "Neuroticism", "mean": 30, "std": 9, "min": 0, "max": 48, "r": 0.40},
            {"name": "Alcohol_drinks", "mean": 4, "std": 3, "min": 0, "max": 25, "r": 0.0},
            {"name": "Screen_Time", "mean": 5.5, "std": 2.8, "min": 0, "max": 16, "r": 0.0},
            {"name": "Social_Media", "mean": 100, "std": 50, "min": 0, "max": 300, "r": 0.20},
            {"name": "Education_yrs", "mean": 15, "std": 2.5, "min": 10, "max": 22, "r": 0.0},
            {"name": "Income_k", "mean": 55, "std": 28, "min": 10, "max": 200, "r": 0.0},
            {"name": "Self_Esteem", "mean": 16, "std": 6, "min": 0, "max": 30, "r": -0.20},
        ],
    },
}


# ---------------------------------------------------------------------------
# 3. Helper: plant a correlation between an edge and a behavioral variable
# ---------------------------------------------------------------------------

def plant_correlation(fc_tensor, behavior, idx_a, idx_b, target_r, seed):
    """
    Adjust one edge so it correlates with behavior at approximately target_r.
    Uses the formula: edge = target_r * z_behavior + sqrt(1 - target_r^2) * z_noise
    Uses a dedicated seed so the result is reproducible regardless of prior RNG state.
    """
    plant_rng = np.random.default_rng(seed)
    n = len(behavior)
    z_beh = (behavior - behavior.mean()) / behavior.std()
    z_noise = plant_rng.standard_normal(n)
    raw = target_r * z_beh + np.sqrt(1 - target_r ** 2) * z_noise

    # Scale to realistic FC range
    raw = raw * 0.25  # SD ~0.25
    # Clip to [-1, 1]
    raw = np.clip(raw, -1, 1)

    fc_tensor[idx_a, idx_b, :] = raw
    fc_tensor[idx_b, idx_a, :] = raw


# ---------------------------------------------------------------------------
# 4. Generate one dataset
# ---------------------------------------------------------------------------

def generate_dataset(topic_name, split, rng):
    """Generate a single discovery or validation dataset."""
    config = TOPICS[topic_name]

    print(f"  Generating {topic_name} {split}...")

    # --- Base FC tensor ---
    fc = rng.standard_normal((N_ROIS, N_ROIS, N_SUBJECTS)).astype(np.float32)

    # Make symmetric
    for s in range(N_SUBJECTS):
        mat = fc[:, :, s]
        fc[:, :, s] = (mat + mat.T) / 2.0

    # Diagonal = 1
    for s in range(N_SUBJECTS):
        np.fill_diagonal(fc[:, :, s], 1.0)

    # --- Within-network structure: heterogeneous boost ---
    # Nearby ROIs within the same network get a stronger boost (~0.20),
    # distant ones get less (~0.08). This creates natural-looking variation.
    for net, indices in NETWORK_INDICES.items():
        n_net = len(indices)
        if n_net < 2:
            continue
        for i_pos, i in enumerate(indices):
            for j_pos, j in enumerate(indices[i_pos + 1:], start=i_pos + 1):
                # Distance-weighted: nearby ROIs (small index gap) get stronger boost
                distance_frac = abs(i_pos - j_pos) / max(n_net - 1, 1)
                boost = 0.20 - 0.12 * distance_frac  # ranges 0.20 -> 0.08
                fc[i, j, :] += boost
                fc[j, i, :] += boost

    # --- Extra boost for sensory/motor networks (strong local organization) ---
    sensory_nets = {"VisCent", "VisPeri", "SomMotA", "SomMotB"}
    for net in sensory_nets:
        indices = NETWORK_INDICES.get(net, [])
        if len(indices) < 2:
            continue
        for i_pos, i in enumerate(indices):
            for j in indices[i_pos + 1:]:
                fc[i, j, :] += 0.08
                fc[j, i, :] += 0.08

    # --- Between-network structure ---
    # Task-positive <-> Default Mode anti-correlations
    task_positive = {"DorsAttnA", "DorsAttnB", "SalVentAttnA", "SalVentAttnB"}
    default_mode = {"DefaultA", "DefaultB", "DefaultC", "TempPar"}
    for tp_net in task_positive:
        for dmn_net in default_mode:
            tp_idx = NETWORK_INDICES.get(tp_net, [])
            dm_idx = NETWORK_INDICES.get(dmn_net, [])
            for i in tp_idx:
                for j in dm_idx:
                    fc[i, j, :] -= 0.08
                    fc[j, i, :] -= 0.08

    # Weak positive coupling between related sub-networks
    related_pairs = [
        ("SalVentAttnA", "SalVentAttnB"),
        ("DefaultA", "DefaultB"),
        ("DefaultB", "DefaultC"),
        ("ContA", "ContB"),
        ("ContB", "ContC"),
        ("DorsAttnA", "DorsAttnB"),
        ("SomMotA", "SomMotB"),
        ("VisCent", "VisPeri"),
    ]
    for net_x, net_y in related_pairs:
        idx_x = NETWORK_INDICES.get(net_x, [])
        idx_y = NETWORK_INDICES.get(net_y, [])
        for i in idx_x:
            for j in idx_y:
                fc[i, j, :] += 0.05
                fc[j, i, :] += 0.05

    # --- Scale non-diagonal to have roughly SD = 0.30 ---
    for s in range(N_SUBJECTS):
        mat = fc[:, :, s]
        mask = ~np.eye(N_ROIS, dtype=bool)
        vals = mat[mask]
        current_std = vals.std()
        if current_std > 0:
            mat[mask] = vals / current_std * 0.30
        fc[:, :, s] = mat
        np.fill_diagonal(fc[:, :, s], 1.0)

    # --- Behavioral outcome (clinical scale) ---
    out_cfg = config["outcome"]
    outcome = np.clip(
        rng.normal(out_cfg["mean"], out_cfg["std"], N_SUBJECTS),
        out_cfg["min"], out_cfg["max"]
    )

    # --- Clip FC to [-1, 1] ---
    for s in range(N_SUBJECTS):
        np.fill_diagonal(fc[:, :, s], 0)  # zero diagonal before clipping
    fc = np.clip(fc, -1, 1)
    for s in range(N_SUBJECTS):
        np.fill_diagonal(fc[:, :, s], 1.0)

    # --- Behavioral / demographic variables with tiered correlations ---
    behavior_columns = [out_cfg["name"]]
    behavior_data = [outcome]

    z_outcome = (outcome - outcome.mean()) / (outcome.std() + 1e-12)

    for var in config["variables"]:
        if var.get("type") == "binary":
            vals = rng.binomial(1, var["p"], N_SUBJECTS).astype(float)
        else:
            target_r = var.get("r", 0.0)
            z_noise = rng.standard_normal(N_SUBJECTS)
            z_var = target_r * z_outcome + np.sqrt(1 - target_r ** 2) * z_noise
            vals = var["mean"] + var["std"] * z_var
            if "min" in var and "max" in var:
                vals = np.clip(vals, var["min"], var["max"])

        behavior_columns.append(var["name"])
        behavior_data.append(vals)

    behavior_data = np.column_stack(behavior_data).astype(np.float32)

    # --- Post-process: inter-variable correlations (mediation + moderation) ---
    # Build a column lookup for easy access
    col_idx = {name: i for i, name in enumerate(behavior_columns)}

    # Mediation: Screen_Time drives both Social_Media and Sleep_Quality
    # Rebuild these variables as: z_var = w_screen * z_screen + w_outcome * z_out + w_noise * z_noise
    if "Screen_Time" in col_idx and "Social_Media" in col_idx:
        z_screen = (behavior_data[:, col_idx["Screen_Time"]]
                    - behavior_data[:, col_idx["Screen_Time"]].mean())
        z_screen = z_screen / (z_screen.std() + 1e-12)

        # Social_Media: driven by Screen_Time (r~0.50) + outcome (keep existing r) + noise
        sm_cfg = next(v for v in config["variables"] if v["name"] == "Social_Media")
        w_screen_sm = 0.50
        w_outcome_sm = sm_cfg.get("r", 0.0) * 0.6  # reduced weight since Screen_Time adds some
        w_noise_sm = np.sqrt(max(1 - w_screen_sm**2 - w_outcome_sm**2, 0.01))
        z_sm = (w_screen_sm * z_screen + w_outcome_sm * z_outcome
                + w_noise_sm * rng.standard_normal(N_SUBJECTS))
        sm_vals = sm_cfg["mean"] + sm_cfg["std"] * z_sm
        sm_vals = np.clip(sm_vals, sm_cfg["min"], sm_cfg["max"])
        behavior_data[:, col_idx["Social_Media"]] = sm_vals

    if "Screen_Time" in col_idx and "Sleep_Quality" in col_idx:
        z_screen = (behavior_data[:, col_idx["Screen_Time"]]
                    - behavior_data[:, col_idx["Screen_Time"]].mean())
        z_screen = z_screen / (z_screen.std() + 1e-12)

        # Sleep_Quality: driven by Screen_Time (r~0.40) + outcome (keep existing r) + noise
        sq_cfg = next(v for v in config["variables"] if v["name"] == "Sleep_Quality")
        w_screen_sq = 0.40
        w_outcome_sq = sq_cfg.get("r", 0.0) * 0.7
        w_noise_sq = np.sqrt(max(1 - w_screen_sq**2 - w_outcome_sq**2, 0.01))
        z_sq = (w_screen_sq * z_screen + w_outcome_sq * z_outcome
                + w_noise_sq * rng.standard_normal(N_SUBJECTS))
        sq_vals = sq_cfg["mean"] + sq_cfg["std"] * z_sq
        sq_vals = np.clip(sq_vals, sq_cfg["min"], sq_cfg["max"])
        behavior_data[:, col_idx["Sleep_Quality"]] = sq_vals

    # Moderation: Social_Media <-> Stress_Level differs by Sex
    # For females (Sex == 0), add extra Stress-correlated component to Social_Media
    if all(k in col_idx for k in ["Social_Media", "Stress_Level", "Sex"]):
        sex = behavior_data[:, col_idx["Sex"]]
        female = sex == 0
        if female.sum() > 10:
            z_stress = (behavior_data[:, col_idx["Stress_Level"]]
                        - behavior_data[:, col_idx["Stress_Level"]].mean())
            z_stress = z_stress / (z_stress.std() + 1e-12)

            sm_cfg = next(v for v in config["variables"] if v["name"] == "Social_Media")
            # Add stress component for females only
            sm_vals = behavior_data[:, col_idx["Social_Media"]].copy()
            sm_z = (sm_vals - sm_vals.mean()) / (sm_vals.std() + 1e-12)
            extra_weight = 0.30
            sm_z[female] = sm_z[female] + extra_weight * z_stress[female]
            # Rescale to preserve marginal distribution
            sm_z = (sm_z - sm_z.mean()) / (sm_z.std() + 1e-12)
            sm_vals = sm_cfg["mean"] + sm_cfg["std"] * sm_z
            sm_vals = np.clip(sm_vals, sm_cfg["min"], sm_cfg["max"])
            behavior_data[:, col_idx["Social_Media"]] = sm_vals

    behavior_data = behavior_data.astype(np.float32)

    # --- Age-related decline in within-network FC (DMN, Salience, Frontoparietal) ---
    if "Age" in col_idx:
        age_vals = behavior_data[:, col_idx["Age"]]
        z_age = (age_vals - age_vals.mean()) / (age_vals.std() + 1e-12)

        aging_networks_17 = {
            "DefaultA", "DefaultB", "DefaultC",
            "SalVentAttnA", "SalVentAttnB",
            "ContA", "ContB", "ContC",
        }
        decline_strength = 0.08  # r ~ -0.25 between age and within-network FC

        for net_name in aging_networks_17:
            indices = NETWORK_INDICES.get(net_name, [])
            if len(indices) < 2:
                continue
            for i_pos, i in enumerate(indices):
                for j in indices[i_pos + 1:]:
                    # Older subjects get lower within-network FC
                    fc[i, j, :] -= decline_strength * z_age
                    fc[j, i, :] -= decline_strength * z_age

        # Re-clip to [-1, 1] and restore diagonal
        fc = np.clip(fc, -1, 1)
        for s in range(N_SUBJECTS):
            np.fill_diagonal(fc[:, :, s], 1.0)

    # --- Plant real effects (both discovery and validation) ---
    # Planted AFTER age-related decline so the effect isn't diluted
    # for within-network edges (e.g., depression's SalVentAttnA effect).
    for effect in config["real_effects"]:
        real_a = roi_index(effect["roi_a"])
        real_b = roi_index(effect["roi_b"])
        real_plant_seed = effect["seeds"][split]
        plant_correlation(fc, outcome, real_a, real_b, effect["r"], real_plant_seed)
        print(f"    Real effect: {ROI_NAMES[real_a]} <-> {ROI_NAMES[real_b]} (target r={effect['r']})")

    # --- Confound edge: AMY-Insula correlated with Stress_Level ---
    # The amygdala-insula circuit tracks stress. Since Stress_Level correlates
    # with the outcome (~0.40), this edge will APPEAR to predict the outcome,
    # but the relationship vanishes when controlling for Stress_Level.
    if "Stress_Level" in col_idx:
        stress_vals = behavior_data[:, col_idx["Stress_Level"]]
        confound_a = roi_index("AMY-rh")
        confound_b = roi_index("17Networks_RH_SalVentAttnA_Ins_1")
        confound_seed = config["seed"] * 100 + 77 if split == "discovery" else config["seed"] * 100 + 78
        plant_correlation(fc, stress_vals, confound_a, confound_b, 0.30, confound_seed)
        print(f"    Confound edge: {ROI_NAMES[confound_a]} <-> {ROI_NAMES[confound_b]} "
              f"(target r=0.30 with Stress_Level)")

    # --- Extract upper triangle ---
    triu_row, triu_col = np.triu_indices(N_ROIS, k=1)
    n_edges = len(triu_row)
    fc_upper = np.empty((N_SUBJECTS, n_edges), dtype=np.float32)
    for s in range(N_SUBJECTS):
        fc_upper[s, :] = fc[triu_row, triu_col, s]

    return {
        "fc_upper": fc_upper,
        "triu_row": triu_row.astype(np.int32),
        "triu_col": triu_col.astype(np.int32),
        "behavior_columns": np.array(behavior_columns),
        "behavior_data": behavior_data,
        "roi_names": np.array(ROI_NAMES),
        "n_rois": np.int32(N_ROIS),
    }


# ---------------------------------------------------------------------------
# 5. Main: generate all 6 datasets
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Atlas: {N_ROIS} ROIs, {N_ROIS * (N_ROIS - 1) // 2} edges")
    print(f"Networks: {UNIQUE_NETWORKS}")
    print()

    for topic_name, config in TOPICS.items():
        base_seed = config["seed"]
        print(f"Topic: {topic_name} (seed={base_seed})")

        # Discovery: use base_seed
        rng_disc = np.random.default_rng(base_seed)
        disc_data = generate_dataset(topic_name, "discovery", rng_disc)
        disc_path = os.path.join(DATA_DIR, f"{topic_name}_discovery.npz")
        np.savez_compressed(disc_path, **disc_data)
        size_mb = os.path.getsize(disc_path) / 1e6
        print(f"    Saved {disc_path} ({size_mb:.1f} MB)")

        # Validation: use base_seed + 1000
        rng_val = np.random.default_rng(base_seed + 1000)
        val_data = generate_dataset(topic_name, "validation", rng_val)
        val_path = os.path.join(DATA_DIR, f"{topic_name}_validation.npz")
        np.savez_compressed(val_path, **val_data)
        size_mb = os.path.getsize(val_path) / 1e6
        print(f"    Saved {val_path} ({size_mb:.1f} MB)")
        print()

    # --- Quick verification ---
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    from scipy.stats import pearsonr

    for topic_name, config in TOPICS.items():
        print(f"\n{topic_name.upper()}:")

        for split in ["discovery", "validation"]:
            fpath = os.path.join(DATA_DIR, f"{topic_name}_{split}.npz")
            d = np.load(fpath, allow_pickle=True)
            fc_upper = d["fc_upper"]
            triu_row = d["triu_row"]
            triu_col = d["triu_col"]
            beh_cols = list(d["behavior_columns"])
            beh_data = d["behavior_data"]
            roi_names = list(d["roi_names"])
            outcome = beh_data[:, 0]

            # Check real effect edges
            for effect in config["real_effects"]:
                real_a = roi_names.index(
                    next(n for n in roi_names if effect["roi_a"] in n)
                )
                real_b = roi_names.index(
                    next(n for n in roi_names if effect["roi_b"] in n)
                )
                # Find edge index in upper triangle
                if real_a > real_b:
                    real_a, real_b = real_b, real_a
                edge_mask = (triu_row == real_a) & (triu_col == real_b)
                edge_idx = np.where(edge_mask)[0][0]
                edge_vals = fc_upper[:, edge_idx]
                r, p = pearsonr(edge_vals, outcome)
                print(f"  {split}: {effect['roi_a']} <-> {effect['roi_b']}: r={r:.3f}, p={p:.2e}")

        # Count false positives in discovery
        fpath = os.path.join(DATA_DIR, f"{topic_name}_discovery.npz")
        d = np.load(fpath, allow_pickle=True)
        fc_upper = d["fc_upper"]
        beh_data = d["behavior_data"]
        beh_cols = list(d["behavior_columns"])
        outcome = beh_data[:, 0]
        n_edges = fc_upper.shape[1]
        n_sig = 0
        for e in range(n_edges):
            _, p = pearsonr(fc_upper[:, e], outcome)
            if p < 0.05:
                n_sig += 1
        expected = n_edges * 0.05
        print(f"  discovery: {n_sig} significant at p<0.05 (expected ~{expected:.0f} of {n_edges})")

        # Check behavioral intercorrelations with outcome
        print(f"  Outcome: {beh_cols[0]}")
        print(f"  Behavioral correlations with outcome:")
        for i, col in enumerate(beh_cols[1:], 1):
            r_beh, _ = pearsonr(beh_data[:, 0], beh_data[:, i])
            print(f"    {col}: r={r_beh:.3f}")

        # Check inter-variable correlations (mediation + moderation)
        col_idx = {name: i for i, name in enumerate(beh_cols)}
        if all(k in col_idx for k in ["Screen_Time", "Social_Media", "Sleep_Quality"]):
            print(f"\n  Inter-variable correlations (mediation check):")
            r_st_sm, _ = pearsonr(beh_data[:, col_idx["Screen_Time"]],
                                   beh_data[:, col_idx["Social_Media"]])
            r_st_sq, _ = pearsonr(beh_data[:, col_idx["Screen_Time"]],
                                   beh_data[:, col_idx["Sleep_Quality"]])
            r_sm_sq, _ = pearsonr(beh_data[:, col_idx["Social_Media"]],
                                   beh_data[:, col_idx["Sleep_Quality"]])
            print(f"    Screen_Time <-> Social_Media: r={r_st_sm:.3f} (target ~0.50)")
            print(f"    Screen_Time <-> Sleep_Quality: r={r_st_sq:.3f} (target ~0.40)")
            print(f"    Social_Media <-> Sleep_Quality: r={r_sm_sq:.3f} (target ~0.25)")

        if all(k in col_idx for k in ["Social_Media", "Stress_Level", "Sex"]):
            print(f"  Moderation check (Social_Media <-> Stress_Level by Sex):")
            sex = beh_data[:, col_idx["Sex"]]
            sm = beh_data[:, col_idx["Social_Media"]]
            stress = beh_data[:, col_idx["Stress_Level"]]
            female = sex == 0
            male = sex == 1
            r_f, _ = pearsonr(sm[female], stress[female])
            r_m, _ = pearsonr(sm[male], stress[male])
            r_all, _ = pearsonr(sm, stress)
            print(f"    Females: r={r_f:.3f} (target ~0.30-0.35)")
            print(f"    Males: r={r_m:.3f} (target ~0.05-0.10)")
            print(f"    Overall: r={r_all:.3f} (target ~0.20)")

        # Check age-FC decline in within-network edges
        if "Age" in col_idx:
            print(f"\n  Age-FC decline check (within-network edges):")
            age = beh_data[:, col_idx["Age"]]
            aging_nets = ["DefaultA", "DefaultB", "DefaultC",
                          "SalVentAttnA", "SalVentAttnB",
                          "ContA", "ContB", "ContC"]
            roi_names = list(d["roi_names"])
            net_map = []
            subcort = {"HIP", "AMY", "pTHA", "aTHA", "NAc", "GP", "PUT", "CAU"}
            for name in roi_names:
                prefix = name.split("-")[0]
                if prefix in subcort:
                    net_map.append("Subcortical")
                else:
                    parts = name.split("_")
                    net_map.append(parts[2] if len(parts) >= 3 else "Unknown")
            for net_name in aging_nets:
                net_idx = [i for i, n in enumerate(net_map) if n == net_name]
                if len(net_idx) < 2:
                    continue
                # Sample a few within-network edges and compute age-FC correlation
                r_vals = []
                count = 0
                for ii in range(len(net_idx)):
                    for jj in range(ii + 1, len(net_idx)):
                        a, b = net_idx[ii], net_idx[jj]
                        if a > b:
                            a, b = b, a
                        emask = (triu_row == a) & (triu_col == b)
                        eidx = np.where(emask)[0]
                        if len(eidx) > 0:
                            r_age, _ = pearsonr(fc_upper[:, eidx[0]], age)
                            r_vals.append(r_age)
                            count += 1
                        if count >= 10:
                            break
                    if count >= 10:
                        break
                if r_vals:
                    mean_r = np.mean(r_vals)
                    print(f"    {net_name}: mean r(Age,FC) = {mean_r:.3f} "
                          f"(n={len(r_vals)} edges sampled, target ~ -0.20 to -0.30)")

        # Check confound edge (AMY-Insula vs Stress and vs Outcome)
        if "Stress_Level" in col_idx:
            print(f"\n  Confound edge check (AMY-rh <-> RH_SalVentAttnA_Ins_1):")
            confound_a = roi_names.index(
                next(n for n in roi_names if "AMY-rh" in n))
            confound_b = roi_names.index(
                next(n for n in roi_names if "17Networks_RH_SalVentAttnA_Ins_1" in n))
            if confound_a > confound_b:
                confound_a, confound_b = confound_b, confound_a
            emask = (triu_row == confound_a) & (triu_col == confound_b)
            eidx = np.where(emask)[0][0]
            edge_vals = fc_upper[:, eidx]
            stress = beh_data[:, col_idx["Stress_Level"]]
            r_stress, _ = pearsonr(edge_vals, stress)
            r_out, _ = pearsonr(edge_vals, outcome)
            # Partial correlation: edge-outcome controlling for Stress
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            s2d = stress.reshape(-1, 1)
            e_resid = edge_vals - lr.fit(s2d, edge_vals).predict(s2d)
            o_resid = outcome - lr.fit(s2d, outcome).predict(s2d)
            r_partial, _ = pearsonr(e_resid, o_resid)
            print(f"    Edge <-> Stress_Level: r={r_stress:.3f} (target ~0.30)")
            print(f"    Edge <-> Outcome:      r={r_out:.3f} (indirect, should be ~0.12)")
            print(f"    Edge <-> Outcome | Stress: r={r_partial:.3f} (should drop to ~0)")

    print("\nDone!")


if __name__ == "__main__":
    main()
