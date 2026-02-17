"""
lab_helpers.py
Student-facing helper functions for the Reproducibility Lab.
Students interact with functional connectivity data through these functions
instead of touching the raw tensor directly.

Usage (in notebook):
    import lab_helpers as helpers
    helpers.load_dataset('pain', 'discovery')
    helpers.list_networks()
    helpers.plot_edge('NAc-rh', 'RH_DefaultA_PFCm_1', 'Pain_VAS')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr, t as t_dist
import os
import warnings

# ============================================================================
# Module-level state — loaded once, used by all functions
# ============================================================================

_fc_tensor = None       # (n_rois, n_rois, n_subjects) reconstructed matrix
_behavior = None        # pandas DataFrame
_roi_names = None       # list of str
_network_map = None     # list of str (one per ROI)
_network_order = None   # list of str (unique network names, ordered)
_roi_coords = None      # (n_rois, 3) MNI coordinates
_n_rois = None
_topic = None
_split = None

SUBCORTICAL_PREFIXES = {"HIP", "AMY", "pTHA", "aTHA", "NAc", "GP", "PUT", "CAU"}

# 17-network to 7-network mapping (same ROIs, just grouped differently)
_NET17_TO_NET7 = {
    "VisCent": "Visual",
    "VisPeri": "Visual",
    "SomMotA": "Somatomotor",
    "SomMotB": "Somatomotor",
    "DorsAttnA": "Dorsal Attention",
    "DorsAttnB": "Dorsal Attention",
    "SalVentAttnA": "Salience",
    "SalVentAttnB": "Salience",
    "LimbicA": "Limbic",
    "LimbicB": "Limbic",
    "ContA": "Frontoparietal",
    "ContB": "Frontoparietal",
    "ContC": "Frontoparietal",
    "DefaultA": "Default Mode",
    "DefaultB": "Default Mode",
    "DefaultC": "Default Mode",
    "TempPar": "Default Mode",
}

# Network display colors (named colors only, no hex) — 7 networks + Subcortical
NETWORK_COLORS = {
    "Subcortical": "slategray",
    "Visual": "purple",
    "Somatomotor": "steelblue",
    "Dorsal Attention": "olivedrab",
    "Salience": "mediumvioletred",
    "Limbic": "goldenrod",
    "Frontoparietal": "darkorange",
    "Default Mode": "indianred",
}


def _residualize(X, y):
    """Remove linear influence of covariates X from variable y using OLS."""
    n = len(y)
    X_int = np.column_stack([X, np.ones(n)])
    beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
    return y - X_int @ beta


def _apply_correction(results_df, method):
    """Apply multiple comparisons correction to a results DataFrame.

    Parameters
    ----------
    results_df : DataFrame
        Must contain a 'p' column.
    method : str
        'fdr' for Benjamini-Hochberg, 'bonferroni' for Bonferroni.

    Returns
    -------
    DataFrame with added 'p_corrected' column.
    """
    from statsmodels.stats.multitest import multipletests

    df = results_df.copy()
    p_vals = df["p"].values
    n_tests = len(p_vals)

    if method == 'fdr':
        _, p_corr, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
        df["p_corrected"] = p_corr
    elif method == 'bonferroni':
        df["p_corrected"] = np.minimum(p_vals * n_tests, 1.0)
    else:
        raise ValueError(f"Unknown correction method: '{method}'. Use 'fdr' or 'bonferroni'.")

    return df


def _get_network_17(label):
    """Return the 17-network name for a given ROI label."""
    # Strip legacy prefix if still present
    label = label.replace("17Networks_", "")
    prefix = label.split("-")[0]
    if prefix in SUBCORTICAL_PREFIXES:
        return "Subcortical"
    # Format: LH_DefaultA_PFCm_1 -> parts[1] = DefaultA
    parts = label.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "Unknown"


def _get_network(label):
    """Return the 7-network name for a given ROI label."""
    net17 = _get_network_17(label)
    return _NET17_TO_NET7.get(net17, net17)  # Subcortical passes through


# ============================================================================
# Colab auto-download
# ============================================================================

_BASE_URL = "https://raw.githubusercontent.com/cmahlen/python-stats-demo/main/"


def _ensure_files(topic):
    """Download data files if running in Colab or if files are missing."""
    files_needed = [
        f"data/{topic}_discovery.npz",
        f"data/{topic}_validation.npz",
        "data/roi_mni_coords.npy",
        "atlas_labels.txt",
        "lab_helpers.py",
    ]
    # Check if we need to download
    need_download = False
    for f in files_needed:
        if not os.path.exists(f):
            need_download = True
            break

    if need_download:
        import urllib.request
        os.makedirs("data", exist_ok=True)
        for f in files_needed:
            if not os.path.exists(f):
                url = _BASE_URL + f
                print(f"  Downloading {f}...")
                try:
                    urllib.request.urlretrieve(url, f)
                except Exception as e:
                    print(f"  Warning: could not download {f}: {e}")


# ============================================================================
# Core: load_dataset
# ============================================================================

def load_dataset(topic, split="discovery"):
    """
    Load a functional connectivity dataset.

    Parameters
    ----------
    topic : str
        One of 'pain', 'depression', or 'anxiety'.
    split : str
        Either 'discovery' or 'validation'.

    Returns
    -------
    None (data is stored internally and accessed through helper functions).
    """
    global _fc_tensor, _behavior, _roi_names, _network_map, _network_order
    global _roi_coords, _n_rois, _topic, _split

    topic = topic.lower().strip()
    split = split.lower().strip()
    valid_topics = ["pain", "depression", "anxiety"]
    valid_splits = ["discovery", "validation"]
    if topic not in valid_topics:
        print(f"Error: topic must be one of {valid_topics}, got '{topic}'")
        return
    if split not in valid_splits:
        print(f"Error: split must be one of {valid_splits}, got '{split}'")
        return

    _ensure_files(topic)

    fpath = f"data/{topic}_{split}.npz"
    if not os.path.exists(fpath):
        print(f"Error: file not found: {fpath}")
        return

    d = np.load(fpath, allow_pickle=True)
    fc_upper = d["fc_upper"]        # (n_subjects, n_edges)
    triu_row = d["triu_row"]
    triu_col = d["triu_col"]
    beh_cols = list(d["behavior_columns"])
    beh_data = d["behavior_data"]
    roi_names = [r.replace("17Networks_", "") for r in d["roi_names"]]
    n_rois = int(d["n_rois"])

    # Reconstruct full tensor
    n_subjects = fc_upper.shape[0]
    tensor = np.zeros((n_rois, n_rois, n_subjects), dtype=np.float32)
    for s in range(n_subjects):
        tensor[triu_row, triu_col, s] = fc_upper[s, :]
        tensor[triu_col, triu_row, s] = fc_upper[s, :]
        np.fill_diagonal(tensor[:, :, s], 1.0)

    # Build network map
    network_map = [_get_network(name) for name in roi_names]

    # Canonical network order for display (7 networks)
    canonical_order = [
        "Subcortical", "Visual", "Somatomotor", "Dorsal Attention",
        "Salience", "Limbic", "Frontoparietal", "Default Mode",
    ]
    present = set(network_map)
    network_order = [n for n in canonical_order if n in present]

    # Load MNI coordinates for ROIs (used by plot_glass_brain)
    coords_path = "data/roi_mni_coords.npy"
    if os.path.exists(coords_path):
        roi_coords = np.load(coords_path)
    else:
        roi_coords = None

    _fc_tensor = tensor
    _behavior = pd.DataFrame(beh_data, columns=beh_cols)
    _roi_names = roi_names
    _network_map = network_map
    _network_order = network_order
    _roi_coords = roi_coords
    _n_rois = n_rois
    _topic = topic
    _split = split

    outcome_col = beh_cols[0]
    print(f"Loaded {topic} {split} dataset:")
    print(f"  {n_subjects} subjects")
    print(f"  {n_rois} brain regions (ROIs)")
    print(f"  {fc_upper.shape[1]:,} connectivity edges")
    print(f"  Outcome variable: {outcome_col}")
    print(f"  Other variables: {', '.join(beh_cols[1:])}")


def _check_loaded():
    """Raise an error if no dataset is loaded."""
    if _fc_tensor is None:
        print("Error: No dataset loaded. Run load_dataset() first.")
        print("  Example: load_dataset('pain', 'discovery')")
        return False
    return True


# ============================================================================
# ROI resolution (fuzzy matching)
# ============================================================================

def _resolve_roi(query):
    """
    Resolve a ROI name from a user query.
    Tries: exact match -> case-insensitive -> substring -> suggestions.
    Returns the index and full name, or (None, None) with a helpful message.
    """
    if not _check_loaded():
        return None, None

    query = query.strip()

    # Exact match
    if query in _roi_names:
        return _roi_names.index(query), query

    # Case-insensitive exact match
    lower_names = [n.lower() for n in _roi_names]
    if query.lower() in lower_names:
        idx = lower_names.index(query.lower())
        return idx, _roi_names[idx]

    # Substring match (case-insensitive)
    matches = [(i, n) for i, n in enumerate(_roi_names)
               if query.lower() in n.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Multiple ROIs match '{query}':")
        for i, (idx, name) in enumerate(matches[:10]):
            print(f"  {name}")
        if len(matches) > 10:
            print(f"  ... and {len(matches) - 10} more")
        print("Please be more specific.")
        return None, None

    # No match — suggest closest
    from difflib import get_close_matches
    suggestions = get_close_matches(query, _roi_names, n=5, cutoff=0.3)
    print(f"No ROI found matching '{query}'.")
    if suggestions:
        print("Did you mean one of these?")
        for s in suggestions:
            print(f"  {s}")
    return None, None


# ============================================================================
# Network resolution (accepts both 7-network and 17-network names)
# ============================================================================

def _resolve_network(user_input):
    """
    Resolve a network name from user input.

    Accepts:
    - 7-network names (e.g., 'Salience', 'Default Mode', 'visual')
    - 17-network sub-names (e.g., 'SalVentAttnA', 'DefaultA', 'ContB')
    - Fuzzy/substring matching for both

    Returns (mode, value):
    - ('net7', name): filter by _network_map[i] == name
    - ('net17', name): filter by 17-network name in ROI label
    - (None, None): no match found
    """
    key = user_input.lower().strip()

    # 1. Exact match against 7-network names
    net7_lower = {n.lower(): n for n in _network_order}
    if key in net7_lower:
        return ('net7', net7_lower[key])

    # 2. Fuzzy match against 7-network names
    matches_7 = [n for n in _network_order if key in n.lower()]
    if len(matches_7) == 1:
        return ('net7', matches_7[0])

    # 3. Check 17-network sub-names (for backward compatibility)
    net17_names = sorted(set(_NET17_TO_NET7.keys()))
    net17_lower = {n.lower(): n for n in net17_names}
    if key in net17_lower:
        return ('net17', net17_lower[key])
    matches_17 = [n for n in net17_names if key in n.lower()]
    if len(matches_17) == 1:
        return ('net17', matches_17[0])
    if len(matches_17) > 1:
        print(f"Multiple sub-networks match '{user_input}': {matches_17}")
        return (None, None)

    # 4. No match
    print(f"No network matching '{user_input}'.")
    print(f"Available networks: {_network_order}")
    print(f"Sub-networks (from ROI names): {net17_names}")
    return (None, None)


def _roi_matches_network(i, mode, value):
    """Check if ROI at index i matches the resolved network filter."""
    if mode == 'net7':
        return _network_map[i] == value
    elif mode == 'net17':
        return value in _roi_names[i]
    return False


# ============================================================================
# list_networks / list_regions
# ============================================================================

def list_networks():
    """Print all brain networks with their ROI counts."""
    if not _check_loaded():
        return

    print(f"Brain networks in the {_topic} dataset:")
    print(f"{'Network':<20} {'ROIs':>6}")
    print("-" * 28)
    for net in _network_order:
        count = sum(1 for n in _network_map if n == net)
        print(f"{net:<20} {count:>6}")
    total = len(_roi_names)
    print("-" * 28)
    print(f"{'Total':<20} {total:>6}")


def list_regions(network=None):
    """
    Print ROI names, optionally filtered by network.

    Parameters
    ----------
    network : str, optional
        Network name (e.g., 'Salience', 'Default Mode', 'Subcortical').
        Also accepts 17-network sub-names (e.g., 'SalVentAttnA', 'DefaultA').
        If None, prints all ROIs.
    """
    if not _check_loaded():
        return

    mode, value = None, None
    if network is not None:
        mode, value = _resolve_network(network)
        if mode is None:
            return

    if network:
        print(f"ROIs in {value}:")
    else:
        print("All ROIs:")

    count = 0
    for i, name in enumerate(_roi_names):
        if network is None or _roi_matches_network(i, mode, value):
            net = _network_map[i]
            print(f"  [{i:3d}] {name:<50} ({net})")
            count += 1

    print(f"\n{count} regions listed.")


def describe_regions(network=None):
    """
    Print ROI names with decoded descriptions explaining each part of the name.

    Parameters
    ----------
    network : str, optional
        Network name to filter by (e.g., 'Subcortical', 'Salience', 'DefaultA').
        If None, shows all regions.
    """
    if not _check_loaded():
        return

    # Resolve network if provided
    mode, value = None, None
    if network is not None:
        mode, value = _resolve_network(network)
        if mode is None:
            return

    # Subregion descriptions for common abbreviations
    _subregion_descriptions = {
        # Subcortical
        "HIP": "Hippocampus", "AMY": "Amygdala", "pTHA": "Posterior thalamus",
        "aTHA": "Anterior thalamus", "NAc": "Nucleus accumbens",
        "GP": "Globus pallidus", "PUT": "Putamen", "CAU": "Caudate",
        # Cortical subregions
        "PFCm": "Medial prefrontal cortex", "PFCdPFCm": "Dorsomedial prefrontal",
        "PFCl": "Lateral prefrontal cortex", "PFCd": "Dorsal prefrontal cortex",
        "PFCv": "Ventral prefrontal cortex", "PFCmp": "Medial posterior PFC",
        "Ins": "Insula", "FrMed": "Frontal medial cortex",
        "FrOper": "Frontal operculum", "Cinga": "Anterior cingulate",
        "Cingm": "Mid-cingulate", "Cingp": "Posterior cingulate",
        "pCun": "Precuneus", "pCunPCC": "Precuneus/posterior cingulate",
        "IPL": "Inferior parietal lobule", "SPL": "Superior parietal lobule",
        "Post": "Postcentral (somatosensory)", "Temp": "Temporal cortex",
        "TempOcc": "Temporal-occipital", "Par": "Parietal cortex",
        "ParOcc": "Parietal-occipital", "OFC": "Orbitofrontal cortex",
        "FEF": "Frontal eye field", "ExStr": "Extrastriate visual",
        "ExStrSup": "Superior extrastriate", "ExStrInf": "Inferior extrastriate",
        "Striate": "Primary visual cortex", "PHC": "Parahippocampal cortex",
        "RSC": "Retrosplenial cortex", "Aud": "Auditory cortex",
    }

    if value:
        print(f"Regions in {value} (decoded):")
    else:
        print("All regions (decoded):")

    print(f"  {'Name':<50} {'Hemisphere':<6} {'Network':<18} {'Region'}")
    print("  " + "-" * 95)

    count = 0
    for i, name in enumerate(_roi_names):
        if network is not None and not _roi_matches_network(i, mode, value):
            continue
        net = _network_map[i]

        # Determine hemisphere
        if "-rh" in name or name.startswith("RH_"):
            hemi = "Right"
        elif "-lh" in name or name.startswith("LH_"):
            hemi = "Left"
        else:
            hemi = "—"

        # Decode the region description
        prefix = name.split("-")[0]
        if prefix in SUBCORTICAL_PREFIXES:
            desc = _subregion_descriptions.get(prefix, prefix)
        else:
            # Format: LH_SalVentAttnA_Ins_1 -> parts[2] = Ins, parts[3] = 1
            parts = name.split("_")
            if len(parts) >= 3:
                subregion_key = parts[2]
                desc = _subregion_descriptions.get(subregion_key, subregion_key)
                if len(parts) >= 4:
                    desc += f" (parcel {parts[3]})"
            else:
                desc = name

        print(f"  {name:<50} {hemi:<6} {net:<18} {desc}")
        count += 1

    print(f"\n{count} regions listed.")


# ============================================================================
# get_edge / plot_edge
# ============================================================================

def get_edge(roi_a, roi_b):
    """
    Get functional connectivity values between two ROIs across all subjects.

    Parameters
    ----------
    roi_a, roi_b : str
        ROI names (fuzzy matching supported).

    Returns
    -------
    numpy array of FC values (one per subject).
    """
    idx_a, name_a = _resolve_roi(roi_a)
    if idx_a is None:
        return None
    idx_b, name_b = _resolve_roi(roi_b)
    if idx_b is None:
        return None

    if idx_a == idx_b:
        print(f"Error: both ROIs resolve to the same region ({name_a}).")
        return None

    return _fc_tensor[idx_a, idx_b, :]


def plot_edge(roi_a, roi_b, behavior_col=None, covariates=None,
              exclude_outliers=None, subgroup=None):
    """
    Scatter plot of an edge's FC values vs. a behavioral variable.

    Parameters
    ----------
    roi_a, roi_b : str
        ROI names (fuzzy matching supported).
    behavior_col : str, optional
        Column name from the behavioral data.
        If None, uses the outcome variable (first column).
    covariates : list of str or 'all', optional
        Variables to control for (partial correlation).
        Pass 'all' to control for all other variables.
    exclude_outliers : float, optional
        Z-score threshold for outlier removal on edge values.
        E.g., exclude_outliers=2 removes subjects with edge z > 2.
    subgroup : dict, optional
        Filter subjects. Keys are column names, values are either:
        - A specific value: {'Sex': 1} keeps only Sex==1
        - 'above_median' or 'below_median': {'Age': 'below_median'}
    """
    if not _check_loaded():
        return

    idx_a, name_a = _resolve_roi(roi_a)
    if idx_a is None:
        return
    idx_b, name_b = _resolve_roi(roi_b)
    if idx_b is None:
        return

    if idx_a == idx_b:
        print(f"Error: both ROIs resolve to the same region ({name_a}).")
        return

    if behavior_col is None:
        behavior_col = _behavior.columns[0]
    elif behavior_col not in _behavior.columns:
        print(f"Error: '{behavior_col}' not found in behavioral data.")
        print(f"Available columns: {list(_behavior.columns)}")
        return

    edge_vals = _fc_tensor[idx_a, idx_b, :].astype(np.float64)
    beh_vals = _behavior[behavior_col].values.astype(np.float64)
    n_original = len(edge_vals)
    adjustments = []

    # 1. Subgroup filter
    keep = np.ones(n_original, dtype=bool)
    if subgroup is not None:
        for col, val in subgroup.items():
            if col not in _behavior.columns:
                print(f"Error: '{col}' not found in behavioral data.")
                return
            col_vals = _behavior[col].values
            if val == 'above_median':
                med = np.median(col_vals)
                keep &= col_vals >= med
                adjustments.append(f"{col} >= {med:.1f}")
            elif val == 'below_median':
                med = np.median(col_vals)
                keep &= col_vals < med
                adjustments.append(f"{col} < {med:.1f}")
            else:
                keep &= col_vals == val
                if col == 'Sex':
                    adjustments.append('male only' if val == 1 else 'female only')
                else:
                    adjustments.append(f"{col} = {val}")
    edge_vals = edge_vals[keep]
    beh_vals = beh_vals[keep]

    # 2. Outlier exclusion (on edge values)
    if exclude_outliers is not None:
        z = np.abs((edge_vals - edge_vals.mean()) / (edge_vals.std() + 1e-12))
        ok = z < exclude_outliers
        n_excl = (~ok).sum()
        edge_vals = edge_vals[ok]
        beh_vals = beh_vals[ok]
        # Also filter the behavioral DataFrame for covariates below
        keep_indices = np.where(keep)[0]
        keep = np.zeros(n_original, dtype=bool)
        keep[keep_indices[ok]] = True
        if n_excl > 0:
            adjustments.append(f"excluding {n_excl} outliers > {exclude_outliers} SD")

    # 3. Covariate residualization
    if covariates is not None:
        if covariates == 'all':
            cov_cols = [c for c in _behavior.columns if c != behavior_col]
        else:
            cov_cols = list(covariates)
            for c in cov_cols:
                if c not in _behavior.columns:
                    print(f"Error: covariate '{c}' not found in behavioral data.")
                    return
        cov_data = _behavior[cov_cols].values[keep].astype(np.float64)
        edge_vals = _residualize(cov_data, edge_vals)
        beh_vals = _residualize(cov_data, beh_vals)
        if covariates == 'all':
            adjustments.append("controlling for all variables")
        else:
            adjustments.append(f"controlling for {', '.join(cov_cols)}")

    r, p = pearsonr(edge_vals, beh_vals)

    # Format p-value
    if p < 0.001:
        p_str = f"p = {p:.2e}"
    else:
        p_str = f"p = {p:.4f}"

    plt.figure(figsize=(7, 5))
    plt.scatter(edge_vals, beh_vals, alpha=0.5, color="steelblue", edgecolors="white",
                linewidth=0.5, s=40)

    # Regression line
    z = np.polyfit(edge_vals, beh_vals, 1)
    x_line = np.linspace(edge_vals.min(), edge_vals.max(), 100)
    plt.plot(x_line, np.polyval(z, x_line), color="coral", linewidth=2)

    net_a = _network_map[idx_a]
    net_b = _network_map[idx_b]
    short_a = "_".join(name_a.split("_")[-2:]) if "_" in name_a else name_a
    short_b = "_".join(name_b.split("_")[-2:]) if "_" in name_b else name_b

    xlabel = f"FC: {short_a}  <->  {short_b}"
    ylabel = behavior_col
    if covariates is not None:
        xlabel += " (residualized)"
        ylabel += " (residualized)"

    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=11)

    title = f"r = {r:.3f}, {p_str}  (n = {len(edge_vals)})\n({net_a} <-> {net_b})"
    if adjustments:
        title += f"\n[{'; '.join(adjustments)}]"
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.show()


# ============================================================================
# test_all_edges / test_network_edges (vectorized)
# ============================================================================

def _vectorized_correlations(fc_tensor, outcome, subject_mask=None,
                             cov_matrix=None):
    """Compute Pearson r and p for all edges vs outcome, vectorized.

    Parameters
    ----------
    fc_tensor : ndarray, shape (n_rois, n_rois, n_subjects)
    outcome : ndarray, shape (n_subjects,)
    subject_mask : bool array, shape (n_subjects,), optional
        If provided, only include these subjects.
    cov_matrix : ndarray, shape (n_included, n_covariates), optional
        Covariates to residualize out. Length must match included subjects.
    """
    if subject_mask is not None:
        fc_tensor = fc_tensor[:, :, subject_mask]
        outcome = outcome[subject_mask].copy()
    else:
        outcome = outcome.copy()

    n_rois = fc_tensor.shape[0]
    n_subjects = fc_tensor.shape[2]
    triu_row, triu_col = np.triu_indices(n_rois, k=1)
    n_edges = len(triu_row)

    # Extract upper triangle values: (n_edges, n_subjects)
    edge_matrix = np.empty((n_edges, n_subjects), dtype=np.float64)
    for s in range(n_subjects):
        edge_matrix[:, s] = fc_tensor[triu_row, triu_col, s]

    # Residualize covariates if provided
    n_cov = 0
    if cov_matrix is not None:
        n_cov = cov_matrix.shape[1]
        X_int = np.column_stack([cov_matrix, np.ones(n_subjects)])
        # Residualize outcome
        beta_y = np.linalg.lstsq(X_int, outcome, rcond=None)[0]
        outcome = outcome - X_int @ beta_y
        # Residualize all edges (vectorized)
        beta_edges = np.linalg.lstsq(X_int, edge_matrix.T, rcond=None)[0]
        edge_matrix = edge_matrix - (X_int @ beta_edges).T

    # Vectorized Pearson r
    x = edge_matrix - edge_matrix.mean(axis=1, keepdims=True)
    y = outcome - outcome.mean()
    r_vals = (x @ y) / (np.sqrt((x ** 2).sum(axis=1)) * np.sqrt((y ** 2).sum()) + 1e-12)

    # Convert to p-values (adjust df for covariates)
    df = n_subjects - 2 - n_cov
    t_vals = r_vals * np.sqrt(df / (1 - r_vals ** 2 + 1e-12))
    p_vals = 2 * t_dist.sf(np.abs(t_vals), df=df)

    return triu_row, triu_col, r_vals, p_vals


def test_all_edges(behavior_col=None, covariates=None, exclude_outliers=None,
                   subgroup=None, correction=None, _quiet=False):
    """
    Test all edges for correlation with a behavioral variable.

    Parameters
    ----------
    behavior_col : str, optional
        Column name from behavioral data. If None, uses the outcome variable.
    covariates : list of str or 'all', optional
        Variables to control for (partial correlation).
    exclude_outliers : float, optional
        Z-score threshold for outlier removal on the outcome variable.
    subgroup : dict, optional
        Filter subjects (e.g., {'Sex': 1} or {'Age': 'below_median'}).
    correction : str, optional
        Multiple comparisons correction: 'fdr' (Benjamini-Hochberg) or 'bonferroni'.
        Adds a 'p_corrected' column to results.

    Returns
    -------
    pandas DataFrame with columns: ROI_A, ROI_B, r, p, network_A, network_B
    Sorted by p-value (ascending).
    """
    if not _check_loaded():
        return None

    if behavior_col is None:
        behavior_col = _behavior.columns[0]
    elif behavior_col not in _behavior.columns:
        print(f"Error: '{behavior_col}' not found.")
        print(f"Available: {list(_behavior.columns)}")
        return None

    outcome = _behavior[behavior_col].values.astype(np.float64)
    n_original = len(outcome)
    adj_parts = []

    # 1. Subgroup filter
    mask = np.ones(n_original, dtype=bool)
    if subgroup is not None:
        for col, val in subgroup.items():
            if col not in _behavior.columns:
                print(f"Error: '{col}' not found in behavioral data.")
                return None
            col_vals = _behavior[col].values
            if val == 'above_median':
                mask &= col_vals >= np.median(col_vals)
                adj_parts.append(f"{col} >= median")
            elif val == 'below_median':
                mask &= col_vals < np.median(col_vals)
                adj_parts.append(f"{col} < median")
            else:
                mask &= col_vals == val
                if col == 'Sex':
                    adj_parts.append('male only' if val == 1 else 'female only')
                else:
                    adj_parts.append(f"{col} = {val}")

    # 2. Outlier exclusion (on outcome variable)
    if exclude_outliers is not None:
        out_vals = outcome[mask]
        z = np.abs((out_vals - out_vals.mean()) / (out_vals.std() + 1e-12))
        ok = z < exclude_outliers
        n_excl = (~ok).sum()
        keep_idx = np.where(mask)[0]
        mask[keep_idx[~ok]] = False
        if n_excl > 0:
            adj_parts.append(f"excluding {n_excl} outcome outliers > {exclude_outliers} SD")

    # 3. Covariates
    cov_matrix = None
    if covariates is not None:
        if covariates == 'all':
            cov_cols = [c for c in _behavior.columns if c != behavior_col]
        else:
            cov_cols = list(covariates)
            for c in cov_cols:
                if c not in _behavior.columns:
                    print(f"Error: covariate '{c}' not found.")
                    return None
        cov_matrix = _behavior[cov_cols].values[mask].astype(np.float64)
        if covariates == 'all':
            adj_parts.append("controlling for all variables")
        else:
            adj_parts.append(f"controlling for {', '.join(cov_cols)}")

    adj_str = f" ({'; '.join(adj_parts)})" if adj_parts else ""
    n_subj = mask.sum()
    use_mask = mask if n_subj < n_original else None

    if not _quiet:
        print(f"Testing all edges vs {behavior_col}{adj_str}...")
    triu_row, triu_col, r_vals, p_vals = _vectorized_correlations(
        _fc_tensor, outcome, subject_mask=use_mask, cov_matrix=cov_matrix
    )

    results = pd.DataFrame({
        "ROI_A": [_roi_names[i] for i in triu_row],
        "ROI_B": [_roi_names[j] for j in triu_col],
        "r": r_vals,
        "p": p_vals,
        "network_A": [_network_map[i] for i in triu_row],
        "network_B": [_network_map[j] for j in triu_col],
    })
    results = results.sort_values("p").reset_index(drop=True)

    # Apply multiple comparisons correction if requested
    if correction is not None:
        results = _apply_correction(results, correction)

    n_sig = (results["p"] < 0.05).sum()
    n_total = len(results)
    if not _quiet:
        print(f"Done! {n_total:,} edges tested ({n_subj} subjects).")
        print(f"  Significant at p < 0.05 (uncorrected): {n_sig:,}")
        if correction is not None:
            n_corr = (results["p_corrected"] < 0.05).sum()
            print(f"  Surviving {correction.upper()} correction: {n_corr:,}")
    return results


def test_network_edges(network, behavior_col=None, covariates=None,
                       exclude_outliers=None, subgroup=None, correction=None,
                       within=False):
    """
    Test edges involving (or within) a specific network for correlation with behavior.

    Parameters
    ----------
    network : str
        Network name (e.g., 'SalVentAttnA', 'Subcortical'). Fuzzy matching supported.
    behavior_col : str, optional
        Column name from behavioral data. If None, uses the outcome variable.
    covariates : list of str or 'all', optional
        Variables to control for (partial correlation).
    exclude_outliers : float, optional
        Z-score threshold for outlier removal on the outcome variable.
    subgroup : dict, optional
        Filter subjects (e.g., {'Sex': 1} or {'Age': 'below_median'}).
    correction : str, optional
        Multiple comparisons correction: 'fdr' (Benjamini-Hochberg) or 'bonferroni'.
        Applied AFTER filtering to network edges (so correction is based on
        the number of network edges, not all 23,220).
    within : bool, optional
        If True, only test edges where BOTH ROIs are in the target network.
        If False (default), test all edges where at least one ROI is in the network.

    Returns
    -------
    pandas DataFrame with columns: ROI_A, ROI_B, r, p, network_A, network_B
    """
    if not _check_loaded():
        return None

    # Resolve network name (accepts both 7-network and 17-network names)
    mode, value = _resolve_network(network)
    if mode is None:
        return None

    if behavior_col is None:
        behavior_col = _behavior.columns[0]
    elif behavior_col not in _behavior.columns:
        print(f"Error: '{behavior_col}' not found.")
        print(f"Available: {list(_behavior.columns)}")
        return None

    # Get all edges WITHOUT correction (correction applied after filtering)
    all_results = test_all_edges(behavior_col, covariates=covariates,
                                 exclude_outliers=exclude_outliers,
                                 subgroup=subgroup, _quiet=True)
    if all_results is None:
        return None

    if within:
        if mode == 'net7':
            net_mask = (all_results["network_A"] == value) & (all_results["network_B"] == value)
        else:  # net17 — match by ROI name substring
            net_mask = (all_results["ROI_A"].str.contains(value)) & (all_results["ROI_B"].str.contains(value))
        scope_label = f"Within {value}"
    else:
        if mode == 'net7':
            net_mask = (all_results["network_A"] == value) | (all_results["network_B"] == value)
        else:  # net17 — match by ROI name substring
            net_mask = (all_results["ROI_A"].str.contains(value)) | (all_results["ROI_B"].str.contains(value))
        scope_label = f"Edges involving {value}"
    filtered = all_results[net_mask].reset_index(drop=True)

    # Apply correction AFTER filtering to network edges
    if correction is not None:
        filtered = _apply_correction(filtered, correction)

    n_sig = (filtered["p"] < 0.05).sum()
    print(f"{scope_label}: {len(filtered):,}")
    print(f"  Significant at p < 0.05 (uncorrected): {n_sig}")
    if correction is not None:
        n_corr = (filtered["p_corrected"] < 0.05).sum()
        print(f"  Surviving {correction.upper()} correction: {n_corr}")
    return filtered


# ============================================================================
# Visualization: connectome heatmap
# ============================================================================

def plot_connectome(subject=None):
    """
    Plot a heatmap of the FC matrix, organized by network.

    Parameters
    ----------
    subject : int, optional
        Subject index (0-based). If None, shows the group average.
    """
    if not _check_loaded():
        return

    # Sort ROIs by network
    sort_order = []
    for net in _network_order:
        indices = [i for i, n in enumerate(_network_map) if n == net]
        sort_order.extend(indices)

    if subject is not None:
        if subject < 0 or subject >= _fc_tensor.shape[2]:
            print(f"Error: subject must be 0-{_fc_tensor.shape[2]-1}")
            return
        mat = _fc_tensor[:, :, subject]
        title = f"Functional Connectivity Matrix (Subject {subject})"
    else:
        mat = _fc_tensor.mean(axis=2)
        title = "Group Average Functional Connectivity Matrix"

    # Reorder matrix by network
    mat_sorted = mat[np.ix_(sort_order, sort_order)]

    fig, ax = plt.subplots(figsize=(10, 9))
    vmax = np.percentile(np.abs(mat_sorted[~np.eye(_n_rois, dtype=bool)]), 95)
    im = ax.imshow(mat_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    # Add network boundary lines
    pos = 0
    tick_positions = []
    tick_labels = []
    for net in _network_order:
        count = sum(1 for n in _network_map if n == net)
        if count == 0:
            continue
        ax.axhline(pos - 0.5, color="black", linewidth=0.5, alpha=0.5)
        ax.axvline(pos - 0.5, color="black", linewidth=0.5, alpha=0.5)
        tick_positions.append(pos + count / 2)
        tick_labels.append(net)
        pos += count

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax, label="Functional Connectivity", shrink=0.8)
    plt.tight_layout()
    plt.show()


def plot_network_matrix(network):
    """
    Zoomed heatmap for edges within/involving one network.

    Parameters
    ----------
    network : str
        Network name (fuzzy matching supported).
    """
    if not _check_loaded():
        return

    # Resolve network (accepts both 7-network and 17-network names)
    mode, value = _resolve_network(network)
    if mode is None:
        return

    indices = [i for i in range(len(_roi_names)) if _roi_matches_network(i, mode, value)]
    if len(indices) == 0:
        print(f"No ROIs found in network '{resolved}'.")
        return

    mat = _fc_tensor.mean(axis=2)
    sub_mat = mat[np.ix_(indices, indices)]
    labels = [_roi_names[i] for i in indices]
    # Shorten labels: LH_SalVentAttnA_Ins_1 -> LH_Ins_1 (drop network name)
    short_labels = []
    for lab in labels:
        parts = lab.split("_")
        if len(parts) >= 3:
            # Keep hemisphere + region + parcel: LH_Ins_1
            short_labels.append(f"{parts[0]}_{'_'.join(parts[2:])}")
        else:
            short_labels.append(lab)

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = max(0.1, np.percentile(np.abs(sub_mat[~np.eye(len(indices), dtype=bool)]), 95))
    im = ax.imshow(sub_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_title(f"{value} Network ({len(indices)} ROIs)", fontsize=13)
    plt.colorbar(im, ax=ax, label="Functional Connectivity", shrink=0.8)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Glass brain plot (uses nilearn)
# ============================================================================

def plot_glass_brain(results_df=None, n_top=10, p_threshold=0.05, corrected=False):
    """
    Plot significant edges on a glass brain using nilearn.

    Parameters
    ----------
    results_df : DataFrame, optional
        Output from test_all_edges() or test_network_edges().
        If None, you must run test_all_edges() first.
    n_top : int
        Number of top edges to display (default: 10).
    p_threshold : float
        Only show edges with p < this value (default: 0.05).
    corrected : bool
        If True, use 'p_corrected' column instead of 'p' (default: False).
    """
    if not _check_loaded():
        return

    if results_df is None:
        print("Error: pass a results DataFrame from test_all_edges() or test_network_edges().")
        print("  Example: results = test_all_edges()")
        print("           plot_glass_brain(results)")
        return

    if _roi_coords is None:
        print("Error: ROI coordinates not found (data/roi_mni_coords.npy).")
        print("  Try re-running load_dataset() to download the coordinate file.")
        return

    try:
        from nilearn import plotting
    except ImportError:
        print("Error: nilearn is required for glass brain plots.")
        print("  Install it with: !pip install nilearn")
        return

    # Filter significant and take top N
    p_col = "p_corrected" if corrected and "p_corrected" in results_df.columns else "p"
    sig = results_df[results_df[p_col] < p_threshold].head(n_top)
    p_label = "corrected p" if p_col == "p_corrected" else "p"
    if len(sig) == 0:
        print(f"No edges with {p_label} < {p_threshold} to display.")
        return

    # Build adjacency matrix with r-values for significant edges only
    n = _n_rois
    adj = np.zeros((n, n), dtype=np.float32)
    for _, row in sig.iterrows():
        idx_a = _roi_names.index(row["ROI_A"])
        idx_b = _roi_names.index(row["ROI_B"])
        adj[idx_a, idx_b] = row["r"]
        adj[idx_b, idx_a] = row["r"]

    # Node sizes: larger for involved ROIs, zero for others
    involved = set()
    for _, row in sig.iterrows():
        involved.add(_roi_names.index(row["ROI_A"]))
        involved.add(_roi_names.index(row["ROI_B"]))
    node_size = np.array([25 if i in involved else 0 for i in range(n)])

    # Node colors by network
    node_color = [NETWORK_COLORS.get(_network_map[i], "gray") if i in involved
                  else "lightgray" for i in range(n)]

    # Edge color limits: symmetric around zero so positive = warm, negative = cool
    max_abs_r = sig["r"].abs().max()
    edge_vmin = -max_abs_r
    edge_vmax = max_abs_r

    # Plot: left, top-down (z), right views
    fig = plt.figure(figsize=(14, 5))
    display = plotting.plot_connectome(
        adj,
        _roi_coords,
        node_size=node_size,
        node_color=node_color,
        edge_cmap="RdBu_r",
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        edge_kwargs={"linewidth": 4, "alpha": 0.7},
        display_mode="lzr",
        figure=fig,
        title=f"Top {len(sig)} Edges ({p_label} < {p_threshold})",
        colorbar=True,
    )

    # Add "r" label to the colorbar
    for ax in fig.get_axes():
        # The colorbar axis is the last one added by nilearn
        if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == '':
            # Check if this looks like a colorbar (narrow axis)
            bbox = ax.get_position()
            if bbox.width < 0.05:
                ax.set_ylabel("r", fontsize=11, rotation=0, labelpad=10)

    # Add network color legend
    involved_nets = sorted(set(_network_map[i] for i in involved))
    legend_patches = [mpatches.Patch(color=NETWORK_COLORS.get(n, "gray"), label=n)
                      for n in involved_nets]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(involved_nets),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.show()


# ============================================================================
# Convenience: behavior DataFrame access
# ============================================================================

def get_behavior(column=None):
    """
    Return the behavioral data as a pandas DataFrame, or a single column as a Series.

    Parameters
    ----------
    column : str, optional
        If provided, return just that column (e.g., 'PHQ9', 'Age').
        If None, return the full DataFrame.
    """
    if not _check_loaded():
        return None
    if column is not None:
        if column not in _behavior.columns:
            print(f"Error: '{column}' not found in behavioral data.")
            print(f"Available columns: {list(_behavior.columns)}")
            return None
        return _behavior[column].copy()
    return _behavior.copy()


# ============================================================================
# Variable descriptions and exploration tools
# ============================================================================

VARIABLE_DESCRIPTIONS = {
    # Outcome variables
    "Pain_VAS": "Pain severity (Visual Analog Scale, 0-10)",
    "PHQ9": "Depression severity (PHQ-9 questionnaire, 0-27)",
    "GAD7": "Anxiety severity (GAD-7 questionnaire, 0-21)",
    # Demographics
    "Age": "Age in years",
    "Sex": "Biological sex (0 = female, 1 = male)",
    "BMI": "Body mass index (kg/m^2)",
    "HRV": "Heart rate variability (RMSSD, ms)",
    "Education_yrs": "Education (years)",
    "Income_k": "Household income ($1000/year)",
    # Lifestyle
    "Physical_Activity": "Physical activity (minutes/week)",
    "Caffeine_mg": "Caffeine intake (mg/day)",
    "Alcohol_drinks": "Alcohol consumption (drinks/week)",
    "Screen_Time": "Screen time (hours/day)",
    "Social_Media": "Social media use (minutes/day)",
    # Clinical / psychological
    "Stress_Level": "Perceived stress (PSS, 0-40)",
    "Sleep_Quality": "Sleep quality (PSQI, 0-21; higher = worse sleep)",
    "Neuroticism": "Neuroticism (NEO, 0-48)",
    "Self_Esteem": "Self-esteem (Rosenberg, 0-30)",
    # Pain-specific
    "Catastrophizing": "Pain catastrophizing (PCS, 0-52)",
    "Fear_Avoidance": "Fear-avoidance beliefs (FABQ, 0-96)",
    "Pain_Duration_mo": "Pain duration (months)",
    # Depression-specific
    "Rumination": "Rumination (RRS, 22-88)",
    "Loneliness": "Loneliness (UCLA scale, 20-80)",
    "Social_Support": "Perceived social support (MSPSS, 12-60)",
    # Anxiety-specific
    "Worry": "Worry (PSWQ, 16-80)",
    "Intolerance_Uncertainty": "Intolerance of uncertainty (IUS, 27-135)",
}


def describe_variables():
    """Print a table of all behavioral variables with their descriptions."""
    if not _check_loaded():
        return

    print(f"Variables in the {_topic} dataset:")
    print(f"{'Variable':<30} {'Description'}")
    print("-" * 75)
    for col in _behavior.columns:
        desc = VARIABLE_DESCRIPTIONS.get(col, "")
        print(f"{col:<30} {desc}")
    print(f"\n{len(_behavior.columns)} variables total.")


def plot_behavior(var_a, var_b, covariates=None, subgroup=None):
    """
    Scatter plot of one behavioral variable vs another with regression line
    and Pearson r.

    Parameters
    ----------
    var_a, var_b : str
        Column names from the behavioral data.
    covariates : list of str, optional
        Variables to control for (residualize both var_a and var_b).
    subgroup : dict, optional
        Filter subjects before plotting. Keys are column names, values are
        specific values or 'above_median'/'below_median'.
    """
    if not _check_loaded():
        return

    for v in [var_a, var_b]:
        if v not in _behavior.columns:
            print(f"Error: '{v}' not found in behavioral data.")
            print(f"Available: {list(_behavior.columns)}")
            return

    x = _behavior[var_a].values.astype(np.float64)
    y = _behavior[var_b].values.astype(np.float64)
    n_original = len(x)
    adjustments = []

    # 1. Subgroup filter
    keep = np.ones(n_original, dtype=bool)
    if subgroup is not None:
        for col, val in subgroup.items():
            if col not in _behavior.columns:
                print(f"Error: '{col}' not found in behavioral data.")
                return
            col_vals = _behavior[col].values
            if val == 'above_median':
                med = np.median(col_vals)
                keep &= col_vals >= med
                adjustments.append(f"{col} >= {med:.1f}")
            elif val == 'below_median':
                med = np.median(col_vals)
                keep &= col_vals < med
                adjustments.append(f"{col} < {med:.1f}")
            else:
                keep &= col_vals == val
                if col == 'Sex':
                    adjustments.append('male only' if val == 1 else 'female only')
                else:
                    adjustments.append(f"{col} = {val}")
    x = x[keep]
    y = y[keep]

    # 2. Covariate residualization
    xlabel = var_a
    ylabel = var_b
    if covariates is not None:
        cov_cols = list(covariates)
        for c in cov_cols:
            if c not in _behavior.columns:
                print(f"Error: covariate '{c}' not found in behavioral data.")
                return
        cov_data = _behavior[cov_cols].values[keep].astype(np.float64)
        x = _residualize(cov_data, x)
        y = _residualize(cov_data, y)
        adjustments.append(f"controlling for {', '.join(cov_cols)}")
        xlabel += " (residualized)"
        ylabel += " (residualized)"

    r, p = pearsonr(x, y)

    if p < 0.001:
        p_str = f"p = {p:.2e}"
    else:
        p_str = f"p = {p:.4f}"

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.5, color="steelblue", edgecolors="white",
                linewidth=0.5, s=40)

    # Regression line
    z = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_line, np.polyval(z, x_line), color="coral", linewidth=2)

    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)

    title = f"r = {r:.3f}, {p_str}  (n = {len(x)})"
    if adjustments:
        title += f"\n[{'; '.join(adjustments)}]"
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()
