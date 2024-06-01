import numpy as np
import pandas as pd


def compute_ground_effect_cl_cd(h_c, alpha_deg, re, aspect_ratio):
    """
    Compute synthetic C_L and C_D in ground effect for a simple wing.

    Parameters
    ----------
    h_c : array-like
        Height above ground divided by chord.
    alpha_deg : array-like
        Angle of attack in degrees.
    re : array-like
        Reynolds number (dimensionless).
    aspect_ratio : array-like
        Wing aspect ratio.

    Returns
    -------
    cl_ge : ndarray
        Lift coefficient in ground effect.
    cd_ge : ndarray
        Drag coefficient in ground effect.
    """
    h_c = np.asarray(h_c, dtype=float)
    alpha_deg = np.asarray(alpha_deg, dtype=float)
    re = np.asarray(re, dtype=float)
    aspect_ratio = np.asarray(aspect_ratio, dtype=float)

    # Convert alpha to radians
    alpha_rad = np.deg2rad(alpha_deg)

    # Simple lift curve slope approximation using finite wing theory
    # a0 ~ 2*pi for thin airfoil, modified by aspect ratio
    cl_alpha = 2.0 * np.pi * aspect_ratio / (aspect_ratio + 2.0)

    # Out-of-ground-effect lift (linear model, no stall)
    cl_inf = cl_alpha * alpha_rad

    # Profile drag coefficient (baseline) - weakly dependent on Re
    re_million = re / 1e6
    cd0_base = 0.018  # base profile drag
    cd0 = cd0_base + 0.002 * (1.0 / np.maximum(re_million, 0.5) - 1.0 / 5.0)

    # Induced drag out of ground effect: CDi = CL^2 / (pi * AR * e)
    e = 0.8  # Oswald efficiency factor
    cdi_inf = cl_inf ** 2 / (np.pi * aspect_ratio * e)

    cd_inf = cd0 + cdi_inf

    # --- Ground effect model (synthetic but physically plausible trends) ---
    # As h/c decreases, CL increases and induced drag reduces.

    # Increase in lift due to ground effect: stronger at small h/c, saturates as h/c grows.
    # Example functional form: 1 + A / (h/c + B)
    A_lift = 0.25
    B_lift = 0.3
    lift_factor = 1.0 + A_lift / (h_c + B_lift)

    # Reduction in induced drag: strongest very near ground, vanishes with height.
    # Example: 1 - C * exp(-h/c)
    C_drag = 0.5
    drag_factor = 1.0 - C_drag * np.exp(-h_c)

    cl_ge = cl_inf * lift_factor
    cdi_ge = cdi_inf * drag_factor
    cd_ge = cd0 + cdi_ge

    return cl_ge, cd_ge


def generate_dataset(
    n_per_config=400,
    random_state=42,
    aspect_ratios=(6.0, 9.0, 12.0),
    h_c_range=(0.1, 3.0),
    alpha_range=(-4.0, 12.0),
    re_range=(5e5, 5e6),
):
    """
    Generate a synthetic dataset for lift and drag in ground effect.

    Each sample corresponds to a combination of:
    - height-to-chord ratio h/c
    - angle of attack alpha
    - Reynolds number Re
    - wing aspect ratio AR

    The targets are C_L and C_D in ground effect.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix [h_c, alpha_deg, Re_million, AR].
    y : ndarray, shape (n_samples, 2)
        Target matrix [C_L_ground, C_D_ground].
    df : pandas.DataFrame
        Full dataset including features and targets.
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for ar in aspect_ratios:
        # Sample parameters for configuration
        h_c = rng.uniform(h_c_range[0], h_c_range[1], size=n_per_config)
        alpha_deg = rng.uniform(alpha_range[0], alpha_range[1], size=n_per_config)
        re = rng.uniform(re_range[0], re_range[1], size=n_per_config)

        cl_ge, cd_ge = compute_ground_effect_cl_cd(h_c, alpha_deg, re, ar)

        for i in range(n_per_config):
            rows.append(
                {
                    "h_c": float(h_c[i]),
                    "alpha_deg": float(alpha_deg[i]),
                    "Re_million": float(re[i] / 1e6),
                    "AR": float(ar),
                    "C_L": float(cl_ge[i]),
                    "C_D": float(cd_ge[i]),
                }
            )

    df = pd.DataFrame(rows)
    feature_cols = ["h_c", "alpha_deg", "Re_million", "AR"]
    target_cols = ["C_L", "C_D"]

    X = df[feature_cols].to_numpy()
    y = df[target_cols].to_numpy()

    return X, y, df
