import numpy as np


def generate_sensing_matrix(
    m: int,
    N: int,
    dist: str = "gaussian",
    normalize: str = "column",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate an (m x N) sensing matrix Phi.

    Parameters:
    -----------
    m : int
        Number of measurements (rows).
    N : int
        Signal dimension (columns).
    dist : str
        Distribution of entries: "gaussian" or "rademacher".
    normalize : str
        - "column": Normalize each column to have Euclidean norm 1.
        - "variance": Scale matrix so entries have variance 1/m (expected column norm is 1).
        - "none": Unscaled matrix (standard normal or +/-1).
    rng : np.random.Generator | None
        NumPy random generator.

    Returns:
    --------
    Phi : np.ndarray of shape (m, N)
    """
    if rng is None:
        rng = np.random.default_rng()

    if dist == "gaussian":
        Phi = rng.normal(loc=0.0, scale=1.0, size=(m, N))
    elif dist == "rademacher":
        Phi = rng.choice([-1.0, 1.0], size=(m, N))
    else:
        raise ValueError(f"Unknown sensing matrix distribution: {dist}")

    if normalize == "column":
        col_norms = np.linalg.norm(Phi, axis=0)
        # Avoid dividing by zero for degenerate columns
        col_norms = np.where(col_norms == 0, 1.0, col_norms)
        Phi = Phi / col_norms
    elif normalize == "variance":
        Phi = Phi / np.sqrt(m)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization mode: {normalize}")

    return Phi


def add_noise(
    y: np.ndarray,
    sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add zero-mean Gaussian noise with standard deviation sigma to measurement vector y.

    Returns:
    --------
    y_noisy : np.ndarray
    noise : np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()

    if sigma <= 0.0:
        noise = np.zeros_like(y)
    else:
        noise = rng.normal(loc=0.0, scale=sigma, size=y.shape)

    return y + noise, noise


def add_noise_snr(
    y: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Add AWGN to signal y to achieve a target Signal-to-Noise Ratio (SNR) in decibels.

    SNR_dB = 10 * log10( ||y||_2^2 / ||noise||_2^2 )

    Returns:
    --------
    y_noisy : np.ndarray
    noise : np.ndarray
    sigma : float
    """
    # Calculate the power of the signal
    signal_power = np.mean(y ** 2)
    # Calculate the power of the noise based on the desired SNR
    noise_power = signal_power / (10.0 ** (snr_db / 10.0)) # /10 as it is decibels, not bels
    sigma = float(np.sqrt(noise_power)) # Noise power is the variance of the noise
    y_noisy, noise = add_noise(y, sigma=sigma, rng=rng) # just gaussian noise, no impulse noise
    return y_noisy, noise, sigma
