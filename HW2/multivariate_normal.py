import numpy as np
from scipy.stats import multivariate_normal as scipy_mvn



def multivariate_normal_density(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Evaluate the multivariate normal PDF at a single D-dimensional point.

    Parameters
    ----------
    x     : (D,) array — the point to evaluate
    mu    : (D,) array — mean vector
    Sigma : (D, D) array — positive-definite covariance matrix

    Returns
    -------
    density : float — p(x | mu, Sigma)

    The PDF is:
        p(x) = (2π)^{-D/2} |Σ|^{-1/2}
               exp( -½ (x-μ)ᵀ Σ⁻¹ (x-μ) )
    """
    x     = np.asarray(x, dtype=float)
    mu    = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    D = len(mu)

    # Use Cholesky for numerically stable log-determinant and solve
    L = np.linalg.cholesky(Sigma)                      # Σ = L Lᵀ
    log_det = 2.0 * np.sum(np.log(np.diag(L)))         # log|Σ| = 2 Σ log(Lᵢᵢ)

    diff = x - mu
    # Solve L v = diff  →  v = L⁻¹ diff
    v = np.linalg.solve(L, diff)
    maha_sq = v @ v                                    # (x-μ)ᵀ Σ⁻¹ (x-μ)

    log_norm = -0.5 * (D * np.log(2 * np.pi) + log_det)
    log_p    = log_norm - 0.5 * maha_sq

    return float(np.exp(log_p))


def compare_methods():
    rng = np.random.default_rng(0)

    parameterizations = {
        # ① Spherical — zero covariance, same variance in every dimension
        "Spherical (σ²I)": {
            "mu":    np.array([1.0, 2.0, 3.0]),
            "Sigma": 2.5 * np.eye(3),
        },
        # ② Diagonal — zero covariance, different variance per dimension
        "Diagonal (diag(σ²))": {
            "mu":    np.array([0.0, -1.0, 4.0]),
            "Sigma": np.diag([1.0, 3.0, 0.5]),
        },
        # ③ Full covariance — non-zero off-diagonals
        "Full covariance": {
            "mu":    np.array([2.0, 0.0, -2.0]),
            "Sigma": np.array([
                [4.0,  1.2, -0.8],
                [1.2,  2.0,  0.6],
                [-0.8, 0.6,  1.5],
            ]),
        },
    }

    print("=" * 65)
    print("  Comparison: multivariate_normal_density vs scipy")
    print("=" * 65)

    for name, params in parameterizations.items():
        mu, Sigma = params["mu"], params["Sigma"]
        # Draw a random test point
        x = rng.multivariate_normal(mu, Sigma)

        our_val   = multivariate_normal_density(x, mu, Sigma)
        scipy_val = scipy_mvn.pdf(x, mean=mu, cov=Sigma)
        abs_err   = abs(our_val - scipy_val)

        print(f"\n  [{name}]")
        print(f"    x          = {np.round(x, 4)}")
        print(f"    Our density  = {our_val:.10e}")
        print(f"    SciPy density= {scipy_val:.10e}")
        print(f"    |error|      = {abs_err:.3e}")

    print()



class MultivariateNormal:
    """
    Multivariate Normal distribution with scipy.

    Parameters
    ----------
    mu    : (D,) array-like — mean vector
    Sigma : (D, D) array-like — positive-definite covariance matrix

    Methods
    -------
    log_pdf(x)    : evaluate log-density at x; x may be (..., D)
    pdf(x)        : evaluate density (exp of log_pdf)
    rvs(shape)    : draw samples of the given shape; returns (*shape, D) array
    """

    def __init__(self, mu: np.ndarray, Sigma: np.ndarray):
        self.mu    = np.asarray(mu,    dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.D     = self.mu.shape[0]

        # Pre-compute Cholesky once; reuse for both sampling and log-pdf
        self._L        = np.linalg.cholesky(self.Sigma)          # Σ = L Lᵀ
        self._log_det  = 2.0 * np.sum(np.log(np.diag(self._L)))
        self._log_norm = -0.5 * (self.D * np.log(2 * np.pi) + self._log_det)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Vectorised log-density.

        Parameters
        ----------
        x : (..., D) array — one or more D-dimensional points

        Returns
        -------
        log_p : (...,) array of log-densities
        """
        x    = np.asarray(x, dtype=float)
        diff = x - self.mu        

        # Solve L v = diff  for each point in batch
        # np.linalg.solve broadcasts over leading dimensions
        v      = np.linalg.solve(self._L, diff[..., np.newaxis])
        v      = v[..., 0]                                         
        maha_sq = np.sum(v ** 2, axis=-1)

        return self._log_norm - 0.5 * maha_sq

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Density (exp of log_pdf). Prefer log_pdf for numerical work."""
        return np.exp(self.log_pdf(x))


    def rvs(self, shape=()) -> np.ndarray:
        """
        Draw random samples.
        
        Parameters
        ----------
        shape : int or tuple of ints — number / shape of samples to draw
        
        Returns
        -------
        samples : (*shape, D) array
        """
        if isinstance(shape, int):
            shape = (shape,)
        z = np.random.standard_normal((*shape, self.D))   # (*shape, D)
        return self.mu + (self._L @ z[..., np.newaxis])[..., 0]

    def __repr__(self) -> str:
        return (f"MultivariateNormal(mu={self.mu}, "
                f"Sigma=\n{self.Sigma})")


# ─────────────────────────────────────────────────────────────
# 4.  Demo / test of the class
# ─────────────────────────────────────────────────────────────

def run_class_demo():
    print("=" * 65)
    print("  MultivariateNormal class demo")
    print("=" * 65)

    mu    = np.array([1.0, -1.0])
    Sigma = np.array([[2.0, 0.8],
                      [0.8, 1.0]])

    dist  = MultivariateNormal(mu, Sigma)
    scipy_ref = scipy_mvn(mean=mu, cov=Sigma)

    # ── single point ──
    x_single = np.array([0.5, 0.3])
    print(f"\n  Single point x = {x_single}")
    print(f"  Our  log_pdf : {dist.log_pdf(x_single):.8f}")
    print(f"  SciPy logpdf : {scipy_ref.logpdf(x_single):.8f}")

    # ── batch of points ──
    rng     = np.random.default_rng(42)
    x_batch = rng.standard_normal((5, 2))
    our_lp  = dist.log_pdf(x_batch)
    sci_lp  = scipy_ref.logpdf(x_batch)

    print(f"\n  Batch (5 × 2) log-pdf comparison:")
    print(f"  {'Our':>12}  {'SciPy':>12}  {'|Δ|':>10}")
    for o, s in zip(our_lp, sci_lp):
        print(f"  {o:12.8f}  {s:12.8f}  {abs(o-s):.3e}")

    # ── sampling ──
    samples = dist.rvs(shape=(10_000,))
    print(f"\n  10 000 samples — empirical vs true moments:")
    print(f"  Mean  true: {mu}")
    print(f"  Mean  emp.: {samples.mean(axis=0).round(3)}")
    print(f"  Cov   true:\n{Sigma}")
    print(f"  Cov   emp.:\n{np.cov(samples.T).round(3)}")

    # ── 3-D shape ──
    grid = dist.rvs(shape=(4, 3))
    print(f"\n  rvs(shape=(4,3)) → shape {grid.shape}  (expected (4,3,2))")
    print()


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    compare_methods()
    
