import numpy as np
from scipy.optimize import minimize

from models.rotation_invariant import RotationInvariantEstimators
from models.silverstein_solver import SilverSteinEquationSolver
from tools.tools import FloatArray


class AdaptedClippingEstimator(RotationInvariantEstimators):
    """
    Adapted eigenvalue clipping estimator: fits a two-atom population
    spectral distribution H to the empirical eigenvalue histogram via the
    Silverstein equation (SilverSteinEquationSolver), derives the
    corresponding theoretical spectrum edge, and clips eigenvalues using
    this data-driven threshold instead of the naive Marchenko-Pastur edge.

    Inherits from RotationInvariantEstimators:
      - fit() handles eigendecomposition and matrix reconstruction
      - Only transform_eigenvalues() needs to be implemented here
    """

    def __init__(
        self,
        *,
        assume_centered: bool = False,
        n_bins: int = 50,
        x0: tuple[float, float] = (0.9, 0.8),
    ):
        """
        Parameters
        ----------
        assume_centered : bool, default=False
            Passed to RotationInvariantEstimators base class.
        n_bins : int, default=50
            Number of bins used to build the empirical eigenvalue histogram.
        x0 : tuple[float, float], default=(0.9, 0.8)
            Initial guess (w1, tau1) for the two-atom H fit.
        """
        super().__init__(assume_centered=assume_centered)
        self.n_bins = n_bins
        self.x0 = x0
        self.threshold_: float | None = None

    def transform_eigenvalues(self, eigvals: FloatArray) -> FloatArray:
        """
        Apply adapted clipping to the raw sample eigenvalues:
          1. Fit a two-atom population spectral distribution H to the
             empirical eigenvalue histogram via the Silverstein equation.
          2. Locate the theoretical spectrum edge from the fitted H.
          3. Clip: keep eigenvalues above the edge unchanged (signal),
             replace the others by their common mean (noise), preserving
             the trace.

        Parameters
        ----------
        eigvals : FloatArray
            Raw eigenvalues of the empirical correlation matrix S,
            in the same order as the eigenvectors stored in self.eigvecs_.

        Returns
        -------
        FloatArray
            Cleaned eigenvalues in the SAME order as eigvals
            (no reordering — eigvals[i] maps to the same eigenvector).
        """
        assert self.p_ is not None and self.n_ is not None
        q: float = self.p_ / self.n_

        #  Step 1: fit the two-atom H to the empirical spectral histogram
        solver = SilverSteinEquationSolver(q)
        hist_counts, bin_edges = np.histogram(
            eigvals, bins=self.n_bins, density=True
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        def loss(params: tuple[float, float]) -> float:
            taus, weights = solver.two_atom_H(params)
            if np.any(taus <= 0):
                return 1e3
            rho_theory = solver.spectral_density(
                bin_centers, taus, weights, eps=4e-3
            )
            return float(np.sum((rho_theory - hist_counts) ** 2))

        res = minimize(
            loss,
            x0=list(self.x0),
            method="Nelder-Mead",
            options=dict(xatol=1e-3, fatol=1e-5, maxiter=200),
        )

        # Step 2: extract the theoretical spectrum edge
        taus_fit, w_fit = solver.two_atom_H(res.x)
        edges_fit = solver.find_edges(taus_fit, w_fit)

        if len(edges_fit) == 0:
            # Fallback: no edge detected (near-degenerate fit, typically at
            # small q) fall back to the naive Marchenko-Pastur upper edge.
            threshold = (1 + np.sqrt(q)) ** 2
        else:
            # edges_fit is sorted: take the upper bulk edge.
            threshold = edges_fit[1] if len(edges_fit) > 2 else edges_fit[-1]

        self.threshold_ = threshold

        # Step 3: clip eigenvalues (same logic as NaiveClippingEstimator)
        mask: np.ndarray = eigvals > threshold
        mean_noise: float = float(np.mean(eigvals[~mask]))
        return np.where(mask, eigvals, mean_noise)