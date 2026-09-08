from __future__ import annotations  # Pour pouvoir faire les -> Nom_de_la_Class

import numpy as np
from scipy.optimize import brentq


class SilverSteinEquationSolver:
    """
    Encapsulates the Silverstein (companion) equation machinery used to
    describe the limiting spectral distribution of a sample covariance
    matrix under a two-atom population spectral distribution H, for a
    fixed concentration ratio q. Used internally by AdaptedClippingEstimator
    to fit H and locate the corresponding theoretical spectrum edges.
    """

    def __init__(self, q: float, n_iter: int = 3000, tol: float = 1e-13, damping: float = 0.5):
        """
        Parameters
        ----------
        q : float
            Concentration ratio p / n.
        n_iter : int, default=3000
            Maximum number of fixed-point iterations used to solve for mbar(z).
        tol : float, default=1e-13
            Convergence tolerance on successive fixed-point iterates.
        damping : float, default=0.5
            Damping factor used to stabilize the fixed-point iteration.
        """
        self.q = q
        self.n_iter = n_iter
        self.tol = tol
        self.damping = damping

    @staticmethod
    def two_atom_H(params: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """
        Reparametrize a two-atom population spectral distribution H,
        guaranteeing w1 in (0, 1) and the trace constraint
        w1 * tau1 + w2 * tau2 = 1.

        Parameters
        ----------
        params : tuple[float, float]
            Raw parameters (w1, tau1) to be reparametrized.

        Returns
        -------
        taus : np.ndarray
            Array [tau1, tau2] of the two atom locations.
        weights : np.ndarray
            Array [w1, w2] of the two atom weights.
        """
        w1, tau1 = params
        w1 = np.clip(w1, 1e-3, 1 - 1e-3)
        tau1 = max(tau1, 1e-3)
        w2 = 1 - w1
        tau2 = (1 - w1 * tau1) / w2
        return np.array([tau1, tau2]), np.array([w1, w2])

    def underline_m(self, z: complex, taus: np.ndarray, weights: np.ndarray) -> complex:
        """
        Solve the Silverstein (companion) equation by fixed-point iteration:
        z = -1/mbar + q * sum_i w_i * tau_i / (1 + tau_i * mbar).

        Parameters
        ----------
        z : complex
            Point of the complex plane at which the companion Stieltjes
            transform is evaluated.
        taus : np.ndarray
            Atom locations of the population spectral distribution H.
        weights : np.ndarray
            Atom weights of the population spectral distribution H.

        Returns
        -------
        mbar : complex
            Companion Stieltjes transform mbar(z).
        """
        mbar = -1.0 / z
        for _ in range(self.n_iter):
            s = np.sum(weights * taus / (1 + taus * mbar))
            mbar_new = 1.0 / (-z + self.q * s)
            mbar_new = self.damping * mbar_new + (1 - self.damping) * mbar
            if abs(mbar_new - mbar) < self.tol:
                mbar = mbar_new
                break
            mbar = mbar_new
        return mbar

    def stieltjes_m(self, z: complex, taus: np.ndarray, weights: np.ndarray) -> complex:
        """
        Compute the Stieltjes transform m(z) of the spectrum of E, from the
        companion transform mbar(z).

        Parameters
        ----------
        z : complex
            Point of the complex plane at which the Stieltjes transform is evaluated.
        taus : np.ndarray
            Atom locations of the population spectral distribution H.
        weights : np.ndarray
            Atom weights of the population spectral distribution H.

        Returns
        -------
        m : complex
            Stieltjes transform m(z).
        """
        mbar = self.underline_m(z, taus, weights)
        return (mbar + (1 - self.q) / z) / self.q

    def spectral_density(self, x_vals: np.ndarray, taus: np.ndarray, weights: np.ndarray, eps: float = 2e-3) -> np.ndarray:
        """
        Compute the limiting spectral density
        rho(lambda) = (1/pi) * Im[m(lambda + i*eps)], with eps a small
        positive regularizer.

        Parameters
        ----------
        x_vals : np.ndarray
            Grid of points at which the spectral density is evaluated.
        taus : np.ndarray
            Atom locations of the population spectral distribution H.
        weights : np.ndarray
            Atom weights of the population spectral distribution H.
        eps : float, default=2e-3
            Small positive imaginary part used to approach the real axis.

        Returns
        -------
        rho : np.ndarray
            Spectral density evaluated on x_vals.
        """
        rho = np.empty_like(np.asarray(x_vals, dtype=float))
        for i, x in enumerate(x_vals):
            m = self.stieltjes_m(x + 1j * eps, taus, weights)
            rho[i] = max(m.imag / np.pi, 0.0)
        return rho

    def z_of_mbar(self, mbar: complex, taus: np.ndarray, weights: np.ndarray) -> complex:
        """
        Evaluate z(mbar), the inverse map of the companion Stieltjes transform.

        Parameters
        ----------
        mbar : complex
            Companion Stieltjes transform value.
        taus : np.ndarray
            Atom locations of the population spectral distribution H.
        weights : np.ndarray
            Atom weights of the population spectral distribution H.

        Returns
        -------
        z : complex
            Corresponding value of z.
        """
        return -1.0 / mbar + self.q * np.sum(weights * taus / (1 + taus * mbar))

    def zprime_of_mbar(self, mbar: complex, taus: np.ndarray, weights: np.ndarray) -> complex:
        """
        Evaluate the derivative z'(mbar), used to locate critical points
        (spectrum edges).

        Parameters
        ----------
        mbar : complex
            Companion Stieltjes transform value.
        taus : np.ndarray
            Atom locations of the population spectral distribution H.
        weights : np.ndarray
            Atom weights of the population spectral distribution H.

        Returns
        -------
        zprime : complex
            Derivative of z with respect to mbar, evaluated at mbar.
        """
        return 1.0 / mbar**2 - self.q * np.sum(weights * taus**2 / (1 + taus * mbar) ** 2)

    def find_edges(self, taus: np.ndarray, weights: np.ndarray, n_scan: int = 20000, mbar_range: float = 80) -> list[float]:
        """
        Find all spectrum edges by locating the roots of z'(mbar) = 0
        (critical points), avoiding the poles at mbar = -1/tau_i.

        Parameters
        ----------
        taus : np.ndarray
            Atom locations of the population spectral distribution H.
        weights : np.ndarray
            Atom weights of the population spectral distribution H.
        n_scan : int, default=20000
            Number of points used to scan the mbar-grid for sign changes.
        mbar_range : float, default=80
            Half-width of the mbar-grid used for the scan.

        Returns
        -------
        edges : list[float]
            Sorted list of spectrum edges z(mbar) at the detected critical points.
        """
        poles = sorted(set((-1.0 / taus).tolist() + [0.0]))
        grid = np.linspace(-mbar_range, mbar_range, n_scan)
        mask = np.ones_like(grid, dtype=bool)
        for p in poles:
            mask &= (np.abs(grid - p) > 1e-6)
        grid = grid[mask]
        vals = np.array([self.zprime_of_mbar(m, taus, weights) for m in grid])
        roots = []
        for i in range(len(grid) - 1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
                try:
                    roots.append(brentq(self.zprime_of_mbar, grid[i], grid[i + 1], args=(taus, weights)))
                except Exception:
                    pass
        return sorted(self.z_of_mbar(r, taus, weights) for r in roots)
