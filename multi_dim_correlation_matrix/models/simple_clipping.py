"""
Ce script contient la classe pour l'estimateur par la methode de clipping simple 
"""

from models.rotation_invariant import RotationInvariantEstimators
from tools.tools import np


class NaiveClippingEstimator(RotationInvariantEstimators):
    """
    Estimateur de clipping : les valeurs propres au-delà
    du bord de Marchenko-Pastur sont gardées telles quelles (signal), les
    autres sont remplacées par leur moyenne commune (bruit, aplati à une
    valeur constante qui préserve la trace).
    """

    def transform_eigenvalues(self, eigvals):
        assert self.p_ is not None and self.n_ is not None
        q: float = self.p_ / self.n_
        lambda_max: float = (1 + np.sqrt(q))**2
        mask: bool = eigvals >lambda_max
        mean_bad: float = np.mean(eigvals[~mask])
        return np.where(mask, eigvals, mean_bad)

