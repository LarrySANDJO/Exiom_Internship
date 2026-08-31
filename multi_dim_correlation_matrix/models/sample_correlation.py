"""
Ce fichier contient la classe SampleCorrelationEstimator de l'estimateur empirique de matrice de correlation
"""

from models.rotation_invariant import RotationInvariantEstimators
from tools.tools import FloatArray



class SampleCorrelationEstimator(RotationInvariantEstimators):
    """
    Covariance empirique S, en tant qu'estimateur RIE avec f(λ) = λ.

    Args:
        RotationInvariantEstimators (
        _type_): classe des estimateurs invariants par rotation
    """

    def __init__(self, *, assume_centered: bool = False):  
        super().__init__(assume_centered=assume_centered)
        self.eigvals_: FloatArray = None
        self.eigvecs_: FloatArray = None


    def transform_eigenvalues(self, eigvals: FloatArray) -> FloatArray:
        return eigvals.copy()

    


