"""
Ce fichier comporte la classe LinearShrinkageEstimator pour le shrinkage lineaire
"""

from models.rotation_invariant import RotationInvariantEstimators
from models.data import DataClass
from tools.tools import *


class LinearShrinkageEstimator(RotationInvariantEstimators):


    # def __init__(self, *, assume_centered: bool = False):
    #     super().__init__()


    def fit(self, X: DataClass) -> RotationInvariantEstimators:
        """
        Dans la classe des estimateurs invariants par rotation, on calcule
        la matrice de correlation empirique et on garde ses valeurs propres et ses vecteurs propres
        
        Args:
            X (DataClass): Object de type DataClass qui contient
            les valeurs et vecteurs propres 
        """
        self.n_, self.p_ = X.n_, X.p_
        m = X.m_linear_shrinkage
        b2 = X.beta_linear_shrinkage
        d2 = X.d_linear_shrinkage
        b2 = min(b2, d2)
        a2 = d2 - b2
        self.eigvals_ = (b2 / d2) * m * np.ones(self.p_) + (a2 / d2) * X.eigvals_
        self.eigvecs_ = X.eigvecs_
        self.correlation_ = self._reconstruct(self.eigvals_)
        
        return self

    def transform_eigenvalues(self, eigvals: FloatArray) -> FloatArray:
        pass