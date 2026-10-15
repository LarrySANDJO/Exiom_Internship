"""
Ce script contient la classe de l'estimateur Oracle (meme vecteurs propres que 
le sample correlation mais valeurs propres de la vraie matrice)
"""

from __future__ import annotations  # Pour pouvoir faire les -> Nom_de_la_Class

from models.data import DataClass
from models.rotation_invariant import RotationInvariantEstimators
from tools.tools import FloatArray, np


class OracleEstimator(RotationInvariantEstimators):
    """
    Estimateur oracle : garde les vecteurs propres empiriques de S, mais
    remplace les valeurs propres par celles de la vraie covariance
    True_Sigma a une mutiplication pres par les vecteurs propres de S. Ne représente pas un estimateur utilisable en pratique, True_Sigma
    n'est jamais connue sur des données réelles. Sert uniquement de
    référence théorique dans les simulations.    

    Args:
        RotationInvariantEstimators (_type_): classe des estimateurs invariants par rotation
        True_Sigma : ndarray de forme (p, p)
                La vraie matrice de covariance, connue seulement en simulation.
    """

    def __init__(self, True_Sigma, *, assume_centered: bool = False) -> None:
        super().__init__(assume_centered=assume_centered)
        self.True_Sigma = True_Sigma

    def transform_eigenvalues(self, eigvecs) -> FloatArray:
        """
        Ici, eigvals (de S) n'est pas utilisée : l'oracle ignore complètement
        le spectre empirique et retourne directement celui de True_Sigma (a une multipication pres par les vecteurs propres de S)
        On utilise donc juste eigvec

        Returns:
            FloatArray: Les valeurs propres de la vraie matrice de correlation
        """
        # eigvecs[:, i] est le i-eme vecteur propre (convention de eigh)
        return np.einsum("ij,ij->j", eigvecs, self.True_Sigma @ eigvecs)

    def fit(self, X: DataClass) -> OracleEstimator:
        """
        Dans la classe des estimateurs invariants par rotation, on calcule
        la matrice de correlation empirique et on garde ses valeurs propres et ses vecteurs propres
        
        Args:
            X (DataClass): Object de type DataClass qui contient
            les valeurs et vecteurs propres 
        """
        self.n_, self.p_ = X.n_, X.p_
        self.eigvals_, self.eigvecs_ = X.eigvals_, X.eigvecs_
        new_vals = self.transform_eigenvalues(X.eigvecs_)
        self.correlation_ = self._reconstruct(new_vals)
        
        return self


