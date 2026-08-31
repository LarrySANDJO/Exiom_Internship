"""
Ce script contient la classe de l'estimateur Oracle (meme vecteurs propres que 
le sample correlation mais valeurs propres de la vraie matrice)
"""

from models.rotation_invariant import RotationInvariantEstimators
from tools.tools import FloatArray, np


class OracleEstimator(RotationInvariantEstimators):
    """
    Estimateur oracle : garde les vecteurs propres empiriques de S, mais
    remplace les valeurs propres par celles de la vraie covariance
    True_Sigma. Ne représente pas un estimateur utilisable en pratique — True_Sigma
    n'est jamais connue sur des données réelles. Sert uniquement de
    référence théorique dans les simulations.    

    Args:
        RotationInvariantEstimators (_type_): classe des estimateurs invariants par rotation
        True_Sigma : ndarray de forme (p, p)
                La vraie matrice de covariance, connue seulement en simulation.
    """

    def __init__(self, True_Sigma, *, assume_centered: bool = False) -> None:
        super().__init__(assume_centered)
        self.True_Sigma = True_Sigma

    def transform_eigenvalues(self) -> FloatArray:
        """
        Ici, eigvals (de S) n'est pas utilisée : l'oracle ignore complètement
        le spectre empirique et retourne directement celui de True_Sigma

        Returns:
            FloatArray: Les valeurs propres de la vraie matrice de correlation
        """
        return np.linalg.eigvalsh(self.True_Sigma)


