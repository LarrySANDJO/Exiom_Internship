"""
Ce script contient la classe principale pour tous les estimateurs invariants par rotation
"""

from models.base import CorrelationEstimator
from tools.tools import FloatArray, normalise, np, sample_correlation_matrix
from abc import abstractmethod
from models.data import DataClass


class RotationInvariantEstimators(CorrelationEstimator):
    """
    Classe intermédiaire pour tous les estimateurs RIE (Rotation-Invariant
    Estimator — estimateur invariant par rotation) : SampleEstimator, Clipping,
    LinearShrinkage, NonLinearShrinkage et OracleRIE en héritent tous.
    Args:
        CorrelationEstimator (class): La classe de base de tout estimateur de matrice de correlation
    """
    
    def __init__(self, *, assume_centered: bool = False):
        super().__init__(assume_centered=assume_centered)
        self.eigvals_: FloatArray  = None
        self.eigvecs_: FloatArray = None

    @abstractmethod
    def transform_eigenvalues(self, eigvals: FloatArray) -> FloatArray:
        """
        La fonction f(λ) propre à chaque estimateur RIE : transforme les
        valeurs propres brutes de la matrice de correlation empirique S en valeurs propres "nettoyées". C'est
        la seule chose qu'une sous-classe concrète doit fournir, fit() et
        la reconstruction sont déjà gérées par cette classe.

        Args:
            eigvals (FloatArray): ndarray de forme (p,)
            Valeurs propres brutes de S (ordre croissant).

        Returns:
            FloatArray: ndarray de forme (p,)
            Valeurs propres transformées, dans le MÊME ordre que eigvals
            (pas de tri ni de réindexation : new_eigvals[i] correspond au
            même vecteur propre que eigvals[i]).
        """
        raise NotImplementedError

    def _reconstruct(self, new_vals: FloatArray) -> FloatArray:
        """
        On reconstruit l'estimateur V diag(new_vals) V.T, suivie d'une symétrisation numérique
        (M + M.T)/2 — nécessaire car les erreurs d'arrondi en virgule
        flottante peuvent casser la symétrie exacte de la matrice
        reconstruite, même quand la formule mathématique est symétrique.

        Args:
            new_vals (FloatArray): valeurs propres corrigées

        Returns:
            FloatArray: matrice estimée avec les valeurs propres corrigées
        """
        assert self.eigvecs_ is not None
        M = self.eigvecs_ @ np.diag(new_vals) @ self.eigvecs_.T
        return (M + M.T) / 2


    def fit(self, X: DataClass) -> RotationInvariantEstimators:
        """
        Dans la classe des estimateurs invariants par rotation, on calcule
        la matrice de correlation empirique et on garde ses valeurs propres et ses vecteurs propres
        
        Args:
            X (DataClass): Object de type DataClass qui contient
            les valeurs et vecteurs propres 
        """
        self.n_, self.p_ = X.n_, X.p_
        self.eigvals_, self.eigvecs_ = X.eigvals_, X.eigvecs_
        new_vals = self.transform_eigenvalues(self.eigvals_)
        self.correlation_ = self._reconstruct(new_vals)
        
        return self


