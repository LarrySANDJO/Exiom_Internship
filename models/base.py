"""
Ce script contient la classe principale CorrelationEstimator pour tous les estimateurs
"""

# Importation des librairies necessaires
from __future__ import annotations  # Pour pouvoir faire les -> Nom_de_la_Class

from abc import ABC, abstractmethod
from typing import Self

from models.data import DataClass
from tools.tools import *


class NotFittedError(RuntimeError):
    """
    Levée quand on essaie d'accéder à un résultat (correlation_, error_to, ...)
    avant d'avoir appelé fit(X). Inspiré de scikit-learn.
    """
    pass


class CorrelationEstimator(ABC):
    """
    Classe principale CorrelationEstimator qui herite de la classe ABC notamment de @abstractmethod
    """
    
    def __init__(self, *, assume_centered: bool = False) -> CorrelationEstimator:
        self.assume_centered = assume_centered
        self.correlation_: FloatArray = None
        self.n_: int = None
        self.p_: int = None

    @abstractmethod
    def fit(self, X:DataClass) -> Self:
        """
        Estime la matrice de correlation à partir des données X.
 
        Args:
            X (DataClass): objet de type DataClass qui contient toutes 
            les infos necessaires sur les donnees
            n observations (lignes) de p variables (colonnes).

        Returns:
            Self: L'instance elle-même (le type exact de la sous-classe, par exemple
            SampleCorrelation)
        """
        raise NotImplementedError

    # @staticmethod
    # def _check_input(X:FloatArray) -> FloatArray:
    #     """
    #     Vérifie que X est bien un ndarray 2D (n, p) avec n >= 2, sinon lève
    #     une erreur explicite plutôt.

    #     Args:
    #         X (FloatArray): Matrice des données

    #     Returns:
    #         FloatArray: Matrice des données verifiée
    #     """
    #     X = np.asarray(X)
    #     if X.ndim != 2:
    #         raise ValueError(
    #             f"X doit être de forme (n, p); reçu un tableau de "
    #             f"dimension {X.ndim}.")
    #     if X.shape[0] < 2:
    #         raise ValueError("X doit contenir au moins 2 observations (lignes) pour "
    #             "calculer une correlation.")

    #     return X

    # def _prepare(self, X:FloatArray) -> FloatArray:
    #     """
    #     Méthode utilitaire à appeler en tout début de fit() par les
    #     sous-classes : vérifie X, enregistre n_ et p_, et retourne X vérifié.

    #     Args:
    #         X (FloatArray): Matrice des données

    #     Returns:
    #         Self: Matrice des données preparée
    #     """
    #     X = self._check_input(X)
    #     self._n, self._p = X.shape
    #     return X

    def _check_is_fitted(self) -> None:
        """
        Vérifie que fit() a déjà été appelé, sinon lève une erreur explicite.
        """
        if self.correlation_ is None:
            raise NotFittedError(
                f"{self.__class__.__name__} : fit(X) doit être appelé avant "
                "d'accéder au résultat."
            )

    def _error_to(self, True_Sigma: FloatArray) -> float:
        """
        Erreur de Frobenius entre l'estimateur courant et une
        matrice de référence True_Sigma.

        Args:
            True_Sigma (FloatArray): vrai matrice de corellation 
            à laquelle on compare l'estimateur

        Returns:
            float: L'erreur de frobenius entre la vraie matrice et l'estimateur
        """
        self._check_is_fitted()
        return frobenius_error(self.correlation_, True_Sigma)


