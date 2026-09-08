"""
Tout par de classe data où l'on calcule directement les inputs essentiels pour les estimateurs.
NB : ls = linear shrinkage
"""
from __future__ import annotations  # Pour pouvoir faire les -> Nom_de_la_Class

import numpy as np

from tools.tools import *


class DataClass:

    def __init__(self, *, assume_centered: bool = False):
        self.assume_centered: bool = assume_centered
        self.eigvals_: FloatArray = None
        self.eigvecs_: FloatArray = None
        self.n_: int = None
        self.p_: int = None
        # ici j'ajoute le bn carre du shrinkage lineaire pour ne pas avoir a le recalculer
        self.beta_linear_shrinkage: float = None
        # Pareil pour le m
        self.m_linear_shrinkage: float = None
        # ... d
        self.d_linear_shrinkage: float = None

    @staticmethod
    def _check_input(X: FloatArray) -> FloatArray:
        """
        Vérifie que X est bien un ndarray 2D (n, p) avec n >= 2, sinon lève
        une erreur explicite plutôt.

        Args:
            X (FloatArray): Matrice des données

        Returns:
            FloatArray: Matrice des données verifiée
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(
                f"X doit être de forme (n, p); reçu un tableau de "
                f"dimension {X.ndim}.")
        if X.shape[0] < 2:
            raise ValueError("X doit contenir au moins 2 observations (lignes) pour "
                "calculer une correlation.")

        return X

    def _prepare(self, X: FloatArray) -> FloatArray:
            """
            Méthode utilitaire à appeler en tout début de fit() par les
            sous-classes : vérifie X, enregistre n_ et p_, et retourne X vérifié.
    
            Args:
                X (FloatArray): Matrice des données
    
            Returns:
                Self: Matrice des données preparée
            """
            X = self._check_input(X)
            self.n_, self.p_ = X.shape
            return X
         

    def fit(self, X:FloatArray) -> DataClass:
        X = self._prepare(X)
        sample_matrix = sample_correlation_matrix(normalise(X, assume_centered = self.assume_centered))
        self.eigvals_, self.eigvecs_ = np.linalg.eigh(sample_matrix)
        # beta_ls
        self.beta_linear_shrinkage = beta_linear_shrinkage_parralel(X, sample_matrix, self.n_, self.p_)
        self.m_linear_shrinkage = frobenius_scalar_product(sample_matrix, np.eye(self.p_))
        self.d_linear_shrinkage = frobenius_norm(sample_matrix - self.m_linear_shrinkage * np.eye(self.p_)) ** 2
        return self