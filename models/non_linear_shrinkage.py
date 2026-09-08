"""
Ce fichier comporte la classe NonLinearShrinkageEstimator pour le shrinkage non lineaire
"""

from __future__ import annotations  # Pour pouvoir faire les -> Nom_de_la_Class

from models.rotation_invariant import RotationInvariantEstimators
from tools.tools import *


class NonLinearShrinkageEstimator(RotationInvariantEstimators):

    def transform_eigenvalues(self):
        pass

    def fit(self, X:FloatArray) -> NonLinearShrinkageEstimator:
        vals, vecs = X.eigvals_, X.eigvecs_
        n, p = X.n_, X.p_
        h = n**(-1/3)
 
        # Valeurs propres non nulles (seuil numérique)
        # eigh retourne en ordre croissant -> les nulles sont en premier
        mask = vals > 1e-10
        lambda_ = vals[mask]     # m valeurs propres non nulles
        m = len(lambda_)
        u  = vecs[:, mask]       # (p x m) vecteurs propres non nuls
        u0 = vecs[:, ~mask]      # (p x (p-m)) vecteurs propres nuls
    
        # Matrices pour calcul vectorisé — traduction directe du MATLAB LW2020
        L  = lambda_[:, None]    # (m, 1)
        Lj = lambda_[None, :]    # (1, m)
        H  = h * Lj              # bande passante locale h_j = lambda_j * h
        x  = (L - Lj) / H       # (m, m)
    
        # Densité spectrale — Eq. (4.7)
        ftilde = (3/(4*np.sqrt(5))) * np.mean(
            np.maximum(1 - x**2/5, 0) / H, axis=1
        )
    
        # Transformée de Hilbert — Eq. (4.8)
        Hftemp = (-3/(10*np.pi)) * x + \
            (3/(4*np.sqrt(5)*np.pi)) * (1 - x**2/5) * \
            log_stable(x)  # log_stable(x)  np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
        Hftemp[np.abs(x) == np.sqrt(5)] = (-3/(10*np.pi)) * x[np.abs(x) == np.sqrt(5)]
        Hftilde = np.mean(Hftemp / H, axis=1)
    
        if p < n:
            denom = (np.pi*(p/n)*lambda_*ftilde)**2 + (1-(p/n)-np.pi*(p/n)*lambda_*Hftilde)**2
    
            # denom = [max(d, 1e-6) for d in denom] # Pour eviter de diviser par zero
            dtilde = lambda_ / denom
            M = u @ np.diag(dtilde) @ u.T
    
        elif p > n:
            # Eq. (C.8) — Hilbert en 0
            Hftilde0 = (1/np.pi) * (
                3/(10*h**2)
            + (3/(4*np.sqrt(5)*h)) * (1 - 1/(5*h**2))
                * np.log((1 + np.sqrt(5)*h) / max((1 - np.sqrt(5)*h), 0.01))
            ) * np.mean(1/lambda_)
    
            dtilde0 = 1 / (np.pi * (p-m)/n * Hftilde0)  
            dtilde1 = lambda_ / (                       
                np.pi**2 * lambda_**2 * (ftilde**2 + Hftilde**2)
            )
            # Reconstruction séparée : espace nul + espace signal
            M = u0 @ np.diag(dtilde0 * np.ones(p-m)) @ u0.T + u  @ np.diag(dtilde1) @ u.T
        else:
            raise ValueError(
                "Le Shrinkage non lineaire n'est pas definie pour un ratio de concentration egal a 1"
            )
        self.correlation_ = (M + M.T) / 2
        return self