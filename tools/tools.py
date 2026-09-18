"""
models/tools.py
================
 
Ici on a l'ensemble des utilitaires mathématiques génériques utilisés par les estimateurs de
correlation (base.py et sous-dossiers de models/) mais aussi, plus tard, par
d'autres briques du projet qui n'héritent pas forcément de CorrelationEstimator.
 
Fonctions
---------
sample_correlation_matrix(X, corrected=False)
    correlation empirique
center(X, assume_centered=False)
    Centrage des données, colonne par colonne.
normalise(X, assume_centered=False)
    Centrage + réduction (matrice de corrélation en sortie de S).
frobenius_norm(A)
    Norme de Frobenius d'une matrice carrée, normalisée par p.
frobenius_error(A, B)
    Erreur de Frobenius entre deux matrices : frobenius_norm(A - B).
"""


import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

# Alias de type : tableau numpy de flottants, de dimension quelconque
FloatArray = NDArray[np.floating]


def sample_correlation_matrix(X: FloatArray, *, corrected: bool = False) -> FloatArray:
    """
    Matrice de correlation empirique S à partir des données X (n, p).
    Pour avoir un estimateur non biaisé, deux cas sont possibles:
        - corrected=False : la matrice est dejà corrigée donc on divise par n
        - corrected=True  : on corrige nous meme la matrice donc on divise par n-1
 
    X doit être centré (moyenne nulle par colonne) au préalable 
 
    Parameters
    ----------
    X : ndarray de forme (n, p)
    corrected : bool, défaut False
 
    Returns
    -------
    S : ndarray de forme (p, p)
    """
    n: int = X.shape[0]
    denom: int = (n - 1) if corrected else n
    return X.T @ X / denom


def center(X: FloatArray, *, assume_centered: bool = False) -> FloatArray:
    """
    Centre les données X colonne par colonne :
     
        X_centré[:, j] = X[:, j] - moyenne(X[:, j])
     
    sauf si assume_centered=True, auquel cas X est renvoyé tel quel : on
    suppose alors que la vraie moyenne mu est déjà nulle.

    Args:
        X (FloatArray): La matrice des données à centrer
        assume_centered (bool, optional): Defaults to False.

    Returns:
        FloatArray: La matrice centrée
    """
    if assume_centered:
        return X
    centered: FloatArray = X - np.mean(X, axis=0)
    return centered


def normalise(X: FloatArray, *, assume_centered: bool = False) -> FloatArray:
    """
    Centre PUIS réduit les données X colonne par colonne, la sortie de
    sample_correlation_matrix() sur ce résultat est donc une matrice de
    CORRÉLATION (diagonale ≈ 1), pas une correlation à l'échelle d'origine :
 
        X_normalisé[:, j] = (X[:, j] - moyenne(X[:, j])) / écart-type(X[:, j])
 
    assume_centered contrôle uniquement l'étape de centrage (cf. center()) :
        - False (défaut) : centre avec la moyenne empirique, puis divise par
          l'écart-type calculé sur les données déjà centrées (np.std usuel).
        - True : ne centre pas (mu supposée nulle), et calcule l'échelle
          directement sur X brut, sqrt(moyenne(X**2)), pour ne pas
          réintroduire une moyenne qu'on a choisi d'ignorer.
 
    NB — convention actuelle du projet : tous les estimateurs utilisent
    cette fonction, donc travaillent en corrélation.

    Args:
        X (FloatArray): La matrice des données à normaliser
        assume_centered (bool, optional): Defaults to False.

    Returns:
        FloatArray: La matrice normalisée
    """
    X_centered = center(X, assume_centered=assume_centered)
    scale: FloatArray
    if assume_centered:
        scale = np.sqrt(np.mean(X**2, axis=0))
    else:
        scale = np.std(X_centered, axis=0)
    return X_centered / scale


def frobenius_scalar_product(A: FloatArray, B: FloatArray) -> float:
    """ produit scalaire de Frobenius normalisée entre deux matrices carrées A et B de taille (p, p) : 
        La division par p (absente de la définition usuelle de la norme de
        Frobenius) rend l'erreur comparable entre différentes valeurs de p :
        c'est une convention cohérente avec Ledoit-Wolf (2004, 2020).
    Args:
        A (FloatArray): _description_
        B (FloatArray): _description_
    Raises:
        ValueError: Les matrices doivent etre bien definies

    Returns:
        float: ...
    """
    if A.shape != B.shape:
        raise ValueError("Les matrices carrees et doivent etre de tailles correctes")

    p: int = A.shape[0]
    return np.trace(A @ B.T) / p


def frobenius_norm(A: FloatArray) -> float:
    """
    Norme de Frobenius normalisée d'une matrice carrée A (p, p) : 
    La division par p (absente de la définition usuelle de la norme de
    Frobenius) rend l'erreur comparable entre différentes valeurs de p :
    c'est une convention cohérente avec Ledoit-Wolf (2004, 2020).

    Args:
        A (FloatArray): La matrice dont on veut calculer la norme

    Returns:
        float: La norme de la matrice A
    """
    return np.sqrt(frobenius_scalar_product(A, A))


def frobenius_error(A: FloatArray, B: FloatArray) -> float:
    """
    Erreur de Frobenius normalisée entre deux matrices A et B (p, p) : 

    Args:
        A (FloatArray)
        B (FloatArray): 

    Returns:
        float
    """
    return frobenius_norm(A - B)


# Fonctions pour le calcul parralelise de beta_linear_shrinkage dans la classe Data
def terme_k(xk, S, p) -> float:
    """cette fonction sert a calculer un terme de la somme du calcul de beta_linear_shrinkage

    Args:
        xk (FloatArray): la ligne k de la matrice X des donnees
        S (FloatArray): la correlation empirique
        p (int): le nombre de variables

    Returns:
        float: le terme k de la somme du calcul de beta_linear_shrinkage
    """
    M = np.outer(xk, xk) - S
    return np.sum(M**2) / p


def beta_linear_shrinkage_parralel(X:FloatArray, S:FloatArray, n:int, p:int, n_jobs:int = -2) -> float:
    """Fonction pour calculer beta_ls par parralelisatio joblib

    Args:
        X (FloatArray): _description_
        S (FloatArray): _description_
        n (int): _description_
        p (int): _description_
        n_jobs (int, optional): _description_. Defaults to -2.

    Returns:
        float: beta_ls
    """
    
    termes = Parallel(n_jobs=n_jobs)(
        delayed(terme_k)(X[k], S, p)
        for k in range(n)
    )
    return sum(termes) / n**2

# Fonction log_stable pour le calcul du logarithme


def log_stable(x: float, a: float = np.sqrt(5), seuil: float = 1e3) -> float:
    """_summary_

    Args:
        x (float): _description_
        a (float, optional): _description_. Defaults to np.sqrt(5).
        seuil (float, optional): _description_. Defaults to 1e3.

    Returns:
        float: _description_
    """
    out = np.empty_like(x)
    small = np.abs(x) <= seuil
    large = ~small
    out[small] = np.log(np.abs((a - x[small]) / (a + x[small])))
    with np.errstate(divide="ignore"):
        out[large] = np.log1p(-a / x[large]) - np.log1p(a / x[large])
    return out