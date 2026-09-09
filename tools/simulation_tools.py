"""
Ce script contient les fonctions necessaires pour les simulations
Elles utilisent l'ensemble des classes crees dans models 
"""

import time

from models.adapted_clipping import *
from models.base import *
from models.data import *
from models.linear_shrinkage import *
from models.non_linear_shrinkage import *
from models.oracle import *
from models.rotation_invariant import *
from models.sample_correlation import *
from models.silverstein_solver import *
from models.simple_clipping import *
from tools.tools import *


def mc_matrix_parallel_sim(X :FloatArray, q :int, n :int, p :int, True_Sigma :FloatArray) -> tuple:
    """Cette fonction permet de faire une simulation monte carlo (ce qui va permettre de paralelliser)
       Elle gere uniquement une simulation 
    Args:
        X (FloatArray): matrice des donnees
        q (int): ratio de concentration
        n (int): n
        p (int): p
        True_Sigma (FloatArray): vraie matrice

    Returns:
        tuple: le tuple des resultats d'une simulation
    """
    data_object = DataClass()
    data_object.fit(X)

    S = SampleCorrelationEstimator().fit(data_object).correlation_
    # vals_vecs = diagmatrix(S)

    resultats = {}

    t0 = time.perf_counter()
    M = S
    resultats["Sample"] = (frobenius_error(M, True_Sigma), np.trace(M)/np.trace(True_Sigma), time.perf_counter()-t0)

    t0 = time.perf_counter()
    M = NaiveClippingEstimator().fit(data_object).correlation_
    resultats["Clipping"] = (frobenius_error(M, True_Sigma), np.trace(M)/np.trace(True_Sigma), time.perf_counter()-t0)

    t0 = time.perf_counter()
    M = AdaptedClippingEstimator().fit(data_object).correlation_
    resultats["Clipping adapted"] = (frobenius_error(M, True_Sigma), np.trace(M)/np.trace(True_Sigma), time.perf_counter()-t0)

    t0 = time.perf_counter()
    M = LinearShrinkageEstimator().fit(data_object).correlation_
    resultats["Linear"] = (frobenius_error(M, True_Sigma), np.trace(M)/np.trace(True_Sigma), time.perf_counter()-t0)

    t0 = time.perf_counter()
    M = NonLinearShrinkageEstimator().fit(data_object).correlation_
    resultats["NLS"] = (frobenius_error(M, True_Sigma), np.trace(M)/np.trace(True_Sigma), time.perf_counter()-t0)

    t0 = time.perf_counter()
    M = OracleEstimator(True_Sigma).fit(data_object).correlation_
    resultats["Oracle"] = (frobenius_error(M, True_Sigma), np.trace(M)/np.trace(True_Sigma), time.perf_counter()-t0)

    return resultats