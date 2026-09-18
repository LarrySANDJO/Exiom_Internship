# Exiom_Internship — Estimation de matrices de corrélation en grande dimension

Stage chez Exiom Partners : estimation et régularisation de matrices de corrélation en grande dimension (théorie des matrices aléatoires), avec application à l'optimisation de portefeuille minimum-variance (GMV).

## Contexte

Quand le nombre d'actifs `p` devient comparable au nombre d'observations `n` (ratio de concentration `q = p/n` non négligeable), la matrice de corrélation empirique S devient un estimateur bruité et mal conditionné de la vraie corrélation. Ce repo implémente et compare plusieurs familles d'estimateurs conçus pour corriger ce biais, en s'appuyant sur les vecteurs propres de S mais en nettoyant ses valeurs propres.

## Structure du repo

```
Exiom_Internship/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── data_importation.py       # téléchargement + nettoyage des données S&P 500 (yfinance)
│   ├── raw/                      # CSV/XLSX bruts (constituants, prix, description)
│   └── clean/                    # données nettoyées
├── documentation/
│   └── Sujets_de_stage_20252026.pdf
├── models/                       # classes des estimateurs
│   ├── base.py                   # CorrelationEstimator (classe abstraite)
│   ├── data.py                   # DataClass (calcule tout ce dont les estimateurs ont besoin)
│   ├── rotation_invariant.py     # RotationInvariantEstimators (classe intermédiaire)
│   ├── sample_correlation.py     # SampleCorrelationEstimator
│   ├── simple_clipping.py        # NaiveClippingEstimator
│   ├── adapted_clipping.py       # AdaptedClippingEstimator
│   ├── silverstein_solver.py     # SilverSteinEquationSolver (utilisé par AdaptedClipping)
│   ├── linear_shrinkage.py       # LinearShrinkageEstimator
│   ├── non_linear_shrinkage.py   # NonLinearShrinkageEstimator
│   └── oracle.py                 # OracleEstimator
├── tools/
│   ├── tools.py                  # utilitaires mathématiques génériques
│   └── simulation_tools.py       # mc_matrix_parallel_sim (lance les 6 estimateurs sur une simulation)
├── notebooks/
│   ├── robust_gaussian_case.ipynb
│   ├── robust_fat_tails_case.ipynb
│   └── portfolio_optimisation.ipynb
├── results/                      # (actuellement : copies des notebooks, voir Limites connues)
└── tests/
```

## Architecture des classes

### `CorrelationEstimator` (`models/base.py`)
Classe abstraite (ABC) commune à tous les estimateurs. Impose une méthode `fit(X)` et expose le résultat dans l'attribut `correlation_`. Fournit `_check_is_fitted()` (lève `NotFittedError` si on accède au résultat avant `fit()`) et `_error_to(True_Sigma)` (erreur de Frobenius contre une référence).

### `DataClass` (`models/data.py`)
Point d'entrée de toute la pipeline : `DataClass().fit(X)` prend les données brutes `X` (n, p) et calcule une fois pour toutes tout ce dont les estimateurs auront besoin, la matrice de corrélation empirique S, sa décomposition en valeurs propres/vecteurs propres (`eigvals_`, `eigvecs_`), et les trois quantités nécessaires au shrinkage linéaire (`m_linear_shrinkage`, `beta_linear_shrinkage`, `d_linear_shrinkage`, au sens de Ledoit-Wolf 2004). **Tous les estimateurs consomment un `DataClass` déjà fitté, jamais les données brutes directement.**

### `RotationInvariantEstimators` (`models/rotation_invariant.py`)
Hérite de `CorrelationEstimator`. Implémente `fit(X: DataClass)` une seule fois pour toute la famille RIE (Rotation-Invariant Estimator) : récupère `eigvals_`/`eigvecs_` depuis le `DataClass`, appelle `transform_eigenvalues()` (la seule méthode que chaque sous-classe doit fournir), puis reconstruit la matrice via `_reconstruct()`.

### Les six estimateurs concrets

| Classe | Fichier | Principe |
|---|---|---|
| `SampleCorrelationEstimator` | `sample_correlation.py` | f(λ)=λ, aucune transformation — la corrélation empirique brute |
| `NaiveClippingEstimator` | `simple_clipping.py` | Nettoyage Marchenko-Pastur (Laloux et al., 1999) : valeurs propres sous le bord théorique remplacées par leur moyenne |
| `AdaptedClippingEstimator` | `adapted_clipping.py` | Version data-driven du clipping : ajuste une distribution spectrale à deux atomes (équation de Silverstein) à l'histogramme empirique pour situer le bord de coupure, au lieu d'utiliser le bord théorique de Marchenko-Pastur |
| `LinearShrinkageEstimator` | `linear_shrinkage.py` | Shrinkage linéaire de Ledoit-Wolf (2004) : Σ̂ = (1-ρ)S + ρμI |
| `NonLinearShrinkageEstimator` | `non_linear_shrinkage.py` | Shrinkage non linéaire asymptotiquement optimal de Ledoit-Wolf (2020), par estimation de la densité spectrale et de la transformée de Hilbert |
| `OracleEstimator` | `oracle.py` | Garde les vecteurs propres empiriques mais utilise les vraies valeurs propres de `True_Sigma` : borne théorique inférieure d'erreur, jamais utilisable en pratique |

`SilverSteinEquationSolver` (`silverstein_solver.py`) n'est pas un estimateur : c'est le solveur numérique (itération à point fixe + recherche de racines) utilisé en interne par `AdaptedClippingEstimator` pour résoudre l'équation de Silverstein et localiser les bords du spectre théorique.

## Installation

```bash
pip install -r requirements.txt
```

Le repo n'étant pas packagé (pas de `setup.py`/`pyproject.toml` à la racine), les imports (`from models... import`, `from tools... import`) supposent que la racine du repo est sur le `PYTHONPATH`. Les notebooks gèrent ça via `%run setup.py` en première cellule (voir `notebooks/setup.py`) ; pour un script autonome, lancez-le depuis la racine avec `PYTHONPATH=.`.

## Utilisation — exemple minimal

```python
from models.data import DataClass
from models.sample_correlation import SampleCorrelationEstimator
from models.simple_clipping import NaiveClippingEstimator
from models.linear_shrinkage import LinearShrinkageEstimator
from models.non_linear_shrinkage import NonLinearShrinkageEstimator
from models.oracle import OracleEstimator
import numpy as np

X = np.random.normal(size=(200, 50))   # n=200 observations, p=50 variables

# Étape obligatoire : calculer une fois les quantités partagées
data = DataClass().fit(X)

# Chaque estimateur consomme ensuite ce même objet DataClass
S_sample = SampleCorrelationEstimator().fit(data).correlation_
S_clip   = NaiveClippingEstimator().fit(data).correlation_
S_lin    = LinearShrinkageEstimator().fit(data).correlation_
S_nls    = NonLinearShrinkageEstimator().fit(data).correlation_

True_Sigma = np.eye(50)  # seulement en simulation
S_oracle = OracleEstimator(True_Sigma).fit(data).correlation_
```

Pour comparer les six estimateurs sur une seule simulation d'un coup :

```python
from tools.simulation_tools import mc_matrix_parallel_sim

resultats = mc_matrix_parallel_sim(X, q=50/200, n=200, p=50, True_Sigma=True_Sigma)
# resultats est un dict {nom_estimateur: (erreur_frobenius, ratio_trace, temps)}
```

## Notebooks

- **`robust_gaussian_case.ipynb`** : simulation Monte Carlo sur données gaussiennes, `True_Sigma` estimée par blocs à partir de vraies données S&P 500, comparaison des 6 estimateurs sur une grille de ratios de concentration γ.
- **`robust_fat_tails_case.ipynb`** : même protocole, données simulées avec une loi de Student (ν=3 degrés de liberté) pour tester la robustesse aux queues épaisses.
- **`portfolio_optimisation.ipynb`** : application au portefeuille minimum-variance (GMV) sur vraies données S&P 500, avec split train/test, comparaison in-sample/out-of-sample des variances et ratios de Sharpe.
