"""
Ce script permet de télécharger via yahoo finance des données d'un actfi de référence
Il est conçu pour le cas du S&P500
"""

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

ANNEES = 15 # On prend par défaut les quinze années précédentes
INDEX = "SP500"
N_ACTIONS = 500 


end_date = datetime(2026, 7, 1).strftime("%Y-%m-%d") # On prend comme date par défaut le premier juillet 2026
start_date = (datetime(2026, 7, 1) - timedelta(days=ANNEES * 365)).strftime("%Y-%m-%d")

# On récupère la liste des constituant du S&P500 (le lien est à remplacer pour un autre ETF) avec leur secteurs

sp500_raw_description_data = pd.read_csv("http://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")
sp500_raw_description_data["Symbol"] = sp500_raw_description_data["Symbol"].str.replace(".", "-", regex=False)

# exportation vers le dossier data
sp500_raw_description_data.to_csv("data/raws/sp500_raw_description_data.csv", sep=",")

# Importation des donnees brutes
secteurs = sp500_raw_description_data["GICS Sector"].dropna().unique()
tickers = sp500_raw_description_data["Symbol"].dropna().unique().tolist()

sp500_raw_data = yf.download(tickers, 
                             start_date, 
                             end_date, 
                             auto_adjust=True,        # auto_adjust=True pour ajuster les dividendes et les splits 
                             progress=True,)["Close"] # progress=True pour afficher une barre de progression



# exportation vers le dossier data
sp500_raw_data.to_csv("data/raws/sp500_raw_data.csv", sep=",")

# Nettoyage des donnees
seuil = 0.1             # seuil de tolerance des donnees manquantes pour les colonnes
# nettoyage des colonnes
sp500_clean_data = sp500_raw_data.dropna(axis=1, thresh = int((1 - seuil) * len(sp500_raw_data)))

# nettoyage des lignes
sp500_clean_data = sp500_clean_data.dropna(axis=0)

# exportation vers le dossier data
sp500_clean_data.to_csv("data/clean/spsp500_clean_data.csv", sep=",")




