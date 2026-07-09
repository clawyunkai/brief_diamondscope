# =============================================================
# app.py — DiamondScope  (squelette à compléter — tu es libre)
# =============================================================
# Construis une app Streamlit à 3 pages :
#   - prédire le PRIX d'un diamant (régression)
#   - prédire si la coupe est PREMIUM (classification)
#   - une page d'ANALYSE rédigée
#
# Place tes fichiers .pkl (sauvegardés dans ton notebook) dans CE dossier.
# Lance avec :  streamlit run app.py
# =============================================================

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, root_mean_squared_error,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# TODO 1 - Charge tes modèles sauvegardés (régression, classification, ET le scaler)
#   Utilise : pickle.load(open("modele_regression.pkl", "rb"))  (idem pour les autres)


# TODO 2 - Affiche un titre pour l'application
#   Utilise : st.title()


# TODO 3 - Crée une navigation entre les 3 pages, et affiche un contenu différent
#          selon la page choisie (structure if / elif / else)
#   Utilise : st.sidebar.radio(...) avec ["Prédire le prix", "Coupe premium", "Analyse"]


# =============================================================
# PAGE "Prédire le prix"  (régression)
# =============================================================

# TODO 4 - Un widget de saisie par feature de TON modèle de régression
#   Utilise : st.number_input()  ou  st.slider()

# TODO 5 - Un bouton "Estimer le prix"
#   Utilise : st.button()

# TODO 6 - Au clic : construis un DataFrame d'UNE ligne avec exactement les
#          mêmes colonnes qu'à l'entraînement, prédis, puis affiche le prix
#   Utilise : pd.DataFrame(...) , model.predict(...) , st.metric()


# =============================================================
# PAGE "Coupe premium"  (classification)
# =============================================================

# TODO 7 - Les widgets de saisie pour les features de TON modèle de classification

# TODO 8 - Applique le MÊME scaler qu'à l'entraînement sur la ligne saisie
#   Utilise : scaler.transform()

# TODO 9 - Prédis puis affiche "premium" ou "non premium"
#   Utilise : model.predict()  ,  st.write()


# =============================================================
# PAGE "Analyse"
# =============================================================

# TODO 10 - Rédige 10 à 15 lignes : features les plus influentes sur le prix,
#           fiabilité du modèle de classification, limites du prototype
#   Utilise : st.write()  ou  st.markdown()

# TODO 11 - (bonus) Affiche un graphique utile (heatmap ou matrice de confusion)
#   Utilise : st.pyplot()
