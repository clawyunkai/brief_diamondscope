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
import joblib

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
# apparemment, c'est mieux de réutiliser joblib

@st.cache_resource
def charger_modeles():
    model_class=joblib.load("model_class.pkl")
    scaler_class=joblib.load("scaler_class.pkl")
    model_price=joblib.load("model_price.pkl")
    scaler_price=joblib.load("scaler_price.pkl")

    return model_class, scaler_class, model_price, scaler_price

model_class, scaler_class, model_price, scaler_price = charger_modeles()

# TODO 2 - Affiche un titre pour l'application
st.title("💎 DiamondScope : L'Analyse et la Prédiction des Diamants")


# TODO 3 - Crée une navigation entre les 3 pages, et affiche un contenu différent
#          selon la page choisie (structure if / elif / else)
#   Utilise : st.sidebar.radio(...) avec ["Prédire le prix", "Coupe premium", "Analyse"]
page_choisie = st.sidebar.radio(
    "Navigation :", 
    ["Prédire le prix", "Prédire la coupe premium", "Analyse"]
)

# =============================================================
# PAGE "Prédire le prix"  (régression)
# =============================================================
if page_choisie == "Prédire le prix":
    st.title("💰 Prédire le prix d'un diamant")
    st.write("C'est ici que tu vas intégrer ton modèle de régression linéaire (`model_price`).")

# TODO 4 - Un widget de saisie par feature de TON modèle de régression
#   Utilise : st.number_input()  ou  st.slider()
    carat = st.number_input("Carat (Poids)", min_value=0.1, max_value=10.0, value=0.5, step=0.01)

    st.divider()

# TODO 5 - Un bouton "Estimer le prix"
#   Utilise : st.button()
    if st.button("Estimer le prix", type="primary"):

        df_input = pd.DataFrame({
                "carat": [carat]
            })
            
            # Affichage du tableau saisi pour contrôle
        st.write("Vos données :")
        st.dataframe(df_input)
    # TODO 6 - Au clic : construis un DataFrame d'UNE ligne avec exactement les
    #          mêmes colonnes qu'à l'entraînement, prédis, puis affiche le prix
    #   Utilise : pd.DataFrame(...) , model.predict(...) , st.metric()
        df_input_sc=scaler_price.transform(df_input)

        prix_pred = np.expm1(model_price.predict(df_input_sc))
        st.metric(label="Prix estimé (en $)", value=f"{prix_pred[0]:,.2f}")

# =============================================================
# PAGE "Coupe premium"  (classification)
# =============================================================
elif page_choisie == "Prédire la coupe premium":
    st.title("💎 Classification de la coupe")
# TODO 7 - Les widgets de saisie pour les features de TON modèle de classification

    col1, col2 = st.columns(2)

    with col1:
        carat = st.number_input("Carat (Poids)", min_value=0.1, max_value=10.0, value=0.5, step=0.01)
        color = st.selectbox("Color (Couleur)", ["J", "I", "H", "G", "F", "E", "D"])

    with col2:
        clarity = st.selectbox("Clarity (Pureté)", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
        depth = st.number_input("Depth (Profondeur %)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
        table = st.number_input("Table", min_value=40.0, max_value=80.0, value=55.0, step=0.1)

    if st.button("Estimer la coupe premium", type="primary"):
# 1. On liste TOUTES les colonnes exactes générées lors de ton entraînement (à adapter selon ton X_train)
        colonnes_entrainement = [
            'carat', 'depth', 'table', 
            'color_D', 'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 
            'clarity_IF', 'clarity_VVS1', 'clarity_VVS2', 'clarity_VS1', 'clarity_VS2', 
            'clarity_SI1', 'clarity_SI2', 'clarity_I1']
        
        # 2. On crée un dictionnaire de base rempli de 0
        donnies_modele = {col: [0] for col in colonnes_entrainement}
        
        # 3. On injecte les valeurs numériques directes
        donnies_modele['carat'] = [carat]
        donnies_modele['depth'] = [depth]
        donnies_modele['table'] = [table]
        
        # 4. L'ASTUCE : On passe à 1 les colonnes choisies par l'utilisateur
        donnies_modele[f"color_{color}"] = [1]
        donnies_modele[f"clarity_{clarity}"] = [1]
        
        # 5. On convertit en DataFrame. Il a instantanément la bonne forme !
        df_input = pd.DataFrame(donnies_modele)
        
        # Affichage du tableau saisi pour contrôle
        st.write("Vos données :")
        st.dataframe(df_input)

# TODO 8 - Applique le MÊME scaler qu'à l'entraînement sur la ligne saisie
#   Utilise : scaler.transform()
        df_input_sc=scaler_class.transform(df_input)

# TODO 9 - Prédis puis affiche "premium" ou "non premium"
#   Utilise : model.predict()  ,  st.write()
        coupe_pred = model_class.predict(df_input_sc)
        if coupe_pred[0] == 1:
            st.write("La coupe est **premium**.")
        else:
            st.write("La coupe n'est pas premium.") 

# =============================================================
# PAGE "Analyse"
# =============================================================

else:
    # Si ce n'est ni le premier, ni le deuxième, c'est forcément "Analyse"
    st.title("📊 Analyse exploratoire")
    st.write("Retrouve ici les graphiques et les statistiques de la base de données.")
# TODO 10 - Rédige 10 à 15 lignes : features les plus influentes sur le prix,
#           fiabilité du modèle de classification, limites du prototype
#   Utilise : st.write()  ou  st.markdown()

# TODO 11 - (bonus) Affiche un graphique utile (heatmap ou matrice de confusion)
#   Utilise : st.pyplot()
