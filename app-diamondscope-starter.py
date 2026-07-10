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

# =============================================================
# 1. CONFIGURATION DE LA PAGE (Doit être la première commande)
# =============================================================
st.set_page_config(
    page_title="DiamondScope | Expert",
    page_icon="💎",
    layout="centered"
)

# =============================================================
# 2. CHARGEMENT DES MODÈLES (Mise en cache pour la performance)
# =============================================================
@st.cache_resource
def charger_modeles():
    model_class = joblib.load("model_class.pkl")
    scaler_class = joblib.load("scaler_class.pkl")
    model_price = joblib.load("model_price.pkl")
    scaler_price = joblib.load("scaler_price.pkl")
    return model_class, scaler_class, model_price, scaler_price

model_class, scaler_class, model_price, scaler_price = charger_modeles()

# =============================================================
# 3. EN-TÊTE PROFESSIONNEL (Image + Titre)
# =============================================================
# Image d'illustration intégrée depuis Unsplash
st.image(
    "mon_image.png", 
    use_container_width=True
)

st.title("💎 DiamondScope")
st.markdown("**L'outil d'analyse et de prédiction expert pour les professionnels du diamant.**")
st.divider()

# =============================================================
# 4. NAVIGATION PAR ONGLETS
# =============================================================
tab_prix, tab_coupe, tab_analyse = st.tabs([
    "💰 Prédire le prix", 
    "💎 Prédire la coupe premium", 
    "📊 Analyse exploratoire"
])

# =============================================================
# ONGLET 1 : PRÉDIRE LE PRIX
# =============================================================
with tab_prix:
    st.header("Estimation de la valeur marchande")
    st.write("Renseignez les caractéristiques physiques pour obtenir une estimation du prix de vente.")
    
    # Conteneur avec bordure pour un fond neutre et propre
    with st.container(border=True):
        carat = st.number_input("Carat (Poids)", min_value=0.1, max_value=10.0, value=0.5, step=0.01)

    if st.button("Estimer le prix", type="primary", use_container_width=True):
        # Création du DataFrame d'entrée
        df_input = pd.DataFrame({"carat": [carat]})
        
        # Transformation et prédiction
        df_input_sc = scaler_price.transform(df_input)
        prediction_log = model_price.predict(df_input_sc)
        prix_final = np.expm1(prediction_log)[0]
        
        # Affichage du résultat dans un conteneur dédié
        with st.container(border=True):
            st.metric(label="Valeur estimée", value=f"{prix_final:,.2f} $")
            st.caption("Estimation basée sur notre modèle de régression linéaire.")

# =============================================================
# ONGLET 2 : PRÉDIRE LA COUPE PREMIUM
# =============================================================
with tab_coupe:
    st.header("Classification de la taille (Cut)")
    st.write("Évaluez si les proportions et la pureté de la pierre correspondent au standard Premium.")
    
    # Conteneur avec bordure pour grouper les paramètres
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            carat_c = st.number_input("Carat", min_value=0.1, max_value=10.0, value=0.5, step=0.01, key="carat_c")
            color = st.selectbox("Color (Couleur)", ["D", "E", "F", "G", "H", "I", "J"])
            table = st.number_input("Table", min_value=40.0, max_value=80.0, value=55.0, step=0.1)
        with col2:
            clarity = st.selectbox("Clarity (Pureté)", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])
            depth = st.number_input("Depth (Profondeur %)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)

    if st.button("Analyser la coupe", type="primary", use_container_width=True):
        # Liste exacte des colonnes d'entraînement
        colonnes_entrainement = [
            'carat', 'depth', 'table', 
            'color_D', 'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 
            'clarity_IF', 'clarity_VVS1', 'clarity_VVS2', 'clarity_VS1', 'clarity_VS2', 
            'clarity_SI1', 'clarity_SI2', 'clarity_I1'
        ]
        
        # Initialisation à 0
        donnees_modele = {col: [0] for col in colonnes_entrainement}
        
        # Injection des valeurs
        donnees_modele['carat'] = [carat_c]
        donnees_modele['depth'] = [depth]
        donnees_modele['table'] = [table]
        donnees_modele[f"color_{color}"] = [1]
        donnees_modele[f"clarity_{clarity}"] = [1]
        
        df_input_c = pd.DataFrame(donnees_modele)
        
        # Scaler et Prédiction
        df_input_sc_c = scaler_class.transform(df_input_c)
        coupe_pred = model_class.predict(df_input_sc_c)[0]
        
        # Affichage visuel du résultat
        with st.container(border=True):
            if coupe_pred == 1:
                st.success("✨ Résultat : **Coupe Premium**")
                st.write("Ce diamant répond aux critères de haute qualité.")
            else:
                st.warning("📊 Résultat : **Coupe Standard** (Non Premium)")
                st.write("Les caractéristiques de ce diamant ne le classent pas dans la catégorie premium.")

# =============================================================
# ONGLET 3 : ANALYSE
# =============================================================
with tab_analyse:
    st.header("Rapport d'analyse global")
        
    with st.container(border=True):
        st.subheader("💡 Les paramètres qui influencent le prix")
        st.write("""
        * **Le poids (Carat)** est de loin la caractéristique la plus influente sur le prix final d'un diamant, expliquant qu'on utilise que cette caractéristique pour prédire le prix.
        Sur les données test, le nombre d'erreur est de 0.
        """)
        
        st.subheader("💎 Les paramètres qui influencent la classification")
        st.write("""
        * **Pour la coupe, on indique si elle est premium (ou parfaite).** On utilise comme paramètres la table, le carat, la profondeur (depth), la couleur et la pureté.
        """)

    st.divider()

    # --- Section Performances du Modèle ---
    st.header("🎯 Performances du modèle de classification")
    st.write("Voici les métriques d'évaluation de notre algorithme sur les données de test inédites.")

    # 1. Affichage de l'Accuracy Globale
    # TODO : Remplace "85 %" par ta vraie valeur
    st.metric(label="Précision Globale (Accuracy)", value="80%") 

    # Création de deux colonnes pour mettre le graphique et le texte côte à côte
    col_matrice, col_rapport = st.columns(2)

    with col_matrice:
        st.subheader("Matrice de confusion")
        
        # TODO : Remplace ces 4 chiffres par ceux obtenus dans ton Notebook (Vrais Négatifs, Faux Positifs, Faux Négatifs, Vrais Positifs)
        matrice_conf = np.array([[1983, 1756], 
                                [399, 6621]]) 
        
        # Création du graphique avec Seaborn
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(matrice_conf, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Standard', 'Premium'], 
                    yticklabels=['Standard', 'Premium'], 
                    cbar=False, ax=ax)
        plt.ylabel('Réalité (Vraies classes)')
        plt.xlabel('Prédiction du modèle')
        
        # Affichage du graphique dans Streamlit
        st.pyplot(fig)

    with col_rapport:
        st.subheader("Classification Report")
        
        # TODO : Fais un copier-coller de la sortie texte de ton Notebook ici
        rapport = """
                    precision    recall  f1-score   support

    Standard            0.83      0.53      0.65      3739
    Premium             0.79      0.94      0.86      7020

    accuracy                                0.80      10759
    macro avg           0.81      0.74      0.75      10759
    weighted avg        0.81      0.80      0.79      10759
        """
        # st.code permet d'afficher le texte avec une police monospace parfaite pour les tableaux de chiffres
        st.code(rapport, language="text")


    with st.container(border=True):
        st.subheader("💡 Limite du modèle")
        st.write("""
        Dans l'entrainement du modèle, nous avions 66 pourcent de données sur les coupe premium contre 33 pourcent pour le reste. Cela a pu biaiser le modèle. Il est donc possible que le modèle ait tendance à prédire plus souvent la coupe premium.
        
        Cela peut être problématique si en fait des diamands ne sont pas premium mais que le modèle les classe comme tel. Ceux qui peut entrainer une perte de confiance à la vente et des pertes financières.
        
        **Il faudrait donc dans un futur proche, ré-entrainer le modèle avec un dataset plus équilibré.**        
        """)