import streamlit as st

# --- Configuration de la page ---
st.set_page_config(
    page_title="Le Guide des Chats Hypoallergéniques", 
    page_icon="🐈", 
    layout="wide"  # Mode large pour mieux répartir le contenu riche
)

# --- CSS personnalisé ---
st.markdown("""
    <style>
    .stApp {
        background-color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #2C3E50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .big-font {
        font-size:18px !important;
        color: #34495E;
    }
    </style>
""", unsafe_allow_html=True)

# --- Base de données des races (Dictionnaire) ---
fiches_races = {
    "Le Sibérien (Le roi des forêts)": {
        "physique": "**Gabarit :** Grand et lourd (jusqu'à 10 kg). Corps \"tonneau\" très musclé.\n\n**Poil :** Mi-long à long, triple et imperméable, collerette très fournie autour du cou.",
        "caractere": "Très affectueux, calme et d'une patience d'ange (idéal avec les enfants). On le qualifie souvent de 'chat-chien' car il s'attache énormément à son maître.",
        "education": "Facile à éduquer par le jeu. Il est intelligent et comprend vite les règles, mais il a besoin de stimulation intellectuelle (jeux d'intelligence).",
        "appartement": "Il s'adapte très bien s'il a de l'espace. **Indispensable :** Un arbre à chat TRÈS robuste (vu son poids) et des griffoirs hauts. Il adore grimper.",
        "sante": "Race naturelle très robuste, peu fragile. Attention toutefois à la CMH (cardiomyopathie hypertrophique), une maladie cardiaque à surveiller chez le vétérinaire."
    },
    "Le Balinais (L'élégance naturelle)": {
        "physique": "**Gabarit :** Moyen (3 à 5 kg), silhouette fine, longiligne et athlétique.\n\n**Poil :** Mi-long, fin et soyeux, sans sous-poil (ce qui réduit encore le risque d'allergies). Queue en panache.",
        "caractere": "Extraverti, grand bavard (il miaule pour communiquer) et ultra-fusionnel. Il déteste la solitude et demande beaucoup d'attention.",
        "education": "Très réceptif au renforcement positif. On peut facilement lui apprendre des petits tours (rapporter un jouet) car il est d'une curiosité sans limites.",
        "appartement": "Parfait pour la vie en intérieur, mais il demande de la présence. Si vous vous absentez 10h par jour, il risque de déprimer ou de faire des bêtises.",
        "sante": "Sensible au niveau des yeux (strabisme fréquent mais bénin) et fragile des dents (sujet au tartre et à la gingivite). Un brossage de dents ou des friandises adaptées sont recommandés."
    },
    "Le Javanais (Le joueur affectueux)": {
        "physique": "**Gabarit :** Moyen et svelte, corps musclé et pattes fines.\n\n**Poil :** Mi-long, très doux, couché sur le corps, s'évasant joliment au niveau de la queue.",
        "caractere": "Plein d'énergie, joueur infatigable et très joueur. Il adore se percher sur les épaules de ses propriétaires.",
        "education": "Il a besoin d'une éducation ferme mais douce. Son énergie débordante doit être canalisée avec des séances de jeu quotidiennes.",
        "appartement": "En appartement, l'enrichissement de l'environnement est vital. Prévoyez des parcours muraux, des tunnels et des jouets interactifs pour combler son besoin d'activité.",
        "sante": "Généralement robuste, mais comme toutes les races dérivées de l'Oriental, il peut être prédisposé à l'amylose (maladie rénale). Un suivi vétérinaire annuel régulier est nécessaire."
    }
}

# 1. Un titre et un en-tête
st.title("🐈 Le Guide d'Expert des Chats Hypoallergéniques à Poils Longs")          
st.write("Découvrez ces races magnifiques qui permettent aux personnes sensibles de réaliser leur rêve d'adoption.")

st.divider()

# 2. Le menu déroulant (selectbox)
choix = st.selectbox(
    "Sélectionnez une race pour afficher sa fiche complète :",
    list(fiches_races.keys())
)

# Récupération des données de la race choisie
info_race = fiches_races[choix]

# Affichage dynamique de la fiche
st.header(f"✨ Zoom sur : {choix.split(' (')[0]}")

# Affichage des informations sans le système de colonnes asymétriques
st.subheader("🧬 Caractéristiques Physiques")
st.write(info_race["physique"])

st.subheader("🎭 Caractère & Comportement")
st.write(info_race["caractere"])

st.divider()

# Section : Vie quotidienne et Contraintes (3 colonnes pour une lecture aérée)
st.subheader("🏠 Guide pratique pour le propriétaire")
col_edu, col_app, col_san = st.columns(3)

with col_edu:
    st.info(f"**🎓 Éducation**\n\n{info_race['education']}")
    
with col_app:
    st.warning(f"**🏢 Habitudes en Appartement & Contraintes**\n\n{info_race['appartement']}\n\n*Contrainte poil :* Un brossage 2 à 3 fois par semaine est obligatoire pour éviter les nœuds !")
    
with col_san:
    st.error(f"**🩺 Santé & Fragilités**\n\n{info_race['sante']}")

st.divider()

# 3. Widgets de saisie interactifs
st.subheader("Faisons connaissance")
note = st.slider("Sur une échelle de 0 à 10, quel est ton niveau d'allergie actuel ?", 0, 10, 6)  
reve = st.checkbox("Je souhaite sauter le pas et visiter un élevage de ce type")      
commentaire = st.text_input("Une question ou une remarque sur l'entretien de ces chats ?")  

st.divider()

# 4. Le bouton de validation
if st.button("Valider mes réponses"):
    st.success("Félicitations, vos préférences ont été enregistrées ! À bientôt dans le monde des amoureux des félins. 🎉")
    if reve:
        st.balloons()  # Petite animation sympa s'ils veulent adopter !
else:
    st.info("Cliquez sur le bouton pour valider vos réponses.")