# %% [markdown]
# # DiamondScope — Prototype ML interactif
# 
# **DiamondScope** est une startup spécialisée dans l'analyse joaillière. Sa directrice data, **Camille Arnaud**, veut un outil interne : ses équipes entrent les caractéristiques d'un diamant, et l'outil prédit son **prix** (régression) ET si le diamant appartient à une **coupe premium** (classification).
# 
# Tu es recruté·e pour construire ce prototype en 1,5 jour.
# 
# **Dataset :** Diamonds (seaborn) — 53 940 diamants, 10 variables.
# 
# ```python
# import seaborn as sns
# df = sns.load_dataset('diamonds')
# ```
# 
# **Livrables attendus :**
# - Ce notebook (Mission 1 : classification, Mission 2 : régression)
# - Un fichier `app.py` dans un dépôt GitHub public (Mission 3 : application Streamlit, avec une page d'analyse rédigée)

# %%
import pandas as pd
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

df = sns.load_dataset('diamonds')


# %% [markdown]
# ---
# ## Mission 1 : classification

# %% [markdown]
# **1.1** — Affiche un aperçu complet du dataset : dimensions, types, statistiques descriptives, valeurs manquantes.

# %%
df_clean=df.drop_duplicates()

# %% [markdown]
# **1.2** — Étudie les corrélations entre les **variables numériques** : affiche une **heatmap sur les variables numériques uniquement** et repère la **multicolinéarité** (deux features trop corrélées entre elles → on n'en gardera qu'une).

# %%
# Heatmap (variables numériques uniquement)
correlation=df_clean.select_dtypes(include='number').corr()

sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title("matrice de correlation")
plt.show

df_clean_1=df_clean.drop(columns=['x', 'y', 'z'])

# %%
df_clean_1['price_log']=np.log1p(df_clean_1['price'])

# %%
sns.boxplot(x=df_clean_1['cut'], y=df_clean_1['price_log'])
plt.show()

# %%
df_cut_Fair=df_clean_1[df_clean_1["cut"]=="Fair"]

# %%
df_cut_Fair=df_clean_1[df_clean_1["cut"]=="Fair"]

Q1 = df_cut_Fair['price_log'].quantile(0.25)
Q3 = df_cut_Fair['price_log'].quantile(0.75)
IQR = Q3 - Q1

borne_basse = Q1 - 1.5 * IQR
borne_haute = Q3 + 1.5 * IQR

# Option 2 : capping (remplacer par les bornes)
# df.loc[df["species"]=="Adelie"]['flipper_length_mm'] = df.loc[df["species"]=="Adelie"]['flipper_length_mm'].clip(lower=borne_basse, upper=borne_haute)

# 1. On stocke la condition dans une variable pour rendre le code plus lisible
mask_Fair = df_clean_1["cut"]=="Fair"

# 2. On utilise .loc pour cibler et modifier directement le DataFrame original
df_clean_1.loc[mask_Fair, 'price_log'] = df_clean_1.loc[mask_Fair, 'price_log'].clip(lower=borne_basse, upper=borne_haute)

# %% [markdown]
# **1.3** — Transforme (encodage et nettoyage) les données pour les exploiter dans un modèle de ML supervisé de **classification**.

# %%
# Encodage + nettoyage
df_clean_encoded=pd.get_dummies(df_clean_1, columns=['cut', 'color', 'clarity'], dtype=int)

# %% [markdown]
# **1.4** — Crée la colonne cible `coupe_premium` issue de `cut` pour la classification : `1` si la coupe est `Ideal` ou `Premium`, `0` sinon. Affiche la répartition.

# %%
# Colonne coupe_premium
df_clean_encoded["coupe_premium"] = np.where((df_clean_encoded["cut_Premium"] == 1) | (df_clean_encoded["cut_Ideal"] == 1), 1, 0)
# Répartition

df_clean_encoded["coupe_premium"].value_counts()

# %%
# Countplot
sns.countplot(data=df_clean_encoded, x="coupe_premium")
plt.title("répartition des valeurs pour la colonne coupe_premium")
plt.show()


# %% [markdown]
# **1.5** — Définis `X` (les features utiles pour prédire `coupe_premium`, sans multicolinéarité, sans `price` ni `cut`) et `y`.

# %%
X = df_clean_encoded.drop(['coupe_premium', 'cut_Ideal', 'cut_Premium', 'cut_Very Good', 'cut_Good', 'cut_Fair', 'price', 'price_log'], axis=1)
y = df_clean_encoded['coupe_premium']

print(f"X : {X.shape}")
print(f"y_classif (coupe_premium) : {y.shape}")

# %% [markdown]
# **1.6** — Sépare ton jeu de données en entraînement / test, puis vois si d'autres transformations sont nécessaires avant d'entraîner ton `KNeighborsClassifier`. Affiche les métriques d'évaluation et interprète les résultats dans une cellule markdown.

# %%
# À compléter (split, scaling si besoin, KNN, métriques)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

scaler_class=StandardScaler()

X_train_sc=scaler_class.fit_transform(X_train)
X_test_sc=scaler_class.transform(X_test)

valeurs_k = [1, 3, 5, 7, 9, 11, 15, 17, 21]
scores_train = []
scores_test  = []

for x in valeurs_k:
    model_class = KNeighborsClassifier(n_neighbors=x)
    model_class.fit(X_train_sc, y_train)
    scores_train.append(model_class.score(X_train_sc, y_train))
    scores_test.append(model_class.score(X_test_sc, y_test))

# Tracer la courbe
plt.plot(valeurs_k, scores_train, label='Train')
plt.plot(valeurs_k, scores_test, label='Test')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy en fonction de k')
plt.show()

# %%
# On va retenir le KNN avec K = 15

model_class = KNeighborsClassifier(n_neighbors=15)
model_class.fit(X_train_sc, y_train)

# %% [markdown]
# **1.7** — Cellule d'analyse : la répartition premium vs non-premium est-elle équilibrée ? Quelle conséquence sur le choix de la métrique ?

# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# La réparition est déséquibrée 66% de premium et 33% de non premium mais l'accuracy est d'environ 80% ce qui est acceptable

y_pred = model_class.predict(X_test_sc)
accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred, labels=model_class.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_class.classes_)
disp.plot()

print(classification_report(y_test, y_pred))

# %% [markdown]
# **1.8** — Sauvegarde le modèle de classification (et le scaler).

# %%
# Sauvegarder
import joblib

joblib.dump(model_class, "model_class.pkl")
joblib.dump(scaler_class, "scaler_class.pkl") 

print("Modèle classification sauvegardé.")

# %% [markdown]
# ---
# ## Mission 2 : régression

# %% [markdown]
# **2.1** — Sélectionne les variables explicatives de la variable expliquée `price`. Explore quelles variables sont utiles (on ne garde pas `coupe_premium`).

# %%
# À compléter (X et y pour la régression)
X = df_clean_encoded.drop(['coupe_premium', 'depth', 'cut_Ideal', 'cut_Premium', 'cut_Very Good', 'cut_Good', 'cut_Fair', 'price_log', 'price'], axis=1)
y = df_clean_encoded['price_log']

print(f"X : {X.shape}")
print(f"y_reg_price : {y.shape}")

# %%
X.head(5)

# %% [markdown]
# **2.2** — Sépare le jeu de données en entraînement / test. Vois si d'autres étapes sont nécessaires avant l'entraînement.

# %%
# À compléter (train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

scaler_price=StandardScaler()
X_train_sc=scaler_price.fit_transform(X_train)
X_test_sc=scaler_price.transform(X_test)

model_price=LinearRegression()
model_price.fit(X_train_sc, y_train)

print(f"R² train : {model_price.score(X_train_sc, y_train):.3f}")
print(f"R² test  : {model_price.score(X_test_sc,  y_test):.3f}")

# %% [markdown]
# **2.3** — Entraîne un `LinearRegression` et affiche : R² train, R² test, MAE test, RMSE test. Analyse ces résultats dans une cellule markdown.

# %%
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

y_pred = model_price.predict(X_test_sc)

print(f"MAE  test : {mean_absolute_error(y_test, y_pred):.0f}")
print(f"RMSE test : {root_mean_squared_error(y_test, y_pred):.0f}")

# %% [markdown]
# On a :
# R² train : 0.889
# R² test  : 0.889
# et 
# MAE  test : 0
# RMSE test : 0
# 
# Il n'y a pas d'overfitting ni d'underfitting et aucune erreur sur la RMSE et la MAE ce qui est excellent

# %%
# Modèle


# Métriques


# %% [markdown]
# **2.4** — Identifie les 3 features qui ont le plus d'influence sur le prix (coefficients les plus élevés en valeur absolue).

# %%
# Coefficients
for feature, coef in zip(X.columns, model_price.coef_):
    print(f"{feature:25s} : {coef:.2f}")

# %% [markdown]
# **2.5** — Sauvegarde le modèle de régression.

# %%
# Sauvegarder
X = df_clean_encoded[['carat']]
y = df_clean_encoded['price_log']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

scaler_price=StandardScaler()
X_train_sc=scaler_price.fit_transform(X_train)
X_test_sc=scaler_price.transform(X_test)

model_price=LinearRegression()
model_price.fit(X_train_sc, y_train)

y_pred = model_price.predict(X_test_sc)

print(f"R² train : {model_price.score(X_train_sc, y_train):.3f}")
print(f"R² test  : {model_price.score(X_test_sc,  y_test):.3f}")
print(f"MAE  test : {mean_absolute_error(y_test, y_pred):.0f}")
print(f"RMSE test : {root_mean_squared_error(y_test, y_pred):.0f}")

joblib.dump(model_price, "model_price.pkl")
joblib.dump(scaler_price, "scaler_price.pkl") 

print("Modèle régression.")

# %% [markdown]
# ---
# ## Mission 3 : Application Streamlit
# 
# Dans un fichier **`app.py`** séparé (pas dans ce notebook) :
# 
# 1. Affiche un titre et une courte description de l'outil.
# 2. Propose un site avec au minimum une page pour prédire le prix d'un diamant (widgets de saisie des variables explicatives), une page pour prédire la catégorie `coupe_premium` (widgets également), et une page répondant à la mission d'analyse (voir ci-dessous).
# 3. Charge les modèles sauvegardés, applique la même transformation qu'à l'entraînement, et affiche les prédictions en temps réel.
# 4. Affiche au moins un graphique utile (heatmap ou matrice de confusion).
# 
# Lance avec :
# ```bash
# streamlit run app.py
# ```
# 
# Pousse `app.py` sur un dépôt GitHub public et soumets le lien.

# %% [markdown]
# ### Page « Analyse » de ton application Streamlit
# 
# Dans une page de ton Streamlit, rédige 10-15 lignes à destination de Camille Arnaud qui répondent à :
# 
# 1. Quelles features influencent le plus le prix d'un diamant ? (d'après ton modèle)
# 2. Le modèle de classification est-il fiable ? Que dit la matrice de confusion ?
# 3. Quelle limite vois-tu dans ce prototype ? Que faudrait-il améliorer avec plus de temps ?


