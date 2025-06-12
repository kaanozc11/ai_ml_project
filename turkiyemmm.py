# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 12:43:33 2025

@author: Kaan
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import unidecode
import matplotlib.colors as mcolors
import logging
from sklearn.decomposition import PCA
import seaborn as sns
from yellowbrick.cluster import SilhouetteVisualizer


# Log ayarları
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

st.set_page_config(layout="wide")
st.title("🇹🇷 K-Means Cluster Analysis of Türkiye")

# ---Read the Dataset ---
df = pd.read_excel("turkiyemmm.xlsx")
st.success("Dataset loaded successfuly .")

# --- Missing Data Imputation ---
print(df.isna().sum())
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# --- 3. Label Encoding ---
label_encoder = LabelEncoder()
for col in ['most_edu_level', 'most_voted_party', 'region']:
    df[col] = label_encoder.fit_transform(df[col])

# --- Selecting Only Numerical Columns---
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numerical_cols].apply(pd.to_numeric, errors='coerce').dropna()
df_clean = df.loc[X.index].copy()

# --- Normalize the Data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# --- k value ---
k = st.slider("How many cluster ? (k)", 2, 10, 3)

# --- K-Means Algorithm---
kmeans = KMeans(n_clusters=k, random_state=42, init="k-means++", n_init='auto')
df_clean["cluster"] = kmeans.fit_predict(X_scaled)

kmeans.labels_

#Principal Component Analysis(PCA)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_clean["pca1"] = X_pca[:, 0]
df_clean["pca2"] = X_pca[:, 1]
explained_var = pca.explained_variance_ratio_.sum()

#K-Means with PCA
kmeans2 = KMeans(n_clusters=k, random_state=42, n_init='auto')
df_clean["cluster_pca"] = kmeans2.fit_predict(X_pca)
score2 = silhouette_score(X_pca, df_clean["cluster_pca"])


# --- Silhouette Score ---
score = silhouette_score(X_scaled, df_clean["cluster"])
score2 = silhouette_score(X_pca, df_clean["cluster_pca"])
st.write(f"📈 **Silhouette Score (k={k})**: `{score:.2f}`")
st.write(f"📌 **Silhouette Score with PCA (k={k})** {score2:.2f}")

# --- Silhouette Score Comparison Chart ---
st.subheader("📊 Silhouette Score Karşılaştırması")

fig_score, ax_score = plt.subplots()
methods = ["Normalized Data", "PCA Data"]
scores = [score, score2]
bar_colors = ['skyblue', 'salmon']

ax_score.bar(methods, scores, color=bar_colors)
ax_score.set_ylim(0, 1)
ax_score.set_ylabel("Silhouette Score")
ax_score.set_title(f"Silhouette Score Comparison (k={k})")

for i, v in enumerate(scores):
    ax_score.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

st.pyplot(fig_score)

# PCA Scatter Plot
st.subheader("🧭 PCA ile 2B Küme Görselleştirmesi")
fig_pca, ax_pca = plt.subplots()
scatter = ax_pca.scatter(df_clean["pca1"], df_clean["pca2"], c=df_clean["cluster_pca"], cmap='Accent')
ax_pca.set_xlabel("PCA 1")
ax_pca.set_ylabel("PCA 2")
ax_pca.set_title(f"PCA Clusters (Explained Variance: {explained_var:.2%})")
st.pyplot(fig_pca)


# --- Cluster Summary ---
st.subheader("🔍 Cluster Summary")
cluster_summary = df_clean.groupby("cluster")[numerical_cols].mean().round(2)
st.dataframe(cluster_summary)

# --- Choosing best k with using Elbow Method ---
st.subheader("📌 Best k with Elbow Method")
inertia_list = []
K_range = range(1, 11)
for i in K_range:
    km = KMeans(n_clusters=i, random_state=42, n_init='auto')
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)

# Elbow Graph
fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(K_range, inertia_list, marker='o')
ax_elbow.set_xlabel("Cluster Number (k)")
ax_elbow.set_ylabel("Inertia")
ax_elbow.set_title("Best k with Elbow Method")
st.pyplot(fig_elbow)


# --- Map Data and Name Normalization ---
gdf = gpd.read_file("turkiye_il.geojson")
gdf["name_lower"] = gdf["shapeName"].apply(lambda x: unidecode.unidecode(x.lower()))
df_clean["country_name_lower"] = df_clean["country_name"].apply(lambda x: unidecode.unidecode(x.lower()))

merged = gdf.merge(df_clean, left_on="name_lower", right_on="country_name_lower")

# --- 12. Drawing Turkey Map ---
colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#ffff33', '#c9b204', '#000000', '#52082c', '#f005f0']
cmap = mcolors.ListedColormap(colors[:k])

fig, ax = plt.subplots(figsize=(10, 10))
merged.plot(column="cluster", cmap=cmap, legend=True, ax=ax, edgecolor='black', linewidth=0.5)
ax.set_title(f"K-means Türkiye Map (k={k})", fontsize=14)
ax.axis("off")
st.pyplot(fig)



