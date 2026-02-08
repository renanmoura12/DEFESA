#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import joblib
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

caminho = "/Users/renanmoura/Documents/mestrado/PE-AI/data/dados.xlsx"
df = pd.read_excel(caminho)
print("Shape original:", df.shape)

target_col = "PreEclampsia"

# MAPAS
map_raca = {"Branco": 1, "Pardo": 2, "Preto": 3}
map_boolean = {
    "Sim": 1, "YES": 1, "SIM": 1, "TRUE": 1,
    "Nao": 0, "NAO": 0, "Nao": 0, "FALSE": 0
}
map_hist_diabetes = {
    "Nao": 0, "NAO": 0, "Nao": 0,
    "1o grau": 3, "1 GRAU": 3,
    "2o grau": 2, "2 GRAU": 2,
    "3o grau": 1, "3 GRAU": 1
}

# FEATURES 
input_features = [
    "idade", "imc", "diabetes", "hipertensao",
    "origemRacial", "historicoFamiliarDiabetes", "TipoDiabetes",
    "mediaIP","perdasGestacionais", "peso",
    "idadeGestacional", "idadeGestacionalCorrigida", "pesoFetal",
    "percentilArteriaUterina", "percentilArtUmbilical",
    "percentilPeso","circunferenciaAbdominal"
]

df_processed = df.copy()

# IDADE POR PACIENTE
if "paciente_id" in df_processed.columns:
    paciente_ids = df_processed["paciente_id"]
else:
    df_processed["paciente_id_temp"] = df_processed[
        ["dataNascimento","origemRacial","imc"]
    ].astype(str).agg("_".join, axis=1)
    paciente_ids = df_processed["paciente_id_temp"]

df_processed["paciente_id_base"] = paciente_ids

data_referencia = pd.to_datetime("2025-12-02")

paciente_to_nasc = {}
for pid in paciente_ids.unique():
    nasc = pd.to_datetime(
        df_processed.loc[paciente_ids == pid, "dataNascimento"],
        errors="coerce"
    ).dropna()
    paciente_to_nasc[pid] = nasc.mode().iloc[0] if len(nasc) else None

def calc_idade(d):
    if pd.isna(d):
        return 28
    return np.clip((data_referencia - d).days / 365.25, 15, 50)

df_processed["idade"] = df_processed["paciente_id_base"].map(
    lambda x: calc_idade(paciente_to_nasc.get(x))
)

# GESTACOES POR DATA
if "Data" not in df_processed.columns:
    raise ValueError("Coluna 'Data' (data da consulta) e obrigatoria")

df_processed["Data"] = pd.to_datetime(
    df_processed["Data"], errors="coerce", dayfirst=True
)

df_processed = df_processed.sort_values(
    ["paciente_id_base","Data"]
).reset_index(drop=True)

MAX_GAP = 270

episodios = []

for pid, grupo in df_processed.groupby("paciente_id_base"):
    datas = grupo["Data"].values
    ep = 1
    
    for i in range(len(grupo)):
        if i == 0:
            episodios.append(f"{pid}")
            continue
        
        gap = (datas[i] - datas[i-1]).astype('timedelta64[D]').astype(int)
        if gap > MAX_GAP:
            ep += 1
            
        suf = "" if ep == 1 else chr(ord("A") + ep - 2)
        episodios.append(f"{pid}{suf}")

df_processed["PacienteIdEpisodio"] = episodios

# CONVERSOES CATEGORICAS
if "origemRacial" in df_processed.columns:
    df_processed["origemRacial"] = (
        df_processed["origemRacial"]
        .astype(str).str.strip()
        .map(map_raca)
        .astype(float)
    )

for col in ["diabetes","hipertensao"]:
    if col in df_processed.columns:
        df_processed[col] = (
            df_processed[col]
            .astype(str).str.strip()
            .map(map_boolean)
            .astype(float)
        )

if "historicoFamiliarDiabetes" in df_processed.columns:
    df_processed["historicoFamiliarDiabetes"] = (
        df_processed["historicoFamiliarDiabetes"]
        .astype(str).str.strip()
        .replace(map_hist_diabetes)
        .astype(float)
    )

if "TipoDiabetes" in df_processed.columns:
    df_processed["TipoDiabetes"] = (
        df_processed["TipoDiabetes"]
        .astype(str).str.strip()
        .replace({
            "Diabetes Gestacional": 1,
            "Tipo 1": 2,
            "Tipo 2": 3
        })
    )
    df_processed["TipoDiabetes"] = pd.to_numeric(
        df_processed["TipoDiabetes"], errors="coerce"
    ).fillna(0)

# DADOS OBSTETRICOS
if "perdasGestacionais" in df_processed.columns:
    df_processed["perdasGestacionais"] = (
        pd.to_numeric(df_processed["perdasGestacionais"], errors="coerce")
        .fillna(0)
    )

if "mediaIP" in df_processed.columns:
    df_processed["mediaIP"] = np.where(
        df_processed["mediaIP"] >= 1.3,
        df_processed["mediaIP"] * 1.3,
        df_processed["mediaIP"]
    )

peso_mae_kg = pd.to_numeric(df_processed["peso"], errors="coerce").fillna(0)
peso_feto_g = pd.to_numeric(df_processed["pesoFetal"], errors="coerce").fillna(0)

peso_feto_kg = peso_feto_g / 1000.0

df_processed["peso"] = (peso_mae_kg - peso_feto_kg).clip(lower=35)

# GARANTIR NUMERICO + NAN
for col in input_features:
    if col not in df_processed.columns:
        print(f" Criando {col}=0")
        df_processed[col] = 0
        
    df_processed[col] = (
        pd.to_numeric(df_processed[col], errors="coerce")
        .fillna(0)
        .astype(float)
    )

# LIMITES CLINICOS
def aplicar_limites_realistas(X, feature_names):
    limites = {
        "idade": (15, 50),
        "peso": (35, 150),
        "imc": (15, 50),
        "pesoFetal": (0, 5000)
    }
    X = X.copy()
    for f,(lo,hi) in limites.items():
        if f in feature_names:
            X[f] = X[f].clip(lo,hi)
    return X

df_processed[input_features] = aplicar_limites_realistas(
    df_processed[input_features],
    input_features
)

# ALVO
alvo = (
    df_processed[target_col]
    .replace({True:1,False:0})
    .astype(str).str.upper()
    .replace({"TRUE":1,"FALSE":0})
)

alvo = pd.to_numeric(alvo, errors="coerce")
df_processed[target_col] = alvo.astype("Int64")

df_processed = df_processed[~df_processed[target_col].isna()]
df_processed[target_col] = df_processed[target_col].astype(int)

# ESTATISTICAS FINAIS
print("\n=== ESTATISTICAS FINAIS ===")

for c in ["idade","peso","imc"]:
    s = df_processed[c]
    print(f"{c}: min={s.min():.1f}, max={s.max():.1f}, mean={s.mean():.1f}")

print(
    f"\nShape final: {df_processed.shape}"
    f"\nGestacoes unicas: {df_processed['PacienteIdEpisodio'].nunique()}"
    f"\nSem NaNs nas features"
)

print("Target:", df_processed[target_col].value_counts().to_dict())

# Preparar dados
X = df_processed[input_features].copy()
y = df_processed[target_col].copy()
groups = df_processed["PacienteIdEpisodio"].values

# Split por paciente
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train = X.iloc[train_idx].reset_index(drop=True)
X_test = X.iloc[test_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)

print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")
print(f"\nDistribuicao y_train:\n{y_train.value_counts()}")
print(f"\nDistribuicao y_test:\n{y_test.value_counts()}")

# SMOTE
sm = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

print(f"Shape apos SMOTE: {X_train_smote.shape}")
print(f"Distribuicao pos-SMOTE:\n{pd.Series(y_train_smote).value_counts()}")

# Augmentation
X_np = X_train_smote.values
y_np = y_train_smote.values.astype(float)

cols_binarias = ["diabetes", "hipertensao"]

cont_cols = [c for c in input_features if c not in cols_binarias]
cont_idx = [input_features.index(c) for c in cont_cols]

def augment_gaussian_noise(X_np, y_np, factor=0.3, noise_std=0.02, random_state=None):
    rng = np.random.RandomState(random_state)
    n_new = int(len(X_np) * factor)
    if n_new == 0:
        return X_np.copy(), y_np.copy()
    idx = rng.randint(0, len(X_np), size=n_new)
    X_new = X_np[idx].copy()

    noise = rng.normal(0, noise_std, size=X_new[:, cont_idx].shape)
    X_new[:, cont_idx] += noise

    y_new = y_np[idx]
    return X_new, y_new

Xg, yg = augment_gaussian_noise(
    X_np, y_np, factor=0.4, noise_std=0.02, random_state=RANDOM_STATE
)

X_aug = np.vstack([X_np, Xg])
y_aug = np.concatenate([y_np, yg])
X_final, y_final = shuffle(X_aug, y_aug, random_state=RANDOM_STATE)

X_final_df = pd.DataFrame(X_final, columns=input_features)

X_final_df = aplicar_limites_realistas(X_final_df, input_features)

for col in cols_binarias:
    if col in X_final_df.columns:
        X_final_df[col] = X_final_df[col].clip(0, 1)

X_final = X_final_df.values

print(f"Shape final apos augmentation: {X_final.shape}")

# Train/Val/Test Split
X_train_val, X_test_final, y_train_val, y_test_final = train_test_split(
    X_final, y_final, test_size=0.15, stratify=np.round(y_final), random_state=RANDOM_STATE
)

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15, stratify=np.round(y_train_val), random_state=RANDOM_STATE
)

print(f"Treino final: {X_train_final.shape}")
print(f"Validacao: {X_val.shape}")
print(f"Teste final: {X_test_final.shape}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_final)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=input_features)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=input_features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=input_features)

# LightGBM Model
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    n_estimators=500,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1, 
    force_row_wise=True 
)

lgb_model.fit(
    X_train_scaled,
    y_train_final,
    eval_set=[(X_val_scaled, y_val)],
    eval_metric='binary_logloss',
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0, show_stdv=False)
    ]
)

print("LightGBM treinado com sucesso!")

class ManualPipeline:
    def __init__(self, scaler, model, feature_names):
        self.scaler = scaler
        self.model = model
        self.feature_names = feature_names
    
    def predict_proba(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names)
        X_scaled = self.scaler.transform(X_df)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names)
        X_scaled = self.scaler.transform(X_df)
        return self.model.predict(X_scaled)

trained_model = ManualPipeline(scaler, lgb_model, input_features)

# Feature Importance
if hasattr(lgb_model, 'feature_importances_'):
    feature_importance = lgb_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': input_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 features mais importantes:")
    print(importance_df.head(10))

# Metrics
from sklearn.metrics import precision_score, recall_score, f1_score

y_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)
auc = roc_auc_score(y_test_final, y_proba)
accuracy = accuracy_score(y_test_final, y_pred)
precision = precision_score(y_test_final, y_pred)
recall = recall_score(y_test_final, y_pred)
f1 = f1_score(y_test_final, y_pred)

print(f"\nMetricas calculadas:")
print(f"AUC: {auc:.4f}")
print(f"Acuracia: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

bundle_lgbm = {
    "model": lgb_model,
    "scaler": scaler,
    "input_features": input_features,
    "target": target_col,
    "map_raca": map_raca,
    "map_boolean": map_boolean,
    "map_hist_diabetes": map_hist_diabetes,
    "performance": {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    },
    "version": "2.0_otimizado_final"
}

bundle_path = "/Users/renanmoura/Documents/mestrado/PE-AI/models/model_lgbm_bundle.pkl"
joblib.dump(bundle_lgbm, bundle_path)

print("Bundle salvo com sucesso!")
print(f"Features no modelo: {len(input_features)}")

# Final Report
y_proba = trained_model.predict_proba(X_test_final)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)
y_test_int = np.round(y_test_final).astype(int)

print(f"\nClassification Report:\n{classification_report(y_test_int, y_pred)}")
print(f"\nMatriz de Confusao:\n{confusion_matrix(y_test_int, y_pred)}")
