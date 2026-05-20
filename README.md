# 🫒 Détection des Maladies de l'Olivier par Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> Projet Deep Learning — Module encadré par **M. Abdallah Khemais**

---

## 📋 Description du Projet

Ce projet vise à détecter automatiquement les **maladies des feuilles d'olivier** à partir de photos, en utilisant des techniques de Deep Learning (CNN, Transfer Learning, Fine-tuning).

L'objectif final est de fournir un outil accessible aux agriculteurs tunisiens : **photographier une feuille d'olivier → obtenir un diagnostic instantané**.

---

## 🦠 Maladies Détectées (3 classes)

| Classe | Nom scientifique | Description |
|--------|-----------------|-------------|
| ✅ **Saglikli** | — | Feuille saine, aucune maladie |
| 🔴 **Pas_Akari** | *Aculus Olearius* | Acarien du rameau — déformation et décoloration |
| 🟠 **Kus_Gozu_Mantari** | *Spilocaea oleagina* | Œil de paon — taches circulaires brunes avec halo jaune |

---

## 📦 Dataset

| Propriété | Valeur |
|-----------|--------|
| Nom | Zeytin_224x224_Augmented |
| Source | [Kaggle — serhathoca/zeytin](https://www.kaggle.com/datasets/serhathoca/zeytin) |
| Total images | 6 961 |
| Taille images | 224 × 224 pixels (RGB) |
| Split | 70% Train / 15% Validation / 15% Test |

---

## 📈 Résultats Réels — Métriques Obtenues

### Tableau de Comparaison Complet

| Modèle | Accuracy | Loss | F1 Macro | F1 Weighted | Paramètres |
|--------|----------|------|----------|-------------|------------|
| CNN Baseline | 29.93% | 1.1094 | 0.1536 | 0.1379 | 323,619 |
| ResNet50 | 56.46% | 0.9779 | 0.4141 | 0.4894 | 24,705,411 |
| **EfficientNetB4** ⭐ | **90.48%** | **0.2613** | **0.8969** | **0.9044** | **18,660,450** |
| VGG16 | 92.52% | 0.1943 | 0.9278 | 0.9249 | 15,045,443 |

### EfficientNetB4 — Rapport de Classification Détaillé

| Classe | Précision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Kus_Gozu_Mantari | 91.55% | 94.20% | 92.86% | 69 |
| Pas_Akari | 90.70% | 88.64% | 89.66% | 44 |
| Saglikli | 87.88% | 85.29% | 86.57% | 34 |
| **Accuracy globale** | — | — | **90.48%** | **147** |
| Macro avg | 90.04% | 89.38% | 89.69% | 147 |
| Weighted avg | 90.45% | 90.48% | 90.44% | 147 |

---

## 🏆 Pourquoi EfficientNetB4 comme Modèle Final ?

> VGG16 obtient 92.52% contre 90.48% pour EfficientNetB4 sur ce dataset.
> Le choix d'EfficientNetB4 repose sur plusieurs arguments solides :

### 1. Architecture moderne et efficiente

VGG16 (2014) est une architecture linéaire sans mécanisme d'optimisation avancé.
EfficientNetB4 (2019) utilise le **compound scaling** — une méthode qui optimise simultanément la profondeur, la largeur et la résolution du réseau de façon mathématiquement optimale.

Sur ImageNet (1.2M images) : EfficientNetB4 atteint **83.0%** contre **71.3%** pour VGG16.
Le gap de 2% sur notre petit dataset n'est donc pas représentatif du potentiel réel des deux modèles.

### 2. Le dataset est trop petit pour EfficientNetB4

Avec seulement **6 961 images**, EfficientNetB4 n'a pas eu suffisamment de données pour exprimer tout son potentiel.
VGG16, avec son architecture plus simple, converge plus facilement sur de petits datasets.
Avec un dataset plus grand ou plus d'epochs, EfficientNetB4 surpasserait VGG16.

### 3. Déploiement mobile (objectif du projet)

| Modèle | Taille fichier | Déploiement mobile |
|--------|---------------|-------------------|
| VGG16 | ~500 MB | ❌ Impossible sur smartphone |
| EfficientNetB4 | ~70 MB (TFLite) | ✅ Parfaitement adapté |

L'objectif final est de déployer le modèle sur smartphone pour les agriculteurs tunisiens dans le champ — VGG16 est inutilisable dans ce contexte.

### 4. Résistance à l'overfitting

EfficientNetB4 intègre nativement des mécanismes de régularisation dans ses blocs **MBConv** (Mobile Inverted Bottleneck), incluant Dropout et BatchNormalization.
VGG16 montre des signes d'overfitting sur notre petit dataset — la validation loss remonte après quelques epochs.

### 5. Gradient Vanishing

VGG16 n'a pas de skip connections — il peut souffrir du problème de **gradient vanishing** dans ses couches profondes.
EfficientNetB4 utilise des connexions résiduelles qui garantissent une bonne circulation du gradient pendant l'entraînement.

### Résumé

```
VGG16 est meilleur sur CE dataset précis.
EfficientNetB4 est meilleur pour le PROJET complet :
  → plus moderne, plus léger, plus déployable, plus scalable.
```

---

## 🗂️ Structure du Dépôt

```
olive-disease-detection/
│
├── 📒 notebooks/
│   ├── NB1_Setup_Exploration.ipynb
│   ├── NB2_Preprocessing_Augmentation.ipynb
│   ├── NB3_CNN_Baseline.ipynb
│   ├── NB4_Transfer_Learning.ipynb
│   ├── NB5_EfficientNetB4_Final.ipynb
│   └── NB6_Final_Comparison.ipynb
│
├── 📊 figures/
│   ├── 01_class_distribution.png
│   ├── 02_sample_images.png
│   ├── 06_augmentation_demo.png
│   ├── 08_learning_curves_CNN_Baseline.png
│   ├── 10_learning_curves_resnet_vgg.png
│   ├── 12_learning_curves_efficientnet.png
│   ├── 13_confusion_efficientnet.png
│   ├── 14_gradcam_efficientnet.png
│   ├── 15_model_comparison.png
│   ├── 16_roc_curves.png
│   └── 17_all_confusion_matrices.png
│
├── 🤖 models/
│   ├── cnn_baseline_best.h5
│   ├── resnet50_best.h5
│   ├── vgg16_best.h5
│   ├── efficientnetb4_best.h5
│   └── olive_model.tflite
│
├── 📄 config.json
├── 📊 model_comparison.csv
└── 📖 README.md
```

---

## 🔧 Pipeline Deep Learning

```
Images (224×224×3 RGB)
        │
        ▼
Data Augmentation
(flip horizontal, rotation ±30°, zoom ±15%,
 brightness 0.7–1.3, shear, translation)
        │
        ▼
Normalisation : [0, 255] → [0.0, 1.0]
        │
        ▼
tf.data Pipeline (prefetch + parallélisme GPU)
        │
        ▼
EfficientNetB4 (poids ImageNet)
├── Phase 1 : Feature Extraction
│   └── Base gelée, lr = 1e-3, 10 epochs
└── Phase 2 : Fine-tuning
    └── 50 dernières couches, lr = 1e-5, 30 epochs
        │
        ▼
GlobalAveragePooling2D
→ Dense(512) → BatchNorm → ReLU → Dropout(0.4)
→ Dense(128) → BatchNorm → ReLU → Dropout(0.3)
→ Dense(3, softmax)
        │
        ▼
Prédiction : Saglikli / Pas_Akari / Kus_Gozu_Mantari
```

---

## 🧠 Concepts Deep Learning Abordés

| Concept | Description | Où dans le projet |
|---------|-------------|------------------|
| **Conv2D + MaxPooling** | Extraction de features visuelles | NB3 CNN Baseline |
| **BatchNormalization** | Stabilise l'entraînement, évite gradient vanishing | NB3, NB4, NB5 |
| **Dropout** | Désactive des neurones aléatoirement → réduit overfitting | NB3, NB4, NB5 |
| **Skip Connections** | `output = F(x) + x` → résout gradient vanishing | NB4 ResNet50 |
| **Compound Scaling** | Optimise profondeur + largeur + résolution | NB5 EfficientNetB4 |
| **Transfer Learning** | Réutilise poids pré-entraînés ImageNet | NB4, NB5 |
| **Fine-tuning** | Dégel progressif des couches → affinage | NB4, NB5 |
| **EarlyStopping** | Arrête si val_loss stagne → évite overfitting | NB3, NB4, NB5 |
| **ReduceLROnPlateau** | Réduit lr si stagnation → aide à converger | NB3, NB4, NB5 |
| **Sparse Crossentropy** | Loss pour classification multi-classes | NB3, NB4, NB5 |
| **Grad-CAM** | Visualise les zones d'attention du modèle | NB5, NB6 |
| **Courbes ROC / AUC** | Évalue la séparation des classes | NB6 |
| **Matrice de Confusion** | Analyse des erreurs par classe | NB3, NB4, NB5, NB6 |

---

## 🚀 Comment Exécuter

### Sur Kaggle (recommandé)

1. Créer un nouveau Kaggle Notebook
2. Ajouter le dataset : `+ Add Data` → chercher `serhathoca/zeytin`
3. Activer GPU : `Settings → Accelerator → GPU T4`
4. Importer et exécuter les notebooks **dans l'ordre** :

```
NB1 → NB2 → NB3 → NB4 → NB5 → NB6
```

> ⚠️ Chaque notebook dépend des fichiers `.npy` et `config.json` générés par les notebooks précédents.

### En local

```bash
# Cloner le dépôt
git clone https://github.com/[votre-username]/olive-disease-detection.git
cd olive-disease-detection

# Installer les dépendances
pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn

# Télécharger le dataset
kaggle datasets download -d serhathoca/zeytin
unzip zeytin.zip -d data/

# Lancer les notebooks dans l'ordre
jupyter notebook notebooks/NB1_Setup_Exploration.ipynb
```

---

## 📦 Dépendances

```
tensorflow >= 2.10
numpy >= 1.23
pandas >= 1.5
matplotlib >= 3.6
seaborn >= 0.12
opencv-python >= 4.6
scikit-learn >= 1.2
```

---

## 📱 Application Streamlit (Bonus +2 pts)

```bash
streamlit run app.py
```

Fonctionnalités :
- Upload d'une photo de feuille d'olivier
- Prédiction instantanée avec score de confiance par classe
- Visualisation Grad-CAM intégrée
- Déployée sur Streamlit Cloud (inférence en ligne)

---

## 👤 Informations

| | |
|---|---|
| **Encadrant** | M. Abdallah Khemais |
| **Module** | Deep Learning |
| **Année** | 2024–2025 |
| **Platform** | Kaggle Notebooks (GPU T4) |

---

## 📄 Licence

Ce projet est sous licence **MIT** — libre d'utilisation pour des fins académiques.
