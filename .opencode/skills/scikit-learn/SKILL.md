---
name: scikit-learn
description: "Machine learning en Python con scikit-learn. Usar para clasificación, regresión, clustering, evaluación de modelos y pipelines de ML. Dispara con: clasificar, regresión, random forest, SVM, clustering, KMeans, PCA, train/test, cross-validation."
---

# Scikit-learn — Machine Learning clásico

## Overview
Guía completa para tareas de machine learning con scikit-learn: clasificación, regresión, clustering, reducción de dimensionalidad, preprocesamiento, evaluación y pipelines.

## Instalación
```bash
uv pip install scikit-learn pandas numpy matplotlib seaborn
```

## Uso rápido

### Clasificación
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print(classification_report(y_test, model.predict(X_test_scaled)))
```

### Pipeline completo con datos mixtos
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

numeric_features = ['age', 'income']
categorical_features = ['gender', 'occupation']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
])

model = Pipeline([('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier(random_state=42))])
model.fit(X_train, y_train)
```

## Capacidades principales

### 1. Aprendizaje Supervisado
- **Modelos lineales**: LogisticRegression, LinearRegression, Ridge, Lasso
- **Árboles y ensembles**: RandomForest, GradientBoosting, DecisionTree
- **SVM**: SVC, SVR con varios kernels
- **Redes neuronales**: MLPClassifier, MLPRegressor
- **Otros**: Naive Bayes, KNN

### 2. Aprendizaje No Supervisado
- **Clustering**: KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture
- **Reducción dimensional**: PCA, t-SNE, UMAP, TruncatedSVD

### 3. Evaluación y Selección
- **Cross-validation**: KFold, StratifiedKFold, TimeSeriesSplit
- **Hyperparameter tuning**: GridSearchCV, RandomizedSearchCV
- **Métricas**: accuracy, precision, recall, F1, ROC AUC, MSE, R²

### 4. Preprocesamiento
- **Escalado**: StandardScaler, MinMaxScaler, RobustScaler
- **Encoding**: OneHotEncoder, OrdinalEncoder
- **Imputación**: SimpleImputer, KNNImputer
- **Selección de features**: RFE, SelectKBest, SelectFromModel

## Buenas prácticas
- **Siempre usar Pipelines**: evitan data leakage y aseguran consistencia
- **Fit en training, transform en test**: `scaler.fit_transform(X_train)` / `scaler.transform(X_test)`
- **Stratify en clasificación**: `train_test_split(..., stratify=y)`
- **Random state**: `random_state=42` para reproducibilidad
- **Escalar cuando sea necesario**: SVM, KNN, K-Means, PCA requieren escalado

## Referencias
- Documentación oficial: https://scikit-learn.org/stable/
- Skill completo: `~/.config/opencode/skill-libraries/ai-ml/scikit-learn/SKILL.md`
