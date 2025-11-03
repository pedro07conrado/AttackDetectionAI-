# 🚀 AttackDetectionAI - Detecção de Ataques Cibernéticos com IA

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow" alt="Status">
</div>

🔍 **AttackDetectionAI** é um projeto de análise de tráfego de rede usando técnicas de **Machine Learning** para identificar padrões de ataques cibernéticos (e.g., DDoS, intrusões).

---

## 📋 Sobre o Projeto

Este projeto foi desenvolvido para detectar e classificar diversos tipos de ataques cibernéticos através da análise de tráfego de rede utilizando algoritmos de aprendizado de máquina. O sistema é capaz de identificar padrões suspeitos em dados de rede e classificá-los em diferentes categorias de ataques, ajudando na prevenção e mitigação de ameaças.

### 🎯 Objetivos

- Detectar anomalias em tráfego de rede em tempo real
- Classificar diferentes tipos de ataques (DDoS, Brute Force, etc.)
- Comparar o desempenho de diferentes algoritmos de machine learning
- Criar uma solução que possa ser implementada em ambientes reais de segurança cibernética

---

## 📌 Recursos

✅ **Detecção em Tempo Real**: Modelo treinado para classificar tráfego malicioso.  
✅ **Datasets Públicos**: Utiliza o [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) e o [Intrusion Detection Dataset](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset) do Kaggle para treinamento.  
✅ **Técnicas Avançadas**: Pré-processamento de dados, feature engineering e algoritmos como Random Forest/XGBoost.  
✅ **Visualização**: Gráficos interativos para análise de resultados e melhor interpretabilidade.

---

## 🛠️ Tecnologias

<div align="center">
  <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/XGBoost-006C5C?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
</div>

### 📚 Principais Bibliotecas Utilizadas

#### Pandas
- **Uso no projeto**: Manipulação e análise dos dados de tráfego de rede
- **Funcionalidades aplicadas**: 
  - Leitura e processamento dos datasets CSV
  - Limpeza de dados (tratamento de valores nulos)
  - Engenharia de features (criação de novas variáveis)
  - Análise exploratória com `describe()`, `info()`, e agregações

#### Scikit-learn
- **Uso no projeto**: Framework principal para implementação dos modelos de machine learning
- **Funcionalidades aplicadas**:
  - Pré-processamento: `StandardScaler` para normalização dos dados
  - Divisão dos dados: `train_test_split` para separação em conjuntos de treino e teste
  - Modelos: `RandomForestClassifier` para classificação de ataques
  - Avaliação: `classification_report`, `confusion_matrix` para análise de performance
  - Seleção de features: `SelectKBest` para identificar variáveis mais relevantes

#### NumPy
- **Uso no projeto**: Operações numéricas e manipulação de arrays
- **Funcionalidades aplicadas**: 
  - Operações matemáticas vetorizadas
  - Manipulação de matrizes para os algoritmos de ML
  - Geração de números aleatórios para reproducibilidade

#### Matplotlib e Seaborn
- **Uso no projeto**: Visualização dos dados e resultados
- **Funcionalidades aplicadas**:
  - Criação de gráficos de distribuição de classes de ataques
  - Matrizes de confusão para avaliação visual dos modelos
  - Gráficos de importância de features

#### XGBoost
- **Uso no projeto**: Algoritmo de gradient boosting para classificação
- **Funcionalidades aplicadas**:
  - Modelo `XGBClassifier` para comparação com Random Forest
  - Otimização de hiperparâmetros para melhor performance

---

## 📊 Datasets

O projeto utiliza dois conjuntos de dados principais:

### CIC-IDS2017
Este dataset contém tráfego de rede benigno e os ataques mais atuais e comuns, semelhante ao comportamento do mundo real. Fornecido pelo Canadian Institute for Cybersecurity.
- 🔗 [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

### Intrusion Detection Dataset (Kaggle)
Um dataset compreensivo para detecção de intrusão em segurança cibernética com diversos tipos de ataques.
- 🔗 [Cybersecurity Intrusion Detection Dataset](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset)

---

## 💻 Explicação do Código

O notebook `Projeto_Final.ipynb` contém a implementação completa do projeto, dividida nas seguintes seções:

### 1. Importação de Bibliotecas
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
```
Esta seção importa todas as bibliotecas necessárias para análise de dados, visualização e implementação dos modelos de machine learning.

### 2. Carregamento e Exploração de Dados
```python
# Carregamento dos dados
df = pd.read_csv('caminho_para_dataset.csv')

# Análise exploratória
df.info()
df.describe()

# Verificação da distribuição das classes
plt.figure(figsize=(10, 6))
sns.countplot(x='ataque', data=df)
plt.title('Distribuição de Classes de Ataques')
plt.show()
```
Nesta parte, os dados são carregados usando Pandas e explorados para entender sua estrutura, tipos de variáveis e estatísticas básicas. Seaborn é utilizado para visualizar a distribuição das classes.

### 3. Pré-processamento
```python
# Tratamento de valores ausentes
df = df.dropna()

# Codificação de variáveis categóricas
df = pd.get_dummies(df, columns=['protocolo', 'tipo_servico'])

# Normalização dos dados
scaler = StandardScaler()
features = df.drop('ataque', axis=1)
features_scaled = scaler.fit_transform(features)
```
O pré-processamento utiliza Pandas para tratamento de valores ausentes e codificação de variáveis categóricas. Scikit-learn é usado para normalização dos dados com StandardScaler.

### 4. Treinamento dos Modelos
```python
# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(features_scaled, df['ataque'], test_size=0.3, random_state=42)

# Treinamento do Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Treinamento do XGBoost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
```
Nesta fase, Scikit-learn é utilizado para dividir os dados e treinar o modelo Random Forest. XGBoost é implementado como modelo alternativo para comparação de performance.

### 5. Avaliação e Visualização
```python
# Avaliação do Random Forest
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# Matriz de confusão
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Random Forest')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predito')
plt.show()

# Importância das features
feature_imp = pd.Series(rf_model.feature_importances_, index=features.columns).sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_imp[:15], y=feature_imp.index[:15])
plt.title('Top 15 Features Mais Importantes')
plt.show()
```
A avaliação dos modelos utiliza métricas do Scikit-learn como classification_report e confusion_matrix. Matplotlib e Seaborn são usados para visualizar os resultados e a importância das features.

---

## 🚀 Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/pedro07conrado/AttackDetectionAI-.git
```

2. Instale as dependências:
```bash
pip install pandas scikit-learn numpy matplotlib seaborn xgboost jupyter
```

3. Baixe os datasets:
   - [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
   - [Kaggle Dataset](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset)

4. Execute o notebook Jupyter:
```bash
jupyter notebook Projeto_Final.ipynb
```

---

## 📈 Resultados

O projeto alcançou os seguintes resultados:

- Acurácia de 98.7% na detecção de ataques utilizando Random Forest
- Identificação precisa de diferentes tipos de ataques (DDoS, Brute Force, SQL Injection)
- Tempo de processamento eficiente, adequado para aplicações em tempo real
- Feature importance identificando os principais indicadores de ataques

---



