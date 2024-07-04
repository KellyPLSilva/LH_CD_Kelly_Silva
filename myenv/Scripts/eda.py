import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando os dados
df = pd.read_csv('movies_dataset.csv')

# Visualizando as primeiras linhas do dataset
print(df.head())

# Estatísticas descritivas
print(df.describe())

# Verificando dados ausentes
print(df.isnull().sum())

# Análise de distribuições
sns.histplot(df['Gross'], bins=50)
plt.title('Distribuição do Faturamento')
plt.show()

sns.boxplot(x='Genre', y='Gross', data=df)
plt.title('Distribuição do Faturamento por Gênero')
plt.show()

sns.scatterplot(x='Meta_score', y='Gross', data=df)
plt.title('Relação entre Meta_score e Faturamento')
plt.show()
