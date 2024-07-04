import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Carregar os dados
df = pd.read_csv('data/movies_dataset.csv')

# Seleção de variáveis para a modelagem
features = ['Meta_score', 'No_of_Votes', 'Gross']
X = df[features].fillna(0)
y = df['IMDB_Rating']

# Separação em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predição e avaliação
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Exemplo de previsão
example = {'Series_Title': 'The Shawshank Redemption', 'Released_Year': '1994', 'Certificate': 'A',
           'Runtime': '142 min', 'Genre': 'Drama', 'Overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
           'Meta_score': 80.0, 'Director': 'Frank Darabont', 'Star1': 'Tim Robbins', 'Star2': 'Morgan Freeman',
           'Star3': 'Bob Gunton', 'Star4': 'William Sadler', 'No_of_Votes': 2343110, 'Gross': '28,341,469'}

example_df = pd.DataFrame([example])
example_X = example_df[features].apply(pd.to_numeric, errors='coerce')
example_prediction = regressor.predict(example_X)
print(f'Predicted IMDB Rating: {example_prediction}')

# Salvando o modelo em formato .pkl
with open('models/imdb_rating_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)
