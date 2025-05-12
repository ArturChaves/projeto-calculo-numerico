from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

df = pd.read_csv('spotify-2023.csv', encoding='cp1252')
print("Número de linhas e colunas:",df.shape)

df = df.dropna()

print(df.columns)

y = df['danceability_p'].astype(float) # ano de lançamento
x = df['released_year'].astype(float) # streams

X = np.array(x).reshape(-1, 1)
y = np.array(y)
modelo = LinearRegression().fit(X, y)
plt.scatter(X, y)
plt.plot(X, modelo.predict(X), color='red')
plt.ylabel('% Danceability')
plt.xlabel('Ano de Lançamento')
plt.title('Mudança da porcentagem de Danceability ao longo dos anos')
print("O modelo é: y = %.3f + (%.3f)x" %(modelo.intercept_, modelo.coef_))
#plt.show()

model = sm.ols(formula='danceability_p ~ released_year', data=df)

results = model.fit()
print(results.summary())


corr, _ = pearsonr(x, y)
print("Coeficiente de Pearson:", corr)


# ====================================================================== #

# Regressão Linear de energia x bpm

y = df['energy_p'].astype(float) # nivel de energia
x = df['bpm'].astype(float) # bpm
y_array = np.array(y)
x_array = np.array(x).reshape(-1, 1)
modelo = LinearRegression().fit(x_array, y_array)

print("A reta ajustada é: y = %.3f + (%.3f)x" %(modelo.intercept_, modelo.coef_))

model = sm.ols(formula='energy_p ~ bpm', data=df)

results = model.fit()
print(results.summary())

corr, _ = pearsonr(x, y)
print("Coeficiente de Pearson:", corr)

plt.figure()
plt.scatter(x_array, y_array)
plt.xlabel('BPM')
plt.plot(x_array, modelo.predict(x_array), color='red')
plt.ylabel('% Energia')
plt.title('Relação entre BPM e % Energia')
#plt.show()  

# ====================================================================== #

# Regressão Linear de valencia x instrumentalness

y = df['valence_p'].astype(float) # batidas por minuto
x = df['instrumentalness_p'].astype(float) # streams
y_array = np.array(y)
x_array = np.array(x).reshape(-1, 1)
modelo = LinearRegression().fit(x_array, y_array)

print("A reta ajustada é: y = %.3f + (%.3f)x" %(modelo.intercept_, modelo.coef_))

model = sm.ols(formula='valence_p ~ instrumentalness_p', data=df)

results = model.fit()
print(results.summary())

corr, _ = pearsonr(x, y)
print("Coeficiente de Pearson:", corr)

plt.figure()
plt.scatter(x_array, y_array)
plt.plot(x_array, modelo.predict(x_array), color='red')
plt.xlabel('Probabilidade de faixa instrumental')
plt.ylabel('Porcentagem de valência')
plt.title('Relação entre a presença de vocais e a valência')


# ====================================================================== #

# Regressão Linear de in_spotify_playlists x streams

y = df['in_spotify_playlists'].astype(float) # batidas por minuto
x = df['streams'].astype(float) # streams
y_array = np.array(y)
x_array = np.array(x).reshape(-1, 1)
modelo = LinearRegression().fit(x_array, y_array)

print("A reta ajustada é: y = %.3f + (%.3f)x" %(modelo.intercept_, modelo.coef_))

model = sm.ols(formula='in_spotify_playlists ~ streams', data=df)

results = model.fit()
print(results.summary())

corr, _ = pearsonr(x, y)
print("Coeficiente de Pearson:", corr)

plt.figure()
plt.scatter(x_array, y_array)
plt.plot(x_array, modelo.predict(x_array), color='red')
plt.xlabel('Streams')
plt.ylabel('Em playlists do Spotify')
plt.title('Gráfico de Dispersão')

# ====================================================================== #

# Regressão Linear de in_apple_playlists x in_apple_charts

y = df['in_apple_playlists'].astype(float) # in_apple_playlists
x = df['in_apple_charts'].astype(float) # in_apple_charts
y_array = np.array(y)
x_array = np.array(x).reshape(-1, 1)
modelo = LinearRegression().fit(x_array, y_array)

print("A reta ajustada é: y = %.3f + (%.3f)x" %(modelo.intercept_, modelo.coef_))

model = sm.ols(formula='in_apple_playlists ~ in_apple_charts', data=df)

results = model.fit()
print(results.summary())

corr, _ = pearsonr(x, y)
print("Coeficiente de Pearson:", corr)

plt.figure()
plt.scatter(x_array, y_array)
plt.plot(x_array, modelo.predict(x_array), color='red')
plt.xlabel('Nos charts do Apple Music')
plt.ylabel('Em playlists do Apple Music')
plt.title('Relação entre playlists do Apple Music e seus charts')
plt.show()