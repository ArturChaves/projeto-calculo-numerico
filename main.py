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

# ====================================================================== #

# Regressão Linear Múltipla

print("\n\n# Regressão Linear Múltipla #\n")

# variáveis independentes (x) e dependente (y)
X_vars = ['danceability_p', 'energy_p', 'acousticness_p', 'speechiness_p', 'instrumentalness_p']
y_var = 'streams'

for col in X_vars + [y_var]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna(subset=X_vars + [y_var])

# 1º Regressão Linear Múltipla com todas as variáveis

formula = f"{y_var} ~ " + " + ".join(X_vars)
model_multi = sm.ols(formula=formula, data=df_clean)
results_multi = model_multi.fit()

print("Regressão múltipla inicial com todas as variáveis:")
print(results_multi.summary())
print(f"Coeficiente de determinação (R²): {results_multi.rsquared:.4f}")
print(f"Coeficiente de determinação ajustado (R² ajustado): {results_multi.rsquared_adj:.4f}")

p_values = results_multi.pvalues[1:] 
least_significant_var = p_values.idxmax()
print(f"\nVariável menos significativa: {least_significant_var} (p-valor: {p_values.max():.4f})")
X_vars_reduced = [var for var in X_vars if var != least_significant_var]

# 2º Regressão Linear Múltipla sem a variável menos significativa

formula_reduced = f"{y_var} ~ " + " + ".join(X_vars_reduced)
model_multi_reduced = sm.ols(formula=formula_reduced, data=df_clean)
results_multi_reduced = model_multi_reduced.fit()

print("\nRegressão múltipla após remover a variável menos significativa:")
print(results_multi_reduced.summary())
print(f"Coeficiente de determinação (R²): {results_multi_reduced.rsquared:.4f}")
print(f"Coeficiente de determinação ajustado (R² ajustado): {results_multi_reduced.rsquared_adj:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(df_clean[y_var], results_multi_reduced.predict(), alpha=0.5)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Valores Reais vs. Previstos - Regressão Múltipla')
plt.plot([df_clean[y_var].min(), df_clean[y_var].max()], 
         [df_clean[y_var].min(), df_clean[y_var].max()], 
         'k--', lw=2)
plt.show()