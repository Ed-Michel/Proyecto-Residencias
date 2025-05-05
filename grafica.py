import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# cargar el dataset
X = pd.read_csv('datasets\\AguaSinaloa_2.csv')

# guardar los NOMBRE DEL SITIO
points = X['NOMBRE DEL SITIO'].values 

# excluir las dos primeras columnas de identificación
columnas_numericas = X.columns[2:]  
# agrupar por "CLAVE SITIO" y calcular el promedio de cada columna numérica
X = X.groupby("CLAVE SITIO")[columnas_numericas].mean()
# convertir a matriz de vectores
X = X.values

# escalar los datos
X = StandardScaler().fit_transform(X)

# calculamos la matriz de covarianza
print('NumPy covariance matrix: \n%s' %np.cov(X.T))
    
# calculamos los autovalores y autovectores de la matriz y los mostramos
cov_mat = np.cov(X.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# hacemos una lista de parejas (autovector, autovalor) 
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# ordenamos estas parejas den orden descendiente con la función sort
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# visualizamos la lista de autovalores en orden desdenciente
print('Autovalores en orden descendiente:')
for i in eig_pairs:
    print(i[0])
    
# a partir de los autovalores, calculamos la varianza explicada
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
    
# representamos en un diagrama de barras la varianza explicada por cada autovalor, y la acumulada
with plt.style.context('classic'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center',
            label='Varianza individual explicada', color='g')
    plt.step(range(len(var_exp)), cum_var_exp, where='mid', linestyle='--', label='Varianza explicada acumulada')
    plt.ylabel('Ratio de Varianza Explicada')
    plt.xlabel('Componentes Principales')
    plt.legend(loc='best')
    plt.tight_layout()

plt.show()