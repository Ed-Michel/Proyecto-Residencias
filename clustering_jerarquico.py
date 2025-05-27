import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# cargar el dataset
X = pd.read_csv('datasets\\AguaSinaloa_2.csv')
# extraer claves de sitio y nombres de los sitios 
site_names = X[['CLAVE SITIO', 'NOMBRE DEL SITIO']].drop_duplicates().set_index("CLAVE SITIO")

# excluir las dos primeras columnas de identificación
columnas_numericas = X.columns[2:]  
# agrupar por "CLAVE SITIO" y calcular el promedio de cada columna numérica
X = X.groupby("CLAVE SITIO")[columnas_numericas].mean()
# extraer los nombres en el orden establecido por el groupby
site_names = site_names.loc[X.index, 'NOMBRE DEL SITIO'].values
# convertir a matriz de vectores
X = X.values

# escalar los datos
X = StandardScaler().fit_transform(X)

plt.figure(figsize=(25, 10))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'), labels=site_names, color_threshold=25)
plt.title('Clustering Jerárquico')
plt.xlabel('Datos')
plt.ylabel('Distancia Euclidiana')
plt.xticks(rotation=90, ha="right", fontsize=5)
plt.tight_layout()
plt.show()