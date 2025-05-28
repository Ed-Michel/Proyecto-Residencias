import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# cargar los datasets
coordenadas = pd.read_csv("datasets\\sitios_sinaloa_coordenadas.csv")
X = pd.read_csv('datasets\\AguaSinaloa_2.csv')

# guardar los NOMBRE DEL SITIO
points = X.groupby("CLAVE SITIO")["NOMBRE DEL SITIO"].first().values

# excluir las dos primeras columnas de identificación
columnas_numericas = X.columns[2:]  
# agrupar por "CLAVE SITIO" y calcular el promedio de cada columna numérica
X = X.groupby("CLAVE SITIO")[columnas_numericas].mean()
# convertir a matriz de vectores
X = X.values

# escalar los datos
X = StandardScaler().fit_transform(X)
    
def affinity_propagation():
    # aplicar PCA para visualizar mejor
    X_pca = PCA(n_components=7).fit_transform(X)
    
    # definir el modelo
    model = AffinityPropagation(damping=0.7, preference=-8.8)
    # ajustar el modelo
    model.fit(X_pca)
    # obtener las etiquetas de clusters
    yhat = model.labels_
    # verificar número de clusters
    print(f"Número de clusters encontrados: {len(np.unique(yhat))}")
    print()
    
    # se convierte la asignación de clusters de datos continuos a datos categóricos para poder ser filtrados en la gráfica
    coordenadas["Cluster"] = (yhat+1).astype(str)
    # comprobar la asignación de clusters, se guarda en un csv para su próxima evaluación
    coordenadas[["NOMBRE DEL SITIO", "Cluster"]].sort_values("Cluster").to_csv("datasets\\clusters_asignados_sitios.csv", index=False)
    
    # crear el mapa con plotly express
    fig = px.scatter_map(coordenadas, 
                        lat="LATITUD", lon="LONGITUD", 
                        color="Cluster", 
                        hover_name="NOMBRE DEL SITIO",
                        category_orders={"Cluster": sorted(coordenadas["Cluster"].unique())},  
                        zoom=7, height=800,
                        color_discrete_sequence=px.colors.qualitative.Bold)
    
    # aumentar el tamaño de los puntos
    fig.update_traces(marker=dict(size=10)) 
    
    # configuración del mapa
    fig.update_layout(mapbox=dict(style="open-street-map"),
                     mapbox_center={"lat": coordenadas["LATITUD"].mean(), "lon": coordenadas["LONGITUD"].mean()},
                     title="Affinity Propagation")
    
    # mostrar la gráfica
    fig.show()

def birch():
    # aplicar PCA para visualizar mejor
    X_pca = PCA(n_components=7).fit_transform(X)
    
    # define the model
    model = Birch(threshold=0.7, n_clusters=30)
    # fit the model
    model.fit(X_pca)
    # assign a cluster to each example
    yhat = model.labels_
    # verificar número de clusters
    print(f"Número de clusters encontrados: {len(np.unique(yhat))}")
    print()
    
    # se convierte la asignación de clusters de datos continuos a datos categóricos para poder ser filtrados en la gráfica
    coordenadas["Cluster"] = (yhat+1).astype(str)
    # comprobar la asignación de clusters, se guarda en un csv para su próxima evaluación
    coordenadas[["NOMBRE DEL SITIO", "Cluster"]].sort_values("Cluster").to_csv("datasets\\clusters_asignados_sitios.csv", index=False)
    
    # crear el mapa con plotly express
    fig = px.scatter_map(coordenadas, 
                        lat="LATITUD", lon="LONGITUD", 
                        color="Cluster", 
                        hover_name="NOMBRE DEL SITIO",
                        category_orders={"Cluster": sorted(coordenadas["Cluster"].unique())},  
                        zoom=7, height=800,
                        color_discrete_sequence=px.colors.qualitative.Bold)

    # aumentar el tamaño de los puntos
    fig.update_traces(marker=dict(size=10))
    
    # configuración del mapa
    fig.update_layout(mapbox_style="open-street-map",
                     mapbox_center={"lat": coordenadas["LATITUD"].mean(), "lon": coordenadas["LONGITUD"].mean()},
                     title="BIRCH")

    # mostrar la gráfica
    fig.show()

def dbscan():
    # aplicar PCA para visualizar mejor
    X_pca = PCA(n_components=7).fit_transform(X)
    
    # define the model
    model = DBSCAN(eps=1.75, min_samples=1)
    # fit the model
    model.fit(X_pca)
    # assign a cluster to each example
    yhat = model.labels_
    # verificar número de clusters
    print(f"Número de clusters encontrados: {len(np.unique(yhat))}")
    print()
    
    # se convierte la asignación de clusters de datos continuos a datos categóricos para poder ser filtrados en la gráfica
    coordenadas["Cluster"] = (yhat+1).astype(str)
    # comprobar la asignación de clusters, se guarda en un csv para su próxima evaluación
    coordenadas[["NOMBRE DEL SITIO", "Cluster"]].sort_values("Cluster").to_csv("datasets\\clusters_asignados_sitios.csv", index=False)
    
    # crear el mapa con plotly express
    fig = px.scatter_map(coordenadas, 
                        lat="LATITUD", lon="LONGITUD", 
                        color="Cluster", 
                        hover_name="NOMBRE DEL SITIO",
                        category_orders={"Cluster": sorted(coordenadas["Cluster"].unique())},  
                        zoom=7, height=800,
                        color_discrete_sequence=px.colors.qualitative.Bold)

    # aumentar el tamaño de los puntos
    fig.update_traces(marker=dict(size=10))
    
    # configuración del mapa
    fig.update_layout(mapbox_style="open-street-map",
                     mapbox_center={"lat": coordenadas["LATITUD"].mean(), "lon": coordenadas["LONGITUD"].mean()},
                     title="DBSCAN")

    # mostrar la gráfica
    fig.show()
    
def kmeans():
    # aplicar PCA para visualizar mejor
    X_pca = PCA(n_components=7).fit_transform(X)
    
    # define the model
    model = KMeans(n_clusters=30)
    # fit the model
    model.fit(X_pca)
    # assign a cluster to each example
    yhat = model.labels_
    # verificar número de clusters
    print(f"Número de clusters encontrados: {len(np.unique(yhat))}")
    print()
    
    # se convierte la asignación de clusters de datos continuos a datos categóricos para poder ser filtrados en la gráfica
    coordenadas["Cluster"] = (yhat+1).astype(str)
    # comprobar la asignación de clusters, se guarda en un csv para su próxima evaluación
    coordenadas[["NOMBRE DEL SITIO", "Cluster"]].sort_values("Cluster").to_csv("datasets\\clusters_asignados_sitios.csv", index=False)
    
    # crear el mapa con plotly express
    fig = px.scatter_map(coordenadas, 
                        lat="LATITUD", lon="LONGITUD", 
                        color="Cluster", 
                        hover_name="NOMBRE DEL SITIO",
                        category_orders={"Cluster": sorted(coordenadas["Cluster"].unique())},  
                        zoom=7, height=800,
                        color_discrete_sequence=px.colors.qualitative.Bold)

    # aumentar el tamaño de los puntos
    fig.update_traces(marker=dict(size=10))
    
    # configuración del mapa
    fig.update_layout(mapbox_style="open-street-map",
                     mapbox_center={"lat": coordenadas["LATITUD"].mean(), "lon": coordenadas["LONGITUD"].mean()},
                     title="K-Means")

    # mostrar la gráfica
    fig.show()

def mini_batch_kmeans():
    # aplicar PCA para visualizar mejor
    X_pca = PCA(n_components=7).fit_transform(X)
    
    # define the model
    model = MiniBatchKMeans(n_clusters=30)
    # fit the model
    model.fit(X_pca)
    # assign a cluster to each example
    yhat = model.labels_
    # verificar número de clusters
    print(f"Número de clusters encontrados: {len(np.unique(yhat))}")
    print()
    
    # se convierte la asignación de clusters de datos continuos a datos categóricos para poder ser filtrados en la gráfica
    coordenadas["Cluster"] = (yhat+1).astype(str)
    # comprobar la asignación de clusters, se guarda en un csv para su próxima evaluación
    coordenadas[["NOMBRE DEL SITIO", "Cluster"]].sort_values("Cluster").to_csv("datasets\\clusters_asignados_sitios.csv", index=False)
    
    # crear el mapa con plotly express
    fig = px.scatter_map(coordenadas, 
                        lat="LATITUD", lon="LONGITUD", 
                        color="Cluster", 
                        hover_name="NOMBRE DEL SITIO",
                        category_orders={"Cluster": sorted(coordenadas["Cluster"].unique())},  
                        zoom=7, height=800,
                        color_discrete_sequence=px.colors.qualitative.Bold)

    # aumentar el tamaño de los puntos
    fig.update_traces(marker=dict(size=10))
    
    # configuración del mapa
    fig.update_layout(mapbox_style="open-street-map",
                     mapbox_center={"lat": coordenadas["LATITUD"].mean(), "lon": coordenadas["LONGITUD"].mean()},
                     title="Mini-Batch K-Means")

    # mostrar la gráfica
    fig.show()

while True:
    print("Clustering Algorithms")
    print("1. Affinity Propagation")
    print("2. BIRCH")
    print("3. DBSCAN")
    print("4. K-Means")
    print("5. Mini-Batch K-Means")
    print("6. Salir")
    
    opcion = input("Ingresa el numero para la opcion que quiera realizar: ")
    print()

    if opcion == '1':
        affinity_propagation()
    elif opcion == '2':
        birch()
    elif opcion == '3':
        dbscan()
    elif opcion == '4':
        kmeans()
    elif opcion == '5':
        mini_batch_kmeans()
    elif opcion == '6':
        break
    else:
        print("dato no valido!")
        print()