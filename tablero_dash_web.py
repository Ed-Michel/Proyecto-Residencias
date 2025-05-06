import os
import pandas as pd
from sklearn.cluster import AffinityPropagation, Birch, DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dash import Dash, dcc, html, Output, Input, State
from dash_extensions import EventListener
import dash_bootstrap_components as dbc
import plotly.express as px

# Cargar datos
coordenadas = pd.read_csv("datasets/sitios_sinaloa_coordenadas.csv")
X = pd.read_csv('datasets/AguaSinaloa_2.csv')

# excluir las dos primeras columnas de identificación
columnas_numericas = X.columns[2:]  
# agrupar por "CLAVE SITIO" y calcular el promedio de cada columna numérica
X = X.groupby("CLAVE SITIO")[columnas_numericas].mean()
# convertir a matriz de vectores
X = X.values

# escalar los datos
X = StandardScaler().fit_transform(X)
# aplicar PCA para visualizar mejor
X_pca = PCA(n_components=7).fit_transform(X)

# inicializar la app Dash
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# establecer etiquetas y datos iniciales para los algoritmos
app.layout = dbc.Container([
    html.H1("Clustering - Sitios de muestro Sinaloa"),
    
    html.Br(),
    dcc.Dropdown(
        id="algoritmo-dropdown",
        options=[
            {"label": "Affinity Propagation", "value": "Affinity Propagation"},
            {"label": "BIRCH", "value": "BIRCH"},
            {"label": "DBSCAN", "value": "DBSCAN"},
            {"label": "K-Means", "value": "K-Means"},
            {"label": "Mini-Batch K-Means", "value": "Mini-Batch K-Means"}
        ],
        placeholder="Selecciona un algoritmo",
        searchable=False,
        clearable=False
    ),
    
    html.Br(),
    html.Div([
        html.Div([html.Label("damping"), dcc.Input(id="damping", type="number", value=0.7, step=0.1)], id="damping-div"),
        html.Div([html.Label("preference"), dcc.Input(id="preference", type="number", value=-8.8, step=0.01)], id="preference-div"),
        html.Div([html.Label("threshold"), dcc.Input(id="threshold", type="number", value=0.7, step=0.1)], id="threshold-div"), 
        html.Div([html.Label("n clusters"), dcc.Input(id="n_clusters", type="number", value=30, step=1)], id="n_clusters-div"), 
        html.Div([html.Label("epsilon"), dcc.Input(id="eps", type="number", value=1.82, step=0.01)], id="eps-div"),
        html.Div([html.Label("min samples"), dcc.Input(id="min_samples", type="number", value=1, step=1)], id="min_samples-div"),
    ], id="parametros-container"),
    
    html.Br(),
    html.Div([
    EventListener(
            html.Button("Reiniciar", id="reload-button"),
            events=[{"event": "click", "props": {"href": "/", "target": "_self"}}]
        ),
        dcc.Location(id='url', refresh=True)
    ]),
    
    html.Br(),
    html.Button("Generar Clustering", id="run-button", style={'margin-right': '10px'}, disabled=True), 
    html.Button("Descargar CSV", id="download-button", n_clicks=0, disabled=True),
    dcc.Download(id="descarga-csv"),
    
    html.Br(), 
    dcc.Graph(id="mapa-clustering")
])

# callback para reiniciar la aplicacion
@app.callback(
    Output("url", "href"),
    [Input("reload-button", "n_clicks")]
)

# método para reiniciar la aplicación
def reiniciar(n_clicks):
    if n_clicks:
        return "/"
    return None

# callback para mostrar y ocultar los parámetros según el algoritmo seleccionado
@app.callback(
    [Output("damping-div", "style"),
    Output("preference-div", "style"),
    Output("threshold-div", "style"),
    Output("n_clusters-div", "style"),
    Output("eps-div", "style"),
    Output("min_samples-div", "style")],
    Input("algoritmo-dropdown", "value")
)

# se actualizan los mapas al cambiar los valores de los parámetros
def actualizar_parametros(algoritmo):
    # se oculta todo al inicio
    oculto = {"display": "none"}
    visible = {}
    
    return (
        visible if algoritmo == "Affinity Propagation" else oculto, # damping
        visible if algoritmo == "Affinity Propagation" else oculto, # preference
        visible if algoritmo == "BIRCH" else oculto, # threshold
        visible if algoritmo in ["BIRCH", "K-Means", "Mini-Batch K-Means"] else oculto, # n_clusters
        visible if algoritmo == "DBSCAN" else oculto, # eps
        visible if algoritmo == "DBSCAN" else oculto, # min_samples
    )

# callback para habilitar el boton Generar Clustering al seleccionar un algoritmo
@app.callback(
    Output("run-button", "disabled"),
    Input("algoritmo-dropdown", "value")
)

# método para habilitar este boton
def habilitar_boton(algoritmo):
    return algoritmo is None

# callback para generar el clustering y su respectiva gráfica
@app.callback(
    Output("mapa-clustering", "figure"),
    Input("run-button", "n_clicks"),
    State("algoritmo-dropdown", "value"),
    State("damping", "value"),
    State("preference", "value"),
    State("threshold", "value"),
    State("n_clusters", "value"),
    State("eps", "value"),
    State("min_samples", "value")
)

# método para generar el mapa de clustering
def generar_clustering(n_clicks, metodo, damping, preference, threshold, n_clusters, eps, min_samples):
    try:
        # retorna un mapa vacío si no se ha generado algún clustering
        if not n_clicks or not metodo: 
            return px.scatter_mapbox()
    
        # selección del algoritmo
        if metodo == "Affinity Propagation":
            model = AffinityPropagation(damping=damping, preference=preference)
        elif metodo == "BIRCH":
            model = Birch(threshold=threshold, n_clusters=n_clusters)
        elif metodo == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif metodo == "K-Means":
            model = KMeans(n_clusters=n_clusters)
        elif metodo == "Mini-Batch K-Means":
            model = MiniBatchKMeans(n_clusters=n_clusters)
        else:
            return px.scatter_mapbox()
    
        # fit the model
        model.fit(X_pca)
        # assign a cluster to each example
        yhat = model.labels_
        # validación para ciertos casos
        coordenadas["Cluster"] = (yhat+1).astype(str)
        
        fig = px.scatter_mapbox(
            coordenadas, lat="LATITUD", lon="LONGITUD",
            color="Cluster", hover_name="NOMBRE DEL SITIO",
            category_orders={"Cluster": sorted(coordenadas["Cluster"].unique())},
            zoom=6.6, height=900, color_discrete_sequence=px.colors.qualitative.Bold
        )
    
        fig.update_traces(marker=dict(size=10))
        fig.update_layout(mapbox_style="open-street-map",
                        mapbox_center={"lat": coordenadas["LATITUD"].mean(), "lon": coordenadas["LONGITUD"].mean()},
                        title=f"Clustering {metodo}")
    
        return fig
    
    except Exception as e:
        print("Ocurrió un error al generar el clustering:")
        print(str(e))
        return None

# callback para habilitar el botón de descarga
@app.callback(
    Output("download-button", "disabled"),
    Input("run-button", "n_clicks")
)
def habilitar_descarga(n_clicks):
    # habilitar el botón de descarga si se ha generado un clustering
    return n_clicks is None or n_clicks == 0

# callback para descargar el CSV del clustering generado
@app.callback(  
    Output("descarga-csv", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)

# método para descargar el CSV del clustering generado
def descargar_csv(n_clicks):
    try:
        df = coordenadas[["NOMBRE DEL SITIO", "Cluster"]].sort_values("Cluster")
        return dcc.send_data_frame(df.to_csv, "clusters_asignados_sitios.csv", index=False)
    except Exception as e:
        print("Error al generar el archivo a descargar")
        print(str(e))
        return None

if __name__ == "__main__":
    # Se establece el puerto para correr la app Dash en Render
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)