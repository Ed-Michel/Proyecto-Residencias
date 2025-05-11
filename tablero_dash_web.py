import os
import pandas as pd
from sklearn.cluster import AffinityPropagation, Birch, DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import dash as dash
from dash_extensions import EventListener
import dash_bootstrap_components as dbc
import plotly.express as px

# Cargar datos
coordenadas = pd.read_csv("datasets/sitios_sinaloa_coordenadas.csv")
X = pd.read_csv('datasets/AguaSinaloa_2.csv')

# excluir las dos primeras columnas de identificación, 
# se guardan las columnas numericas con las se van a generar los clusters en columnas_numericas
columnas_numericas = X.columns[2:]  

# inicializar la app Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# establecer etiquetas y datos iniciales para los algoritmos
app.layout = dbc.Container([
    dash.html.H1("Clustering - Sitios de muestro Sinaloa"),
    
    dash.html.Br(),
    dbc.Card([
        dbc.CardHeader("Selección de parámetros"),
        dbc.CardBody([
            dash.dcc.Checklist(
                id="columnas-agrupacion",
                options=[{"label": col, "value": col} for col in columnas_numericas],
                value=columnas_numericas.tolist(),
                inline=True,
                inputStyle={"margin-right": "5px", "margin-left": "10px"},
                labelStyle={"display": "inline-block", "margin-right": "15px"}
            )
        ]),
    ], style={
        "margin-top": "20px",
        "background-color": "#f9f9f9",
        "box-shadow": "0 2px 5px rgba(0,0,0,0.1)"
    }),
    
    dash.html.Br(),
    dash.dcc.Dropdown(
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
    
    dash.html.Br(),
    dash.html.Div([
        dash.html.Div([dash.html.Label("damping"), dash.dcc.Input(id="damping", type="number", value=0.7, step=0.1)], id="damping-div"),
        dash.html.Div([dash.html.Label("preference"), dash.dcc.Input(id="preference", type="number", value=-8.8, step=0.01)], id="preference-div"),
        dash.html.Div([dash.html.Label("threshold"), dash.dcc.Input(id="threshold", type="number", value=0.7, step=0.1)], id="threshold-div"), 
        dash.html.Div([dash.html.Label("n clusters"), dash.dcc.Input(id="n_clusters", type="number", value=30, step=1)], id="n_clusters-div"), 
        dash.html.Div([dash.html.Label("epsilon"), dash.dcc.Input(id="eps", type="number", value=1.82, step=0.01)], id="eps-div"),
        dash.html.Div([dash.html.Label("min samples"), dash.dcc.Input(id="min_samples", type="number", value=1, step=1)], id="min_samples-div"),
    ], id="parametros-container"),
    
    dash.html.Br(),
    dash.html.Div([
    EventListener(
            dash.html.Button("Reiniciar", id="reload-button"),
            events=[{"event": "click", "props": {"href": "/", "target": "_self"}}]
        ),
        dash.dcc.Location(id='url', refresh=True)
    ]),
    
    dash.html.Br(),
    dash.html.Button("Generar clusters", id="run-button", n_clicks=0, style={'margin-right': '10px'}, disabled=True), 
    dash.html.Button("Descargar CSV", id="download-button", n_clicks=0, disabled=True),
    dash.dcc.Download(id="descarga-csv"),
    
    dash.html.Br(),
    dash.dcc.Store(id="clustering-data"),
    dbc.Card([
        dbc.CardHeader("Filtrar clusters generados"),
        dbc.CardBody([
            dash.dcc.Checklist(
                id="cluster-checklist",
                options=[],
                value=[],
                inline=True,
                inputStyle={"margin-right": "5px", "margin-left": "10px"},
                labelStyle={"display": "inline-block", "margin-right": "15px"}
            )
        ])
    ], style={
        "margin-top": "20px",
        "background-color": "#f9f9f9",
        "box-shadow": "0 2px 5px rgba(0,0,0,0.1)"
    }),
    
    dash.dcc.ConfirmDialog(
        id='mensaje-error',
        message="Cantidad de parámetros no valida",
        displayed=False
    ),
    dash.html.Br(), 
    dash.dcc.Graph(id="mapa-clustering")
])

# callback y método para el botón de Reiniciar
@app.callback(
    dash.Output("url", "href"),
    [dash.Input("reload-button", "n_clicks")]
)
def reiniciar(n_clicks):
    if n_clicks:
        return "/"
    return None

# callback para mostrar y ocultar los parámetros según el algoritmo seleccionado
@app.callback(
    [dash.Output("damping-div", "style"),
    dash.Output("preference-div", "style"),
    dash.Output("threshold-div", "style"),
    dash.Output("n_clusters-div", "style"),
    dash.Output("eps-div", "style"),
    dash.Output("min_samples-div", "style")],
    dash.Input("algoritmo-dropdown", "value")
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

# callback y método para habilitar el boton Generar Clustering al seleccionar un algoritmo
@app.callback(
    dash.Output("run-button", "disabled"),
    dash.Input("algoritmo-dropdown", "value")
)

def habilitar_boton(algoritmo):
    return algoritmo is None

# callback y método para generar el mapa de clustering
@app.callback(
    dash.Output("mensaje-error", "message"),
    dash.Output("mensaje-error", "displayed"),
    dash.Output("clustering-data", "data"),
    dash.Input("run-button", "n_clicks"),
    [
        dash.State("algoritmo-dropdown", "value"),
        dash.State("damping", "value"),
        dash.State("preference", "value"),
        dash.State("threshold", "value"),
        dash.State("n_clusters", "value"),
        dash.State("eps", "value"),
        dash.State("min_samples", "value"),
        dash.State("columnas-agrupacion", "value")
    ]
)
def generar_clustering(n_clicks, metodo, damping, preference, threshold, n_clusters, eps, min_samples, columnas_seleccionadas):
        # evitar ejecucion al iniciar la aplicacion
        if n_clicks is None or n_clicks == 0:
            return "", False, dash.no_update
        # validacion adicional para evitar problemas en la generacion de clusters
        if not metodo or not columnas_seleccionadas:
            return "Debe seleccionar un algoritmo y al menos una columna para el clustering.", True, dash.no_update
        
        # agrupar por "CLAVE SITIO" y calcular el promedio de cada columna numérica seleccionada
        datos_filtrados = X.groupby("CLAVE SITIO")[columnas_seleccionadas].mean() 
        # validacion en caso de seleccionar pocas columnas (por lo establecido en el PCA)
        if datos_filtrados.shape[1] < 7:
            return "Debe seleccionar al menos 7 columnas (PCA) para el clustering.", True, dash.no_update
        
        # convertir a matriz de vectores
        datos_filtrados = datos_filtrados.values
        # escalar los datos
        datos_filtrados = StandardScaler().fit_transform(datos_filtrados)
        # aplicar PCA para visualizar mejor
        X_pca = PCA(n_components=7).fit_transform(datos_filtrados)
        
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
        coordenadas["Cluster"] = (yhat + 1).astype(str)
        
        fig = px.scatter_mapbox(
            coordenadas, lat="LATITUD", lon="LONGITUD",
            color="Cluster", hover_name="NOMBRE DEL SITIO",
            category_orders={"Cluster": sorted(coordenadas["Cluster"].unique())},
            zoom=6.6, height=900, color_discrete_sequence=px.colors.qualitative.Bold
        )
    
        fig.update_traces(marker=dict(size=10))
        fig.update_layout(showlegend=False, mapbox_style="open-street-map",
                        mapbox_center={"lat": coordenadas["LATITUD"].mean(), "lon": coordenadas["LONGITUD"].mean()},
                        title=f"Clustering {metodo}")
         
        # convierte a string los nombres de cluster
        coordenadas["Cluster"] = coordenadas["Cluster"].astype(str)

        # caso exitoso, se actualiza el mapa y se habilita el botón de descarga
        return (
            "", False, {
                "datos": coordenadas.to_dict("records"),
                "metodo": metodo
            }
        )

# callback y método para actualizar el checklist con los clusters detectados
@app.callback(
    [dash.Output("cluster-checklist", "options"),
     dash.Output("cluster-checklist", "value")],
    dash.Input("clustering-data", "data")
)
def actualizar_checklist(data):
    if not data or "datos" not in data:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame(data["datos"])
    df["Cluster"] = df["Cluster"].astype(str)
    opciones = [{"label": c, "value": c} for c in sorted(df["Cluster"].unique())]
    seleccionados = [op["value"] for op in opciones]

    return opciones, seleccionados

# callback y método para actualizar el mapa al seleccionar los clusters detectados en el checklist
@app.callback(
    dash.Output("mapa-clustering", "figure"),
    [dash.Input("clustering-data", "data"),
     dash.Input("cluster-checklist", "value")]
)
def actualizar_mapa(data, clusters_filtrados):
    if not data or "datos" not in data or not clusters_filtrados:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame(data["datos"])
    metodo = data.get("metodo", "desconocido")
    df["Cluster"] = df["Cluster"].astype(str)
    df_filtrado = df[df["Cluster"].isin(clusters_filtrados)]

    fig = px.scatter_mapbox(
        df_filtrado, lat="LATITUD", lon="LONGITUD",
        color="Cluster", hover_name="NOMBRE DEL SITIO",
        category_orders={"Cluster": sorted(df["Cluster"].unique())},
        zoom=6.6, height=900, color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(
        showlegend=False, mapbox_style="open-street-map", 
        mapbox_center={"lat": df_filtrado["LATITUD"].mean(), "lon": df_filtrado["LONGITUD"].mean()},
        title=f"Mapa de Clustering - {metodo}"
    )

    return fig

# callback y método para habilitar el botón de descarga
@app.callback(
    dash.Output("download-button", "disabled"),
    dash.Input("run-button", "n_clicks")
)
def habilitar_descarga(n_clicks):
    # habilitar el botón de descarga si se ha generado un clustering
    return n_clicks is None or n_clicks == 0

# callback y método para descargar el CSV del clustering generado, según lo que se haya seleccionado en el checklist
@app.callback(
    dash.Output("descarga-csv", "data"),
    [dash.Input("download-button", "n_clicks"),
     dash.State("cluster-checklist", "value")], 
    prevent_initial_call=True
)
def descargar_csv(n_clicks, checklist_value):
    try:
        # filtrar según las opciones seleccionadas en el checklist
        if checklist_value:
            df_filtrado = coordenadas[coordenadas["Cluster"].isin(checklist_value)]
        else:
            df_filtrado = coordenadas
        
        # ordenar el DataFrame filtrado por la columna "Cluster"
        df_filtrado = df_filtrado[["NOMBRE DEL SITIO", "Cluster"]].sort_values("Cluster")

        # devolver el archivo CSV filtrado
        return dash.dcc.send_data_frame(df_filtrado.to_csv, "clusters_asignados_sitios.csv", index=False)
    
    except Exception as e:
        print("Error al generar el archivo a descargar")
        print(str(e))
        return None

if __name__ == "__main__":
    # se establece el puerto para correr la app Dash en Render
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)