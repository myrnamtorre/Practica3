#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install scikit-learn')


# In[2]:


import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from ipywidgets import interact, Dropdown
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import ipywidgets as widgets
from IPython.display import display



# #PROCESAMIENTO DE DATOS

# In[3]:


df = pd.read_csv('/Users/myrnamigueldelatorre/Documents/Modulo1/UNAM-DCD-G33/Practica3/CTG.csv')


# In[4]:


print(df)


# In[5]:


df.shape


# In[6]:


# Porcentaje de nulos por columna
null_percent = df.isnull().mean()

# Filtrar columnas con más del 20%
cols_to_drop = null_percent[null_percent > 0.20].index

# Eliminar columnas
df = df.drop(columns=cols_to_drop)

print("Columnas eliminadas por exceso de nulos (>20%):")
print(cols_to_drop)


# In[7]:


# Separar numéricas y categóricas
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object", "category"]).columns

# Imputación para numéricos (puedes cambiar "mean" por "median")
num_imputer = SimpleImputer(strategy="median")
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# Imputación para categóricas
cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])


# In[8]:


print(df)


# In[9]:


##USANDO KNN IMPUTER
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Escalar datos numéricos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.select_dtypes(include=["float64", "int64"]))

# Aplicar KNN
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = knn_imputer.fit_transform(df_scaled)

# Volver a DataFrame
df[df.select_dtypes(include=["float64", "int64"]).columns] = scaler.inverse_transform(df_knn)


# In[11]:


print(df)


# In[10]:


##DETECCIÓN Y TRATAMIENTO DE OUTLIERS CON IQR (WINSORIZING), NO UTILIZO Z-SORES PORQUE HAY MUCHAS VARIABLES QUE NO SON NORMALES

# Seleccionar solo columnas numéricas
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Límites inferior y superior
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Winsorization (reemplazar outliers por los límites)
    df[col] = np.where(df[col] < lower, lower,
                       np.where(df[col] > upper, upper, df[col]))

print("✔ Outliers tratados con IQR")


# ANALISIS DE DATOS

# In[11]:


def check_data_completeness_MyrnaMiguel(df):
    """
    Retorna un DataFrame con:
    - Conteo de nulos
    - Porcentaje de completitud
    - Tipo de dato
    - Estadísticos de dispersión (para columnas numéricas)
    """

    # Métricas básicas
    summary = pd.DataFrame({
        'Nulos': df.isnull().sum(),
        'Completitud (%)': (1 - df.isnull().mean()) * 100,
        'Tipo de dato': df.dtypes
    })

    # Crear columnas vacías para estadísticas
    stats_cols = ['Min', 'Q1', 'Mediana', 'Media', 'Q3', 'Max', 'STD']
    for col in stats_cols:
        summary[col] = None

    # Agregar estadísticas solo a columnas numéricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        summary.loc[col, 'Min'] = df[col].min()
        summary.loc[col, 'Q1'] = df[col].quantile(0.25)
        summary.loc[col, 'Mediana'] = df[col].median()
        summary.loc[col, 'Media'] = df[col].mean()
        summary.loc[col, 'Q3'] = df[col].quantile(0.75)
        summary.loc[col, 'Max'] = df[col].max()
        summary.loc[col, 'STD'] = df[col].std()

    return summary


# In[12]:


resultado = check_data_completeness_MyrnaMiguel(df)
resultado


# In[13]:


def clasificar_columnas_MyrnaMiguel(df):
    """
    Clasifica las columnas en:
    - Continuas: numéricas con más de 10 valores únicos
    - Discretas: columnas con 10 o menos valores únicos
    """

    columnas_continuas = []
    columnas_discretas = []

    for col in df.columns:
        n_unique = df[col].nunique()

        # Continuas: numéricas y con más de 10 valores únicos
        if df[col].dtype in ['int64', 'float64'] and n_unique > 10:
            columnas_continuas.append(col)

        # Discretas: cualquier tipo con 10 o menos valores únicos
        elif n_unique <= 10:
            columnas_discretas.append(col)

    return columnas_continuas, columnas_discretas


# In[14]:


cont, disc = clasificar_columnas_MyrnaMiguel(df)

print("Columnas Continuas:")
print(cont)

print("\nColumnas Discretas:")
print(disc)


# 3. VISUALIZACIONES 

# In[15]:


# Obtener columnas numéricas
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

def hist_kde_group_nsp_interactivo(col):
    data = df[[col, "NSP"]].dropna()

    grupos = data["NSP"].unique()

    plt.figure(figsize=(9,5))

    for g in grupos:
        subset = data[data["NSP"] == g][col]

        # Histograma
        plt.hist(subset, bins=30, density=True, alpha=0.35, label=f"NSP={g}")

        # KDE
        kde = gaussian_kde(subset)
        x = np.linspace(subset.min(), subset.max(), 300)
        plt.plot(x, kde(x), linewidth=2)

    plt.title(f"Histograma + KDE: {col} (agrupado por NSP)")
    plt.xlabel(col)
    plt.legend()
    plt.show()

# Interfaz interactiva
interact(
    hist_kde_group_nsp_interactivo,
    col=Dropdown(options=numeric_cols, description="Variable:")
)


# In[16]:


# Detectar columnas numéricas
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

def boxplot_interactivo_dropdown(col):

    fig = px.box(
        df,
        x="NSP",
        y=col,
        color="NSP",
        title=f"Boxplot interactivo de {col} por clase NSP",
        points="all"  # muestra todos los puntos individuales
    )

    fig.update_layout(template="plotly_white", height=500)
    fig.show()

# Dropdown interactivo
interact(
    boxplot_interactivo_dropdown,
    col=Dropdown(options=numeric_cols, description="Variable:")
)


# In[17]:


# Detectar columnas categóricas
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Agregar numéricas discretas (<10 categorías)
for col in df.select_dtypes(include=['int64','float64']).columns:
    if df[col].nunique() < 10:
        cat_cols.append(col)

cat_cols = sorted(set(cat_cols))  # eliminar duplicados

def grafico_combinado(col):
    # calcular frecuencias
    counts = df[col].value_counts()
    percentages = counts / counts.sum() * 100

    # ordenar ascendente para barras horizontales
    counts = counts.sort_values(ascending=True)
    percentages = percentages[counts.index]

    fig = go.Figure()

    # --- BARRAS HORIZONTALES ---
    fig.add_trace(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation='h',
        name='Frecuencia',
        marker=dict(opacity=0.6)
    ))

    # --- LÍNEA DE PORCENTAJE ---
    fig.add_trace(go.Scatter(
        x=percentages.values,
        y=percentages.index,
        mode='lines+markers',
        name='Porcentaje (%)',
        xaxis='x2'  # segunda escala en eje X
    ))

    # configuración de doble eje en X
    fig.update_layout(
        title=f"Frecuencia + Porcentaje (%) de {col}",
        template="plotly_white",
        height=500,
        xaxis=dict(title="Frecuencia"),
        xaxis2=dict(
            overlaying='x',
            side='top',
            title='Porcentaje (%)',
        ),
        yaxis=dict(title=col)
    )

    fig.show()

# Dropdown interactivo
interact(
    grafico_combinado,
    col=Dropdown(options=cat_cols, description="Variable:")
)


# In[18]:


# ---- Asegurar que DATE sea datetime ----
df_time = df.copy()
df_time["Date"] = pd.to_datetime(df_time["Date"], errors="coerce")

# Ordenar por fecha
df_time = df_time.sort_values("Date")

# columnas numéricas (except NSP)
numeric_cols = df_time.select_dtypes(include=['int64','float64']).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["NSP"]]

# -------- FUNCIÓN PARA GRAFICAR ---------
def series_por_nsp_real(col):
    fig = px.line(
        df_time,
        x="Date",
        y=col,
        color="NSP",             # una línea por cada clase
        markers=False,           # líneas más limpias
        title=f"Serie Temporal Real de {col} por Clase NSP"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Fecha",
        yaxis_title=col,
        height=550
    )

    fig.show()

# -------- DROPDOWN INTERACTIVO ---------
interact(
    series_por_nsp_real,
    col=Dropdown(options=numeric_cols, description="Variable:")
)


# In[19]:


# Detectar columnas numéricas
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if "NSP" in numeric_cols:
    numeric_cols.remove("NSP")

# Detectar columnas categóricas con EXACTAMENTE 2 grupos
cat_cols_two = [c for c in df.columns if df[c].nunique() == 2]

def dotplot_overlay(num_col, group_col):
    # Filtrar grupos
    grupos = df[group_col].dropna().unique()
    grupos.sort()

    g1, g2 = grupos[0], grupos[1]

    df1 = df[df[group_col] == g1][num_col]
    df2 = df[df[group_col] == g2][num_col]

    # Figuras
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df1,
        y=[0]*len(df1),
        mode='markers',
        name=f"{group_col} = {g1}",
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=df2,
        y=[0.15]*len(df2),
        mode='markers',
        name=f"{group_col} = {g2}",
        marker=dict(size=8)
    ))

    # Formato general
    fig.update_layout(
        title=f"Dot Plot Overlay — Comparación de {num_col} entre {group_col}={g1} y {g2}",
        template="plotly_white",
        xaxis_title=num_col,
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=450
    )

    fig.show()

# Dropdown interactivo
interact(
    dotplot_overlay,
    num_col=Dropdown(options=numeric_cols, description="Variable:"),
    group_col=Dropdown(options=cat_cols_two, description="Grupo 2-clases:")
)


# In[20]:


# Detectar columnas numéricas (excepto NSP)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "NSP"]

# Variable inicial
initial_var = numeric_cols[0]

# Crear listas por clase NSP
classes = sorted(df["NSP"].dropna().unique())

def get_kde_data(variable):
    groups = []
    labels = []
    for c in classes:
        vals = df[df["NSP"] == c][variable].dropna().tolist()
        groups.append(vals)
        labels.append(f"NSP={c}")
    return groups, labels

# KDE inicial
groups, labels = get_kde_data(initial_var)
fig = ff.create_distplot(groups, labels, show_hist=False, show_rug=False)

# Crear botones del dropdown
dropdown_buttons = []

for col in numeric_cols:
    g, l = get_kde_data(col)
    dropdown_buttons.append(
        dict(
            label=col,
            method="update",
            args=[
                {"x": g},   # actualizar datos
                {"title": f"Densidad KDE de {col} por clase NSP"}
            ]
        )
    )

# Añadir dropdown al layout
fig.update_layout(
    updatemenus=[
        dict(
            buttons=dropdown_buttons,
            direction="down",
            showactive=True,
            x=1.15,
            y=1.05
        )
    ],
    title=f"Densidad KDE de {initial_var} por clase NSP"
)

fig.update_layout(
    xaxis_title="Valor",
    yaxis_title="Densidad"
)

fig.show()


# In[21]:


# Detectar columnas numéricas (excepto NSP)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "NSP"]

# Variable inicial
initial_var = numeric_cols[0]

# Crear figura inicial
fig = px.violin(
    df,
    x="NSP",
    y=initial_var,
    color="NSP",
    box=False,
    points=False,
    title=f"Violín + Swarm para {initial_var} por clase NSP"
)

# Agregar SWARM (scatter jitter)
swarm = px.strip(
    df,
    x="NSP",
    y=initial_var,
    color="NSP",
    stripmode="overlay",
)

# Overlay swarm dentro del violín
for trace in swarm.data:
    fig.add_trace(trace)

# Crear dropdown
dropdown_buttons = []

for col in numeric_cols:
    violin = px.violin(df, x="NSP", y=col, color="NSP", box=False, points=False).data
    swarm = px.strip(df, x="NSP", y=col, color="NSP", stripmode="overlay").data

    dropdown_buttons.append(
        dict(
            label=col,
            method="update",
            args=[
                {"data": violin + swarm},   # Actualiza trazas
                {"title": f"Violín + Swarm para {col} por clase NSP"}
            ]
        )
    )

# Insertar dropdown en layout
fig.update_layout(
    updatemenus=[
        dict(
            buttons=dropdown_buttons,
            direction="down",
            x=1.18,
            y=1.05
        )
    ]
)

fig.update_layout(
    xaxis_title="NSP",
    yaxis_title="Valor",
)

fig.show()


# In[22]:


# Variables numéricas
numeric_cols = df.select_dtypes(include="number").columns.tolist()


def plot_heatmap(selected_vars, method):
    """
    Genera un heatmap interactivo con anotaciones para variables seleccionadas.
    """
    if len(selected_vars) < 2:
        print("Selecciona al menos 2 variables para generar el heatmap.")
        return

    data = df[selected_vars].corr(method=method)

    fig = go.Figure(
        data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.columns,
            colorscale="RdBu",
            reversescale=True,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlación")
        )
    )

    # Agregar anotaciones
    annotations = []
    for i, row in enumerate(data.values):
        for j, val in enumerate(row):
            annotations.append(
                go.layout.Annotation(
                    text=str(round(val, 2)),
                    x=data.columns[j],
                    y=data.columns[i],
                    showarrow=False,
                    font=dict(color="black", size=10)
                )
            )

    fig.update_layout(
        title=f"Heatmap de Correlación ({method.capitalize()})",
        annotations=annotations,
        width=800,
        height=800
    )

    fig.show()


# --- WIDGETS INTERACTIVOS ---

var_selector = widgets.SelectMultiple(
    options=numeric_cols,
    value=numeric_cols[:5],   # primeras 5 por defecto
    description="Variables",
    rows=10
)

method_selector = widgets.Dropdown(
    options=["pearson", "spearman"],
    value="pearson",
    description="Método"
)

button = widgets.Button(
    description="Actualizar",
    button_style="success"
)

output = widgets.Output()

def update_plot(b):
    with output:
        output.clear_output()
        plot_heatmap(list(var_selector.value), method_selector.value)

button.on_click(update_plot)

display(var_selector, method_selector, button, output)


# 4.CONSTRUCCION DE LIBRERIA

# In[29]:


import sys
sys.path.append('/Users/myrnamigueldelatorre/Documents/Practica3')


# In[35]:


import pandas as pd
from ctg_viz.plots.histograms import plot_hist
from ctg_viz.plots.boxplots import plot_box
from ctg_viz.plots.barplots import plot_bar
from ctg_viz.plots.density import plot_density
from ctg_viz.plots.heatmap import plot_correlation_heatmap


def test_plot_hist():
    df = pd.DataFrame({"a": [1, 2, 3]})
    plot_hist(df, "a")  # Solo validar que no lanza error


def test_plot_box():
    df = pd.DataFrame({"a": [1, 2, 3]})
    plot_box(df, "a")


def test_plot_bar():
    df = pd.DataFrame({"a": ["x", "x", "y"]})
    plot_bar(df, "a")


def test_plot_density():
    df = pd.DataFrame({"a": [1, 2, 3]})
    plot_density(df, "a")


def test_heatmap():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    plot_correlation_heatmap(df)


# In[ ]:


##Solo correr si sale error en la importacion
#import importlib
#import ctg_viz.plots.histograms as h
#importlib.reload(h)


# In[38]:


import sys
sys.path.append('/Users/myrnamigueldelatorre/Documents/Practica3')


# In[41]:


get_ipython().system('pytest -v')





# In[1]:


get_ipython().system('jupyter nbconvert --to python MyrnaMiguel.ipynb')


# In[2]:


import pytest
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

# Importa tus funciones (ajusta el nombre de archivo si es necesario)
from MyrnaMiguel import (
    check_data_completeness_MyrnaMiguel,
    clasificar_columnas_MyrnaMiguel,
    get_kde_data,
    dotplot_overlay
)

# -------------------------------------------------------
# 1) Test para check_data_completeness_MyrnaMiguel
# -------------------------------------------------------

def test_check_data_completeness():
    df = pd.DataFrame({
        "A": [1, 2, None],
        "B": [3, 4, 5]
    })

    completeness_df = check_data_completeness_MyrnaMiguel(df)

    assert "columna" in completeness_df.columns
    assert "nulos" in completeness_df.columns
    assert "completitud" in completeness_df.columns

    # Valores esperados
    assert completeness_df.loc[completeness_df["columna"]=="A", "nulos"].iloc[0] == 1
    assert completeness_df.loc[completeness_df["columna"]=="B", "nulos"].iloc[0] == 0


# -------------------------------------------------------
# 2) Test para clasificar_columnas_MyrnaMiguel
# -------------------------------------------------------

def test_clasificar_columnas():
    df = pd.DataFrame({
        "num": [1, 2, 3],
        "cat": ["a", "b", "c"],
        "bin": [0, 1, 0]
    })

    clasif = clasificar_columnas_MyrnaMiguel(df)

    assert "numericas" in clasif
    assert "categoricas" in clasif
    assert isinstance(clasif["numericas"], list)
    assert isinstance(clasif["categoricas"], list)

    # Verifica que detecte variables numéricas
    assert "num" in clasif["numericas"]
    # Verifica que detecte variables categóricas
    assert "cat" in clasif["categoricas"] or "cat" in clasif["object"]


# -------------------------------------------------------
# 3) Test para get_kde_data
# -------------------------------------------------------

def test_get_kde_data():
    variable = np.array([1, 2, 3, 4, 5])

    x_vals, y_vals = get_kde_data(variable)

    assert len(x_vals) == len(y_vals)
    assert len(x_vals) > 10
    assert all(isinstance(v, (int, float, np.floating)) for v in x_vals)


# -------------------------------------------------------
# 4) Test para dotplot_overlay
# -------------------------------------------------------

def test_dotplot_overlay():
    df = pd.DataFrame({
        "value": [1, 2, 3, 4],
        "group": ["A", "A", "B", "B"]
    })

    fig = dotplot_overlay("value", "group")

    # Verifica que devuelve una figura de matplotlib
    assert isinstance(fig, Figure)

