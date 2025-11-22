# Práctica 3 – Análisis del Dataset CTG

Este repositorio contiene el desarrollo completo de la Práctica 3, enfocado en el análisis exploratorio del dataset CTG (Cardiotocographic), la generación de visualizaciones, funciones personalizadas y la creación de un reporte PDF automatizado usando Python.

**Objetivo del proyecto**
Analizar el comportamiento de diferentes variables del dataset CTG aplicando:
Estadística descriptiva
Visualización de datos
Análisis de correlación
Limpieza y diagnóstico del dataset
Funciones personalizadas para análisis
Generación automática de un reporte PDF con gráficos e interpretación

**Descripción del dataset CTG**
El dataset contiene medidas clínicas relacionadas con estudios de monitoreo fetal.
Incluye 2129 muestras y 40 variables, entre ellas:
Frecuencia cardiaca fetal
Variación a corto y largo plazo
Contracciones uterinas
Movimientos fetales
Aceleraciones y desaceleraciones
El conjunto permite evaluar patrones fisiológicos y detectar riesgos durante el embarazo.

**Funciones implementadas**
1. check_data_completeness_MyrnaMiguel(df)
Evalúa la completitud del dataset y calcula:
Valores nulos por columna
Porcentaje total de datos faltantes
Diagnóstico de calidad de datos
2. clasificar_columnas_MyrnaMiguel(df)
Clasifica automáticamente las columnas en:
Numéricas
Categóricas
Booleanas
Facilitando el preprocesamiento y análisis.
3. get_kde_data(serie)
Genera datos KDE para obtener distribuciones suavizadas, permitiendo:
Detectar concentraciones
Identificar multimodalidad
Comparar densidades
4. dotplot_overlay(df, col)
Genera gráficos dot‐plot sobre una variable:
Representación punto a punto
Detección clara de outliers
Visualización compacta de dispersión

**Visualizaciones generadas**
El proyecto incluye:
Histogramas
Gráficos de dispersión (scatterplots)
KDE plots
Diagramas superpuestos
Estas figuras se encuentran en la carpeta reporte_ctg_outputs/ y también están integradas en el PDF final.

**Reporte final**
El archivo Reporte_Practica_3.pdf incluye:
Título y estructura profesional
Descripción técnica de funciones
Visualizaciones explicadas
Conclusiones
Recomendaciones analíticas

**Conclusiones principales**
El dataset presenta suficientes datos para modelado avanzado.
Se observan asimetrías y outliers en múltiples variables.
Existe alta correlación entre algunas variables (ej. LB y LBE).
Las densidades muestran comportamientos fisiológicos estables.
Se requiere imputación y normalización durante el preprocesamiento.

**Selección de atributos**
Remover variables altamente correlacionadas.
Usar ANOVA, chi-cuadrada y mutual information.
Modelado
Probar Random Forest, XGBoost y SVM.
Usar validación cruzada estratificada.
Análisis avanzado
Aplicar PCA para reducir dimensionalidad.
Usar clustering para detectar patrones naturales.


**Tecnologías utilizadas**
Python 3
Pandas
NumPy
Matplotlib
SciPy
ReportLab
Jupyter Notebook

**Cómo ejecutar el proyecto**
Clonar el repositorio:
git clone https://github.com/usuario/Práctica3-CTG.git
Instalar dependencias:
pip install -r requirements.txt
Abrir el notebook:
jupyter notebook MyrnaMiguel.ipynb
Para regenerar el PDF:
Ejecutar las celdas correspondientes en el notebook.
