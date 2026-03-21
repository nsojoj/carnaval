import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mticker

from scipy.stats import linregress
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Carnaval de Barranquilla",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fff7e6 0%, #fffdf7 45%, #f8fbff 100%);
    }
    .hero {
        background: linear-gradient(135deg, #ffb703 0%, #fb8500 35%, #d62828 70%, #6a4c93 100%);
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        color: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
        margin-bottom: 1.2rem;
    }
    .section-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .highlight {
        background: linear-gradient(90deg, rgba(255,183,3,0.18), rgba(251,133,0,0.10));
        padding: 0.8rem 1rem;
        border-radius: 14px;
        border-left: 6px solid #fb8500;
        margin: 0.5rem 0 1rem 0;
    }
    .small-note {
        font-size: 0.95rem;
        color: #444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_number(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)


@st.cache_data
def load_data():
    df = pd.read_csv("dataset_final_carnaval_fe-6.csv")

    # Renombrar como en el notebook
    if "trends_ene_co" in df.columns:
        df = df.rename(columns={"trends_ene_co": "trends_ene"})

    # Variables creadas manualmente en el notebook
    new_cols_data = {
        "año": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "pax_enero": [14000, 15000, 16000, 17000, 17894, 12446, 16433, 17923, 17312, 22000, 25000, 27000, 29000],
        "pax_dic_anterior": [13500, 14500, 15500, 16500, 17894, 13501, 16679, 19388, 14361, 21000, 24000, 26000, 28000],
        "pax_promedio_año": [10000, 10000, 10000, 15618, 13699, 12540, 16296, 6961, 15695, 24000, 20000, 22000, 23000],
        "tendencia_anual": [-5.0, 10.0, 15.0, 20.0, -30.83, 65.42, 22.65, -27.64, 43.58, 10.0, 12.0, 15.0, 18.0],
        "trends_dic_ant": [25, 20, 28, 32, 30, 7, 6, 7, 2, 10, 15, 18, 22],
        "momentum_trends": [1.0, 1.2, 1.5, 1.8, 1.0, 4.1429, 3.3333, 3.1429, 2.5, 3.0, 3.2, 3.5, 3.8]
    }

    df_indexed = df.set_index("año").copy()

    for col_name, data_list in new_cols_data.items():
        if col_name != "año":
            df_indexed[col_name] = pd.Series(data_list, index=new_cols_data["año"])

    # Derivadas del notebook
    df_indexed["gasto_normalizado"] = df_indexed["gasto_prom_cop"] / 10000
    df_indexed["efecto_carnaval"] = df_indexed["pax_feb"] / df_indexed["pax_promedio_año"]

    df = df_indexed.reset_index()
    return df


@st.cache_data
def prepare_model_outputs(df):
    df_model = df.copy()

    # ========= CLUSTERING =========
    df_clustering = df_model.copy()
    numerical_cols = df_clustering.select_dtypes(include=["number"]).columns
    cols_to_exclude = [
        "año", "cluster", "cluster_kmeans", "nivel_impacto_encoded",
        "cluster_k3", "cluster_k4", "cluster_jerarquico"
    ]
    features_for_scaling = [col for col in numerical_cols if col not in cols_to_exclude]
    df_clustering_numerical = df_clustering[features_for_scaling]

    scaler_cluster = StandardScaler()
    df_scaled_array = scaler_cluster.fit_transform(df_clustering_numerical)
    df_scaled = pd.DataFrame(df_scaled_array, columns=df_clustering_numerical.columns)

    # Elbow + silhouette
    inertia_values = []
    silhouette_scores = []
    max_clusters = len(df_scaled) - 1

    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertia_values.append(kmeans.inertia_)
        if k > 1:
            score = silhouette_score(df_scaled, kmeans.labels_)
            silhouette_scores.append({"k": k, "score": score})

    silhouette_df = pd.DataFrame(silhouette_scores)

    # KMeans k=4
    kmeans_k4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_model["cluster"] = kmeans_k4.fit_predict(df_scaled)

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)
    df_pca_clusters = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
    df_pca_clusters["año"] = df_model["año"].values
    df_pca_clusters["cluster"] = df_model["cluster"].values

    # KMeans k=3 para nivel de impacto
    kmeans_k3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_model["cluster_kmeans"] = kmeans_k3.fit_predict(df_scaled)

    medias = df_model.groupby("cluster_kmeans")["pax_feb"].mean().sort_values()
    mapa_etiquetas = {
        medias.index[0]: "BAJO",
        medias.index[1]: "MEDIO",
        medias.index[2]: "ALTO"
    }
    df_model["nivel_impacto"] = df_model["cluster_kmeans"].map(mapa_etiquetas)

    # HAC
    df_scaled_for_hac = df_scaled.drop(columns=["cluster", "cluster_kmeans", "nivel_impacto"], errors="ignore")
    linked_data = linkage(df_scaled_for_hac, method="ward")
    df_model["cluster_jerarquico"] = fcluster(linked_data, 4, criterion="maxclust")

    # Medias de clusters
    cluster_means_k4 = df_model.drop(columns=["cluster_kmeans", "nivel_impacto"], errors="ignore").groupby("cluster").mean(numeric_only=True)
    cluster_means_hac = df_model.drop(columns=["cluster", "cluster_kmeans", "nivel_impacto"], errors="ignore").groupby("cluster_jerarquico").mean(numeric_only=True)

    # ========= CLASIFICACIÓN =========
    encoder = LabelEncoder()
    df_model["nivel_impacto_encoded"] = encoder.fit_transform(df_model["nivel_impacto"])

    exclude_cols = ["año", "cluster", "cluster_kmeans", "cluster_jerarquico", "nivel_impacto_encoded"]
    numerical_cols_full = df_model.select_dtypes(include=["int64", "float64"]).columns.tolist()
    X_cols = [col for col in numerical_cols_full if col not in exclude_cols]
    X = df_model[X_cols]
    y = df_model["nivel_impacto_encoded"]

    # Importancia de variables
    rf_model_importance = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_importance.fit(X, y)
    feature_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model_importance.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    # División train/test como en el notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    reverse_map = {0: "ALTO", 1: "BAJO", 2: "MEDIO"}
    all_labels = sorted(y_train.unique())

    # Logistic Regression
    log_reg_model = LogisticRegression(random_state=42, solver="liblinear", multi_class="auto")
    log_reg_model.fit(X_train, y_train)
    y_pred_log_reg = log_reg_model.predict(X_test)

    # Decision Tree
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train, y_train)
    y_pred_decision_tree = decision_tree_model.predict(X_test)

    # Random Forest
    random_forest_model = RandomForestClassifier(random_state=42)
    random_forest_model.fit(X_train, y_train)
    y_pred_random_forest = random_forest_model.predict(X_test)

    def get_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
        }

    model_results = pd.DataFrame({
        "Logistic Regression": get_metrics(y_test, y_pred_log_reg),
        "Decision Tree": get_metrics(y_test, y_pred_decision_tree),
        "Random Forest": get_metrics(y_test, y_pred_random_forest)
    }).T

    best_model_name = model_results["F1-Score"].idxmax()
    if best_model_name == "Logistic Regression":
        best_model = log_reg_model
    elif best_model_name == "Decision Tree":
        best_model = decision_tree_model
    else:
        best_model = random_forest_model

    # Datos hipotéticos 2027 como en notebook
    datos_2027_unscaled = pd.DataFrame({
        "pax_feb": [22000],
        "crecimiento_pax_yoy": [0.25],
        "visitantes_carnaval": [700000],
        "gasto_prom_cop": [900000],
        "ratio_visitantes_pax": [33.0],
        "ocup_hotel_feb": [95.0],
        "trm_feb_usdcop": [4300],
        "tur_int_colombia_miles": [4200],
        "trends_ene": [20],
        "pax_enero": [28000],
        "pax_dic_anterior": [22000],
        "pax_promedio_año": [23000],
        "tendencia_anual": [-10.0],
        "trends_dic_ant": [6],
        "momentum_trends": [3.5],
        "gasto_normalizado": [90.0],
        "efecto_carnaval": [0.95]
    })

    datos_2027_unscaled = datos_2027_unscaled[X.columns]

    scaler_prediction = StandardScaler()
    scaler_prediction.fit(X)
    datos_2027_scaled = pd.DataFrame(
        scaler_prediction.transform(datos_2027_unscaled),
        columns=X.columns
    )

    pred_clase_encoded = best_model.predict(datos_2027_scaled)
    pred_proba = best_model.predict_proba(datos_2027_scaled)
    pred_clase = reverse_map[pred_clase_encoded[0]]

    proba_df = pd.DataFrame(
        pred_proba,
        columns=[reverse_map[i] for i in best_model.classes_]
    ).T
    proba_df.columns = ["Probabilidad"]
    proba_df = proba_df.sort_values(by="Probabilidad", ascending=False)

    return {
        "df_model": df_model,
        "df_scaled": df_scaled,
        "silhouette_df": silhouette_df,
        "inertia_values": inertia_values,
        "df_pca_clusters": df_pca_clusters,
        "linked_data": linked_data,
        "cluster_means_k4": cluster_means_k4,
        "cluster_means_hac": cluster_means_hac,
        "feature_importance_df": feature_importance_df,
        "model_results": model_results,
        "best_model_name": best_model_name,
        "pred_clase": pred_clase,
        "proba_df": proba_df,
        "y_test": y_test,
        "y_pred_log_reg": y_pred_log_reg,
        "y_pred_decision_tree": y_pred_decision_tree,
        "y_pred_random_forest": y_pred_random_forest,
        "all_labels": all_labels,
        "reverse_map": reverse_map,
        "X": X
    }


df = load_data()
outputs = prepare_model_outputs(df)

df_model = outputs["df_model"]

# ========= HERO =========
st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.3rem;">Carnaval de Barranquilla: analítica del impacto turístico</h1>
        <p style="font-size:1.08rem; margin-bottom:0.35rem;">
            Análisis del comportamiento histórico del Carnaval de Barranquilla mediante técnicas de exploración,
            segmentación y predicción del impacto turístico.
        </p>
        <p style="margin-bottom:0.2rem;"><b>Programa:</b> Maestría en Analítica de Datos</p>
        <p style="margin-bottom:0;"><b>Integrantes:</b> Mario Orozco · Rosa Mora · Natalia Sojo · Donnys Torres</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Años analizados", int(df["año"].nunique()))
c2.metric("Máx. pasajeros en febrero", format_number(df["pax_feb"].max()))
c3.metric("Variables del dataset base", df.shape[1])

st.markdown(
    """
    <div class="highlight">
        Este proyecto busca responder una pregunta concreta de planeación pública y privada:
        ¿es posible anticipar si una edición del Carnaval tendrá un impacto ALTO, MEDIO o BAJO
        antes de que el evento comience?
    </div>
    """,
    unsafe_allow_html=True,
)

# ========= TABS =========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Presentación",
    "Datos y variables",
    "EDA QUEST",
    "Modelo",
    "Predicción",
    "Conclusiones"
])

with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Contexto")
    st.write(
        "El Carnaval de Barranquilla, Patrimonio Oral e Intangible de la Humanidad, no es solo una fiesta cultural: "
        "es uno de los principales motores económicos de la ciudad. En 2025 generó más de **$880.000 millones de pesos**, "
        "movilizó alrededor de **193.000 empleos** y recibió cerca de **800.000 visitantes de 15 países**. "
        "Esto lo convierte en un fenómeno turístico, económico y logístico de gran escala."
    )
    st.write(
        "La relevancia del proyecto no está únicamente en describir el Carnaval, sino en su capacidad para ayudar a planificarlo. "
        "La Alcaldía, el gremio hotelero, Carnaval S.A.S. BIC, comerciantes y aerolíneas toman decisiones costosas meses antes del evento: "
        "seguridad, capacidad hotelera, escenarios, inventarios y operación aérea. Un Carnaval planificado como BAJO cuando en realidad resulta ALTO "
        "puede generar saturación, mala experiencia de visitante y presión sobre la ciudad. A la inversa, planificar como ALTO un Carnaval que resulta BAJO "
        "genera costos innecesarios."
    )
    st.write(
        "Hoy muchas de esas decisiones se toman con intuición, experiencia acumulada o señales parciales. "
        "Este proyecto propone una alternativa: **tomar decisiones con datos**."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Stakeholders y momento de decisión")
    stakeholders = pd.DataFrame({
        "Stakeholder": [
            "Alcaldía de Barranquilla",
            "COTELCO Atlántico",
            "Carnaval SAS BIC",
            "Comerciantes",
            "Aerolíneas"
        ],
        "Lo que necesita saber": [
            "¿Cuántos policías y logísticos contratar?",
            "¿Cuántas habitaciones preparar y a qué precio?",
            "¿Cuántos eventos y escenarios montar?",
            "¿Cuánto inventario pedir?",
            "¿Cuántos pasajeros recibirá?"
        ],
        "Cuándo": [
            "Nov-Dic anterior",
            "Oct-Nov anterior",
            "Sep-Oct anterior",
            "Dic-Ene",
            "Sep-Nov anterior"
        ]
    })
    st.dataframe(stakeholders, use_container_width=True, hide_index=True)
    st.write(
        "El problema compartido entre todos ellos es el mismo: deben decidir con antelación, cuando el comportamiento final del Carnaval todavía no se ha materializado. "
        "Por eso, un modelo útil no puede depender de variables observadas después del evento, sino de señales disponibles previamente."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Objetivo del estudio")
    st.write(
        "Desarrollar un pipeline de **clustering + clasificación** que permita identificar patrones históricos en el desempeño turístico del Carnaval "
        "y predecir si una futura edición tendrá un nivel de impacto **ALTO, MEDIO o BAJO**."
    )
    st.markdown(
        """
        **Objetivos específicos**
        
        1. Identificar grupos naturales de años del Carnaval según su comportamiento turístico.  
        2. Determinar qué variables distinguen mejor un Carnaval ALTO de uno MEDIO o BAJO.  
        3. Entrenar modelos de clasificación con variables disponibles antes del evento.  
        4. Explorar el comportamiento del Carnaval bajo la metodología QUEST y evaluar la robustez de las conclusiones con una muestra pequeña.  
        5. Obtener una predicción exploratoria del Carnaval 2027.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Pregunta de investigación")
    st.markdown(
        """
        > **¿Podemos predecir si un Carnaval será ALTO, MEDIO o BAJO antes de que comience, usando señales turísticas, macroeconómicas y digitales observables con anterioridad?**
        """
    )
    st.write(
        "Esta pregunta articula tanto el EDA como la modelación: primero entender la estructura histórica del fenómeno, luego segmentar las ediciones y finalmente entrenar un modelo que pueda anticipar niveles de impacto."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Por qué empezamos en 2013")
    st.write(
        "La serie histórica del aeropuerto BAQ muestra un **quiebre estructural** entre 2011 y 2013. Antes de ese punto, el aeropuerto registraba aproximadamente entre 3.000 y 6.000 pasajeros mensuales; "
        "después de 2013 comienza a operar en una escala superior a 10.000 pasajeros. Entre las explicaciones plausibles están la entrada de aerolíneas de bajo costo y la transformación operativa del aeropuerto."
    )
    st.write(
        "Comparar periodos tan distintos introduciría una ruptura de escala que haría menos consistente la lectura del fenómeno. Por eso, el análisis toma como universo comparable el periodo 2013–2025."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Fuentes utilizadas")
    fuentes = pd.DataFrame({
        "Variable": [
            "pax_feb",
            "visitantes_carnaval (2017-2025)",
            "visitantes_carnaval (2013-2016)",
            "gasto_prom_cop",
            "ocup_hotel_feb",
            "trm_feb_usdcop",
            "tur_int_colombia_miles",
            "trends_ene"
        ],
        "Fuente principal": [
            "Aerocivil - datos.gov.co",
            "Alcaldía BQ / Carnaval SAS BIC",
            "Estimación ratio + prensa",
            "Alcaldía / CCB + deflación IPC DANE",
            "COTELCO Atlántico vía prensa",
            "Banco de la República",
            "Migración Colombia",
            "Google Trends Colombia"
        ],
        "Cobertura": [
            "2013-2025",
            "2017-2025",
            "2013-2016",
            "2013-2025",
            "2013-2025",
            "2013-2025",
            "2013-2025",
            "2013-2025"
        ],
        "Calidad": [
            "Real",
            "Oficial",
            "Estimado",
            "Oficial/Estimado",
            "Oficial/Prensa",
            "Real",
            "Real",
            "Real"
        ]
    })
    st.dataframe(fuentes, use_container_width=True, hide_index=True)
    st.write(
        "Una de las principales fortalezas metodológicas del proyecto es que no trabaja con una sola fuente, sino con un cruce de fuentes heterogéneas. "
        "Eso permitió construir una base integrada del fenómeno turístico del Carnaval, pero también implicó resolver inconsistencias metodológicas y estimar algunos años donde no existían datos oficiales completos."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Cómo se resolvieron los datos faltantes")
    st.write(
        "**Visitantes 2013-2016:** no existían cifras oficiales, así que se estimaron a partir del ratio promedio entre visitantes y pasajeros aéreos de febrero observado entre 2016 y 2019. "
        "Ese ratio fue cercano a **23.91 visitantes por cada pasajero aéreo**, lo cual refleja la importancia del ingreso por vía terrestre."
    )
    st.write(
        "**Gasto promedio 2013-2016:** se reconstruyó mediante deflación inversa usando IPC desde 2017, primer año con valor oficial de referencia. "
        "El resultado se contrastó con reportes secundarios para validar coherencia."
    )
    st.write(
        "**Ocupación hotelera 2013-2016:** se obtuvo mediante levantamiento manual en prensa económica y regional, en especial reportes de COTELCO Atlántico replicados por medios."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Vista general del dataset")
    st.dataframe(df, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.write(f"**Filas:** {df.shape[0]}")
    c2.write(f"**Columnas (base + derivadas):** {df.shape[1]}")
    st.write(
        "El dataset original tiene **13 observaciones (2013–2025)** y 10 variables base. "
        "Posteriormente se añadieron variables derivadas para apoyar el análisis exploratorio y la modelación."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Descripción de variables")
    variables = pd.DataFrame({
        "Variable": [
            "pax_feb",
            "crecimiento_pax_yoy",
            "visitantes_carnaval",
            "gasto_prom_cop",
            "ratio_visitantes_pax",
            "ocup_hotel_feb",
            "trm_feb_usdcop",
            "tur_int_colombia_miles",
            "trends_ene"
        ],
        "Rol": [
            "Indicador central de flujo externo",
            "Feature engineering",
            "Target analítico del turismo total",
            "Variable de EDA, no predictiva",
            "Feature engineering con leakage",
            "Indicador de saturación de ciudad",
            "Variable macroeconómica",
            "Contexto internacional del destino",
            "Señal digital pre-evento"
        ],
        "Descripción": [
            "Pasajeros que llegaron al aeropuerto Ernesto Cortissoz en febrero; termómetro del flujo turístico aéreo.",
            "Variación porcentual del tráfico aéreo respecto al año anterior.",
            "Total de personas de fuera de Barranquilla que llegaron al Carnaval.",
            "Gasto promedio por turista en COP. Se usa para análisis descriptivo, no para predicción práctica.",
            "Visitantes totales por cada pasajero aéreo; útil para entender estructura de transporte, pero no entra al modelo por leakage.",
            "Porcentaje de habitaciones ocupadas durante el Carnaval; refleja intensidad de demanda hotelera.",
            "Precio del dólar en COP en febrero; aproxima competitividad del destino para extranjeros.",
            "Turistas internacionales que visitaron Colombia ese año; captura contexto externo favorable o desfavorable.",
            "Interés de búsqueda de 'carnaval barranquilla' en Google Colombia durante enero."
        ]
    })
    st.dataframe(variables, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Feature engineering")
    st.write(
        "Dos variables del dataset no se descargaron de ninguna fuente: fueron calculadas explícitamente."
    )
    st.markdown(
        """
        - **crecimiento_pax_yoy**: variación porcentual de pasajeros aéreos de febrero respecto al año anterior.  
          Fórmula: `(pax_feb[t] - pax_feb[t-1]) / pax_feb[t-1]`
        
        - **ratio_visitantes_pax**: cuántos visitantes totales llegaron por cada pasajero aéreo.  
          Fórmula: `visitantes_carnaval / pax_feb`
        """
    )
    st.write(
        "Además, en el notebook se añadieron variables auxiliares como `pax_enero`, `pax_dic_anterior`, `pax_promedio_año`, `efecto_carnaval`, `tendencia_anual`, "
        "`trends_dic_ant`, `momentum_trends` y `gasto_normalizado`, con fines de análisis, segmentación y modelado."
    )
    st.write(
        "En términos metodológicos, no todas las variables derivadas entran al modelo predictivo final. "
        "Por ejemplo, `gasto_prom_cop` es una variable post-evento y `ratio_visitantes_pax` contiene información del target, por lo que deben interpretarse con cautela cuando el objetivo es predicción anticipada."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Q - Question: misión analítica")
    st.write(
        "En la metodología QUEST, el análisis no comienza con una gráfica, sino con una pregunta de negocio y una misión analítica concreta. "
        "En este caso, la misión fue identificar si el comportamiento turístico del Carnaval presenta patrones históricos segmentables y si es posible "
        "anticipar el nivel de impacto antes de que el evento ocurra."
    )
    preguntas_df = pd.DataFrame({
        "Preguntas que debe responder el análisis": [
            "¿Existen grupos naturales de años según su desempeño turístico?",
            "¿Qué variables distinguen mejor un Carnaval ALTO de uno MEDIO?",
            "¿Es posible predecir el nivel con variables disponibles antes del evento?",
            "¿Qué tan confiable es esa predicción con solo 13 observaciones?",
            "¿Qué le espera al Carnaval 2027?"
        ]
    })
    st.dataframe(preguntas_df, use_container_width=True, hide_index=True)

    criterios_df = pd.DataFrame({
        "Criterio": ["Mínimo", "Objetivo", "Excelente", "Interpretabilidad"],
        "Definición": [
            "Accuracy LOO >= 80% en al menos dos algoritmos",
            "Accuracy LOO >= 85% promedio",
            "Accuracy LOO >= 90% en al menos uno",
            "Explicable a un alcalde en 2 minutos"
        ]
    })
    st.dataframe(criterios_df, use_container_width=True, hide_index=True)

    st.write(
        "Desde el inicio también se reconocieron restricciones importantes: solo hay **13 ediciones** comparables, 2021 es un año de cancelación, "
        "parte de la serie temprana es estimada y el modelo debe construirse con variables útiles antes del evento. "
        "Estas restricciones no invalidan el ejercicio, pero sí obligan a interpretar con prudencia cualquier resultado predictivo."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("U - Understand: auditoría de los datos")
    st.write(
        "Antes de explorar relaciones, se auditó la estructura y calidad del dataset. El objetivo fue identificar valores faltantes, variables con comportamiento irregular "
        "y posibles limitaciones heredadas de la construcción de la base."
    )

    missing_percentages = df.isnull().sum() / len(df) * 100
    missing_percentages = missing_percentages[missing_percentages > 0]

    if missing_percentages.empty:
        st.success("No se identificaron valores faltantes: el dataset está completo al 100%.")
    else:
        missing_df = missing_percentages.reset_index()
        missing_df.columns = ["Variable", "Porcentaje de Nulos"]
        fig_missing, ax_missing = plt.subplots(figsize=(12, 6))
        sns.barplot(x="Porcentaje de Nulos", y="Variable", data=missing_df, palette="viridis", ax=ax_missing)
        ax_missing.set_title("Porcentaje de valores faltantes por variable")
        ax_missing.grid(axis="x", linestyle="--", alpha=0.7)
        st.pyplot(fig_missing)

    st.dataframe(df.describe().T, use_container_width=True)

    st.markdown(
        """
        **Hallazgos clave de auditoría**
        1. `visitantes_carnaval`: outlier extremo en 2021 (valor 0, COVID).  
        2. `crecimiento_pax_yoy`: valor muy negativo en 2018 (remodelación del aeropuerto).  
        3. `trm_feb_usdcop`: tendencia creciente clara, sin outliers, refleja macroeconomía.  
        4. `trends_ene`: distribución irregular por normalización de Google Trends.  
        5. Sin valores faltantes: dataset completo al 100%.
        """
    )
    st.write(
        "La auditoría confirma que el dataset es usable y coherente, pero no homogéneo en el sentido estadístico: varios indicadores están atravesados por choques estructurales, "
        "en particular la pandemia y ciertos eventos operativos del aeropuerto. Esto anticipa la presencia de asimetrías y outliers que luego efectivamente se observan en la exploración."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("E - Explore: perfilado univariado")
    st.write(
        "En esta etapa se analizaron las variables individualmente para detectar dispersión, estabilidad, asimetrías y comportamiento temporal. "
        "La idea no es solo ver su forma estadística, sino entender qué cuentan sobre la dinámica del Carnaval."
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Selecciona una variable para ver su histograma y boxplot", numeric_cols)

    fig_dist, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig_dist.suptitle(f"Distribution of {selected_col}", fontsize=16)

    sns.histplot(df[selected_col], kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram of {selected_col}")
    axes[0].set_xlabel(selected_col)
    axes[0].set_ylabel("Frequency")

    sns.boxplot(y=df[selected_col], ax=axes[1])
    axes[1].set_title(f"Boxplot of {selected_col}")
    axes[1].set_ylabel(selected_col)

    st.pyplot(fig_dist)

    st.write(
        "El análisis univariado confirma una estructura general estable, pero con alta variabilidad en los indicadores turísticos. "
        "Variables como **pax_feb**, **visitantes_carnaval** y **ocup_hotel_feb** muestran rangos amplios pero relativamente consistentes; "
        "por el contrario, **crecimiento_pax_yoy** evidencia gran inestabilidad, con años de expansión y contracción pronunciada."
    )
    st.write(
        "También se observan asimetrías importantes en **gasto_prom_cop** y **ratio_visitantes_pax**, especialmente alrededor de periodos de disrupción. "
        "En conjunto, esto refuerza la idea de que la dinámica del turismo asociado al Carnaval no sigue una distribución normal simple, sino que combina regularidad estructural con choques específicos."
    )
    st.markdown(
        """
        **Lecturas puntuales destacadas del EDA**
        - **TRM (dólar):** es la variable más equilibrada; se mueve de manera relativamente ordenada y sin saltos aberrantes.
        - **Turismo internacional:** presenta tendencia positiva, pero incluye un valor atípico muy bajo en pandemia.
        - **Google Trends:** concentra valores en un rango relativamente acotado y su comportamiento depende de la normalización del índice.
        - **Conclusión univariada:** el turismo es la dimensión más sensible a choques, mientras las variables macro muestran trayectorias más limpias.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Correlación entre variables")
    corr = df.select_dtypes(include=np.number).corr()
    fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax_corr)
    ax_corr.set_title("Correlation Matrix of Numerical Variables", fontsize=16)
    st.pyplot(fig_corr)

    st.write(
        "La matriz de correlación muestra un núcleo fuerte de variables interrelacionadas en torno al comportamiento económico-turístico del Carnaval. "
        "En especial, **gasto_prom_cop**, **ocup_hotel_feb** y **visitantes_carnaval** presentan asociaciones altas, lo que sugiere que mayor afluencia de turistas "
        "empuja simultáneamente el gasto y la demanda hotelera."
    )
    st.write(
        "También destaca el papel de **tur_int_colombia_miles**, que parece actuar como contexto externo favorable al desempeño del Carnaval. "
        "En contraste, **crecimiento_pax_yoy** presenta correlaciones más débiles y **trends_ene** muestra relaciones bajas o incluso negativas con algunas variables, "
        "lo que sugiere una desconexión parcial entre la señal digital y la dinámica turística efectiva."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("S - Study: hipótesis, análisis bivariado y descubrimiento")
    hip_df = pd.DataFrame({
        "#": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10"],
        "Hipótesis": [
            "El flujo de pasajeros en febrero es significativamente mayor que el promedio del año",
            "El interés en Google en enero predice el flujo de pasajeros en febrero",
            "Los años post-pandemia tienen mayor flujo que los pre-pandemia",
            "El gasto promedio del turista ha crecido sostenidamente cada año",
            "Existen al menos 3 grupos claramente diferenciados de ediciones del Carnaval",
            "A mayor interés en Google, mayor gasto promedio del turista",
            "Existe relación entre el flujo de turistas y el gasto promedio",
            "El Carnaval presenta un cambio estructural después de la pandemia",
            "El flujo de pasajeros en febrero presenta una tendencia creciente en el tiempo",
            "Existen valores atípicos en años específicos"
        ],
        "Resultado esperado": [
            "Febrero > promedio anual",
            "Relación positiva",
            "Post > Pre",
            "Tendencia creciente",
            "3 segmentos identificables",
            "Relación positiva",
            "Relación positiva",
            "Ruptura en tendencia",
            "Tendencia creciente",
            "Identificación de anomalías"
        ]
    })
    st.dataframe(hip_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    subtabs = st.tabs(["H1-H2", "H3-H4", "H6-H7", "H8-H10"])

    with subtabs[0]:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("H1. Pasajeros de febrero vs promedio anual")
        fig_h1, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(df))
        w = 0.35
        axes[0].bar(x - w/2, df["pax_feb"], width=w, label="Febrero (Carnaval)", color="#E74C3C", alpha=0.85)
        axes[0].bar(x + w/2, df["pax_promedio_año"], width=w, label="Promedio del año", color="#3498DB", alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df["año"], rotation=45)
        axes[0].set_title("Pasajeros febrero vs promedio anual\n(Verificación H1)")
        axes[0].set_ylabel("Pasajeros")
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        axes[0].legend()
        if 2020 in df["año"].values:
            idx_2020 = df.index[df["año"] == 2020][0]
            axes[0].annotate(
                "COVID\n2020",
                xy=(idx_2020, df.loc[df["año"] == 2020, "pax_promedio_año"].values[0]),
                xytext=(idx_2020, 5000),
                ha="center",
                fontsize=8,
                color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8)
            )

        colores_barra = ["#E74C3C" if e > 1 else "#3498DB" for e in df["efecto_carnaval"]]
        axes[1].bar(df["año"], df["efecto_carnaval"], color=colores_barra, alpha=0.85)
        axes[1].axhline(y=1, color="black", linestyle="--", linewidth=1.2, label="Sin efecto (ratio=1)")
        axes[1].set_title("Efecto Carnaval por año\n(pax_feb / pax_promedio_año)")
        axes[1].set_ylabel("Ratio")
        axes[1].set_xticks(df["año"])
        axes[1].set_xticklabels(df["año"], rotation=45)
        axes[1].legend()
        for año, ef in zip(df["año"], df["efecto_carnaval"]):
            axes[1].text(año, ef + 0.03, f"{ef:.2f}", ha="center", fontsize=8)

        st.pyplot(fig_h1)

        efecto_promedio = df["efecto_carnaval"].mean()
        st.write(
            f"La hipótesis **H1 se confirma**. En promedio, febrero se sitúa en **{efecto_promedio:.2f} veces** el nivel base del año, lo que valida que el Carnaval introduce una intensificación turística real."
        )
        st.write(
            "La excepción visual más fuerte es 2020, donde el efecto Carnaval fue particularmente alto antes de que la pandemia alterara el sistema. "
            "Este hallazgo es importante porque justifica tratar febrero como mes núcleo del análisis."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("H2. Google Trends enero vs flujo de pasajeros en febrero")
        fig_h2, ax_h2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="trends_ene", y="pax_feb", data=df, s=100, hue="año", palette="viridis", ax=ax_h2)
        ax_h2.set_title("Interés en Google en Enero vs. Flujo de Pasajeros en Febrero (H2)")
        ax_h2.set_xlabel("Interés en Google (trends_ene)")
        ax_h2.set_ylabel("Pasajeros en Febrero (pax_feb)")
        ax_h2.grid(True, linestyle="--", alpha=0.7)
        for i in range(len(df)):
            ax_h2.text(df["trends_ene"].iloc[i] + 0.5, df["pax_feb"].iloc[i], str(df["año"].iloc[i]), fontsize=9)
        st.pyplot(fig_h2)

        corr_h2 = df["trends_ene"].corr(df["pax_feb"])
        st.write(
            f"La hipótesis **H2 se rechaza**. La correlación observada es de **{corr_h2:.2f}**, valor cercano a cero. "
            "Esto indica que el interés de búsqueda en enero no predice de forma robusta el flujo real de pasajeros en febrero."
        )
        st.write(
            "Este resultado es particularmente valioso porque rompe una intuición común: más búsquedas no significan necesariamente más visitantes. "
            "En este caso, Google Trends parece capturar atención, pero no comportamiento turístico efectivo."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with subtabs[1]:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("H3. Recuperación post-pandemia y H4. crecimiento del gasto")
        fig_h34, axes = plt.subplots(1, 2, figsize=(15, 7))

        colores_vis = ["#E74C3C" if a in [2021] else ("#F39C12" if a == 2020 else "#2ECC71") for a in df["año"]]
        axes[0].bar(df["año"], df["visitantes_carnaval"], color=colores_vis, alpha=0.85)
        axes[0].set_title("Visitantes totales al Carnaval\n(H3 — crecimiento post-pandemia)")
        axes[0].set_ylabel("Visitantes")
        axes[0].set_xticks(df["año"])
        axes[0].set_xticklabels(df["año"], rotation=45)
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))
        for año, vis in zip(df["año"], df["visitantes_carnaval"]):
            if vis > 0:
                axes[0].text(año, vis + 5000, f"{vis/1000:.0f}K", ha="center", fontsize=8)
        axes[0].annotate(
            "Cancelado", xy=(2021, 0), xytext=(2021, 30000),
            ha="center", fontsize=8, color="#E74C3C",
            arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=0.8)
        )

        df_gasto = df[df["gasto_normalizado"] > 0]
        axes[1].plot(
            df_gasto["año"], df_gasto["gasto_normalizado"] / 1e6,
            marker="o", color="#E67E22", linewidth=2.5, markersize=8
        )
        axes[1].fill_between(df_gasto["año"], df_gasto["gasto_normalizado"] / 1e6, alpha=0.15, color="#E67E22")
        axes[1].set_title("Gasto promedio del turista (millones COP)\n(H4 — crecimiento sostenido)")
        axes[1].set_ylabel("Millones COP")
        axes[1].set_xticks(df_gasto["año"])
        axes[1].set_xticklabels(df_gasto["año"], rotation=45)
        axes[1].grid(axis="y", alpha=0.3)
        for año, gasto in zip(df_gasto["año"], df_gasto["gasto_normalizado"]):
            axes[1].text(año, gasto/1e6 + 0.03, f"${gasto/1e6:.1f}M", ha="center", fontsize=8)

        st.pyplot(fig_h34)

        crecimiento_h3 = (df.loc[df["año"] == 2025, "visitantes_carnaval"].values[0] / df.loc[df["año"] == 2022, "visitantes_carnaval"].values[0] - 1) * 100
        crecimiento_h4 = (df.loc[df["año"] == 2025, "gasto_normalizado"].values[0] / df.loc[df["año"] == 2013, "gasto_normalizado"].values[0] - 1) * 100

        st.write(
            f"**H3 confirmada:** entre 2022 y 2025 los visitantes crecieron aproximadamente **{crecimiento_h3:.1f}%**, evidenciando una recuperación fuerte del Carnaval posterior a la pandemia."
        )
        st.write(
            f"**H4 confirmada:** el gasto promedio muestra una trayectoria creciente de largo plazo, con un aumento aproximado de **{crecimiento_h4:.1f}%** entre 2013 y 2025."
        )
        st.write(
            "Ambas hipótesis, vistas en conjunto, muestran que el Carnaval no solo se recuperó en volumen de visitantes, sino también en capacidad de movilizar gasto económico por turista."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with subtabs[2]:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("H6. Google Trends vs gasto")
        fig_h6, ax_h6 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="trends_ene", y="gasto_normalizado", data=df, s=100, hue="año", palette="viridis", ax=ax_h6)
        ax_h6.set_title("Interés en Google en Enero vs. Gasto Normalizado (H6)")
        ax_h6.set_xlabel("Interés en Google (trends_ene)")
        ax_h6.set_ylabel("Gasto Normalizado")
        ax_h6.grid(True, linestyle="--", alpha=0.7)
        for i in range(len(df)):
            ax_h6.text(df["trends_ene"].iloc[i] + 0.5, df["gasto_normalizado"].iloc[i], str(df["año"].iloc[i]), fontsize=9)
        st.pyplot(fig_h6)

        corr_h6 = df["trends_ene"].corr(df["gasto_normalizado"])
        st.write(
            f"La hipótesis **H6 se rechaza**. La correlación es **{corr_h6:.2f}**, lo que indica una relación débil o nula entre el interés digital y el gasto promedio."
        )
        st.write(
            "Esto sugiere que la señal de búsqueda previa no captura adecuadamente la capacidad de gasto del turista. "
            "En otras palabras, la atención digital no se traduce automáticamente en mayor derrama económica."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("H7. Flujo de pasajeros vs gasto turístico")
        fig_h7, ax_h7 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="pax_feb", y="gasto_normalizado", data=df, s=100, hue="año", palette="viridis", ax=ax_h7)
        ax_h7.set_title("Flujo de Pasajeros en Febrero vs. Gasto Normalizado (H7)")
        ax_h7.set_xlabel("Pasajeros en Febrero (pax_feb)")
        ax_h7.set_ylabel("Gasto Normalizado")
        ax_h7.grid(True, linestyle="--", alpha=0.7)
        for i in range(len(df)):
            ax_h7.text(df["pax_feb"].iloc[i] + 0.5, df["gasto_normalizado"].iloc[i], str(df["año"].iloc[i]), fontsize=9)
        st.pyplot(fig_h7)

        corr_h7 = df["pax_feb"].corr(df["gasto_normalizado"])
        st.write(
            f"La hipótesis **H7 se confirma**. La correlación entre flujo de pasajeros y gasto es de **{corr_h7:.2f}**, una asociación positiva moderada a fuerte."
        )
        st.write(
            "Esto significa que el aumento en el volumen de turistas se relaciona directamente con una mayor capacidad de gasto, reforzando la conexión entre demanda y dinamismo económico."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with subtabs[3]:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("H8. Cambio estructural post-pandemia")
        fig_h8, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        fig_h8.suptitle("Análisis de Cambio Estructural Post-Pandemia (H8)", fontsize=16)

        sns.lineplot(x="año", y="pax_feb", data=df, marker="o", color="skyblue", ax=axes[0])
        sns.scatterplot(x="año", y="pax_feb", data=df, s=100, hue="año", palette="viridis", legend=False, ax=axes[0])
        axes[0].set_title("Flujo de Pasajeros en Febrero (pax_feb)")
        axes[0].set_ylabel("Pasajeros en Febrero")
        axes[0].grid(True, linestyle="--", alpha=0.7)
        axes[0].axvspan(2019.5, 2021.5, color="red", alpha=0.15)
        axes[0].text(2020.5, axes[0].get_ylim()[1] * 0.9, "Pandemia", horizontalalignment="center", color="red", fontsize=10)

        sns.lineplot(x="año", y="visitantes_carnaval", data=df, marker="o", color="lightcoral", ax=axes[1])
        sns.scatterplot(x="año", y="visitantes_carnaval", data=df, s=100, hue="año", palette="viridis", legend=False, ax=axes[1])
        axes[1].set_title("Visitantes al Carnaval")
        axes[1].set_ylabel("Número de Visitantes")
        axes[1].grid(True, linestyle="--", alpha=0.7)
        axes[1].axvspan(2019.5, 2021.5, color="red", alpha=0.15)
        axes[1].text(2020.5, axes[1].get_ylim()[1] * 0.9, "Pandemia", horizontalalignment="center", color="red", fontsize=10)

        sns.lineplot(x="año", y="gasto_normalizado", data=df, marker="o", color="lightgreen", ax=axes[2])
        sns.scatterplot(x="año", y="gasto_normalizado", data=df, s=100, hue="año", palette="viridis", legend=False, ax=axes[2])
        axes[2].set_title("Gasto Normalizado del Turista")
        axes[2].set_ylabel("Gasto Normalizado")
        axes[2].set_xlabel("Año")
        axes[2].grid(True, linestyle="--", alpha=0.7)
        axes[2].axvspan(2019.5, 2021.5, color="red", alpha=0.15)
        axes[2].text(2020.5, axes[2].get_ylim()[1] * 0.9, "Pandemia", horizontalalignment="center", color="red", fontsize=10)

        st.pyplot(fig_h8)

        st.write(
            "La hipótesis **H8 se confirma**. Existe una ruptura estructural evidente entre 2020 y 2021, seguida de un proceso de recuperación. "
            "La pandemia no fue una simple oscilación, sino un quiebre del patrón histórico del Carnaval."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("H9. Tendencia del flujo de pasajeros")
        fig_h9, ax_h9 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x="año", y="pax_feb", data=df, marker="o", color="#28B463", ax=ax_h9)
        sns.scatterplot(x="año", y="pax_feb", data=df, s=100, hue="año", palette="viridis", legend=False, ax=ax_h9)
        ax_h9.set_title("Tendencia del Flujo de Pasajeros en Febrero a lo largo del Tiempo (H9)")
        ax_h9.set_xlabel("Año")
        ax_h9.set_ylabel("Pasajeros en Febrero (pax_feb)")
        ax_h9.grid(True, linestyle="--", alpha=0.7)
        ax_h9.set_xticks(df["año"])
        for i in range(len(df)):
            ax_h9.text(df["año"].iloc[i], df["pax_feb"].iloc[i] + 500, str(df["año"].iloc[i]), ha="center", fontsize=9)
        st.pyplot(fig_h9)

        slope, intercept, r_value, p_value, std_err = linregress(df["año"], df["pax_feb"])
        st.write(
            f"La hipótesis **H9 se confirma**. La pendiente estimada es **{slope:.2f}**, con **p = {p_value:.3f}** y **R² = {r_value**2:.2f}**. "
            "Esto indica una tendencia creciente estadísticamente significativa del flujo de pasajeros en febrero."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("H10. Detección de valores atípicos")
        columns_for_outliers = ["pax_feb", "visitantes_carnaval", "gasto_normalizado"]
        fig_h10, axes = plt.subplots(1, len(columns_for_outliers), figsize=(15, 6))
        for i, col in enumerate(columns_for_outliers):
            sns.boxplot(y=df[col], color="lightcoral", ax=axes[i])
            sns.stripplot(y=df[col], x=df["año"], jitter=True, color="black", size=5, ax=axes[i])
            axes[i].set_title(f"Boxplot de {col}")
            axes[i].set_ylabel(col)
            axes[i].set_xlabel("Año")
            axes[i].tick_params(axis="x", rotation=45)
            axes[i].grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig_h10)

        st.write(
            "La hipótesis **H10 se confirma**. El año 2021 aparece como outlier estructural en múltiples variables: flujo, visitantes y gasto. "
            "No es ruido estadístico, sino la marca de una disrupción real del sistema."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("T - Tell: hallazgos ejecutivos del QUEST")
    st.markdown(
        """
        1. **Crecimiento estructural:** el Carnaval más que duplicó/triplicó su escala turística entre 2013 y 2025.  
        2. **La TRM emerge como variable estratégica:** cuando el peso se debilita, Colombia se vuelve más atractiva para visitantes extranjeros.  
        3. **Los hoteles no mienten:** la ocupación hotelera refleja de forma muy limpia la intensidad real del Carnaval.  
        4. **COVID 2021 es un outlier estructural, no una tendencia.**  
        5. **Google Trends ofrece un hallazgo contraintuitivo:** más búsquedas no significan necesariamente más visitantes.  
        6. **Existe sustento empírico para avanzar al modelado:** las hipótesis confirmadas permiten justificar clustering y clasificación.
        """
    )
    st.write(
        "En conjunto, el QUEST deja una base interpretativa robusta: el Carnaval es un sistema turístico con crecimiento de largo plazo, "
        "interrupciones fuertes ante choques exógenos y una estructura suficientemente heterogénea como para justificar su segmentación en grupos y su posterior modelado predictivo."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Determinación del número óptimo de clusters")
    fig_elbow, axes = plt.subplots(1, 2, figsize=(12, 5))

    max_clusters = len(outputs["df_scaled"]) - 1
    axes[0].plot(range(1, max_clusters + 1), outputs["inertia_values"], marker="o", linestyle="--")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method for Optimal K")
    axes[0].grid(True)

    sil_df = outputs["silhouette_df"]
    axes[1].plot(sil_df["k"], sil_df["score"], marker="o", linestyle="--")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score for Optimal K")
    axes[1].grid(True)

    st.pyplot(fig_elbow)

    st.write(
        "El método del codo no mostró un quiebre totalmente nítido, pero sugirió una región plausible entre k=4 y k=6. "
        "El coeficiente de silueta, por su parte, indicó que **k=3** y **k=4** eran soluciones razonables."
    )
    st.write(
        "A partir del notebook se exploraron ambas alternativas, pero la solución de **k=4** se consideró más interpretable para describir la evolución histórica del Carnaval."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("K-Means (k=4) con PCA")
    fig_pca, ax_pca = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="cluster",
        size="año",
        sizes=(50, 400),
        palette="viridis",
        data=outputs["df_pca_clusters"],
        legend="full",
        ax=ax_pca
    )
    for _, row in outputs["df_pca_clusters"].iterrows():
        ax_pca.text(row["PC1"] + 0.1, row["PC2"] + 0.1, str(row["año"]), fontsize=9)
    ax_pca.set_title("Clusters of Carnival Editions (PCA)", fontsize=16)
    ax_pca.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig_pca)

    st.write(
        "La proyección PCA permite ver que las ediciones del Carnaval no forman una nube homogénea. "
        "Se distinguen grupos asociados a un periodo pre-pandemia temprano, una transición 2018–2020, el año extremo 2021 y la recuperación fuerte de 2022–2025."
    )
    st.dataframe(outputs["cluster_means_k4"], use_container_width=True)

    st.markdown(
        """
        **Caracterización de clusters K-Means (k=4)**
        - **Cluster 0 (2013-2017):** periodo pre-pandemia temprano y estable.  
        - **Cluster 1 (2022-2025):** periodo post-pandemia de recuperación y crecimiento robusto.  
        - **Cluster 2 (2018-2020):** transición/anomalía temprana, incluyendo el comportamiento singular de 2020.  
        - **Cluster 3 (2021):** año de impacto severo por pandemia y cancelación.
        """
    )
    st.write(
        "Esta solución resulta muy valiosa porque separa con claridad el año de cancelación total y evita mezclarlo con periodos que, aunque cercanos temporalmente, responden a lógicas turísticas completamente distintas."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Clustering K-Means (k=3) y variable nivel_impacto")
    fig_compare, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.scatterplot(
        x="año", y="nivel_impacto", hue="nivel_impacto",
        data=df_model, palette="viridis", s=100, ax=axes[0]
    )
    axes[0].set_title("K-Means Clustering (k=3)")
    axes[0].grid(True, linestyle="--", alpha=0.7)

    sns.scatterplot(
        x="año", y="cluster", hue="cluster",
        data=df_model, palette="viridis", s=100, ax=axes[1]
    )
    axes[1].set_title("K-Means Clustering (k=4)")
    axes[1].grid(True, linestyle="--", alpha=0.7)

    st.pyplot(fig_compare)

    resultado_cluster = df_model[["año", "pax_feb", "efecto_carnaval", "nivel_impacto"]].copy()
    st.dataframe(resultado_cluster, use_container_width=True, hide_index=True)

    st.write(
        "Para construir la variable objetivo del modelo de clasificación, se aplicó K-Means con **k=3** y se etiquetaron los clusters como **BAJO**, **MEDIO** y **ALTO** según el promedio de `pax_feb`."
    )
    st.write(
        "Esto permite pasar de una segmentación no supervisada a una clasificación supervisada. En términos simples: el clustering identifica patrones de años; "
        "luego la clasificación aprende a reconocerlos."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Clustering jerárquico")
    fig_dendro, ax_dendro = plt.subplots(figsize=(15, 8))
    dendrogram(
        outputs["linked_data"],
        orientation="top",
        labels=df_model["año"].values.astype(str),
        distance_sort="descending",
        show_leaf_counts=True,
        ax=ax_dendro
    )
    ax_dendro.set_title("Dendrograma de Clustering Jerárquico para Años del Carnaval")
    ax_dendro.set_xlabel("Año del Carnaval")
    ax_dendro.set_ylabel("Distancia Euclidiana (Ward)")
    ax_dendro.axhline(y=3.5, color="r", linestyle="--", label="Corte para 3 clusters")
    ax_dendro.axhline(y=2.0, color="g", linestyle="--", label="Corte para 4 clusters")
    ax_dendro.legend()
    st.pyplot(fig_dendro)

    st.write(
        "El clustering jerárquico confirma la estructura general observada con K-Means: existe un bloque pre-pandemia temprano, un bloque de recuperación post-pandemia "
        "y un aislamiento claro del año 2021. La solución de 4 grupos ofrece una granularidad más interpretativa que la de 3 grupos."
    )
    st.dataframe(outputs["cluster_means_hac"], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Importancia de variables para la predicción")
    fig_imp, ax_imp = plt.subplots(figsize=(12, 7))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=outputs["feature_importance_df"],
        palette="viridis",
        ax=ax_imp
    )
    ax_imp.set_title("Importancia de Variables para Predecir el Nivel de Impacto del Carnaval (Random Forest)")
    ax_imp.grid(axis="x", linestyle="--", alpha=0.7)
    st.pyplot(fig_imp)

    st.write(
        "La importancia de variables sugiere que el modelo no distingue los niveles de impacto con una sola señal, sino con una combinación de factores turísticos, estructurales y macroeconómicos. "
        "En línea con el análisis descriptivo, variables como flujo, ocupación, gasto y contexto económico aparecen entre las más relevantes."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Entrenamiento y evaluación de modelos de clasificación")
    st.dataframe(outputs["model_results"].round(2), use_container_width=True)

    st.write(
        "Se entrenaron tres modelos: **Regresión Logística**, **Árbol de Decisión** y **Random Forest**. "
        "La comparación muestra que **Árbol de Decisión** y **Random Forest** alcanzan el mejor rendimiento en este split, con métricas superiores a la Regresión Logística."
    )
    st.write(
        "Sin embargo, esta comparación debe leerse con cuidado: el tamaño del conjunto es extremadamente pequeño y el set de prueba contiene solo unas pocas observaciones. "
        "Por lo tanto, los resultados no deben interpretarse como evidencia definitiva de generalización."
    )

    cm_tabs = st.tabs(["Logistic Regression", "Decision Tree", "Random Forest"])

    def plot_cm(y_true, y_pred, labels, reverse_map, title):
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        tick_labels = [reverse_map[i] for i in labels]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        st.pyplot(fig)

    with cm_tabs[0]:
        plot_cm(outputs["y_test"], outputs["y_pred_log_reg"], outputs["all_labels"], outputs["reverse_map"], "Matriz de confusión - Logistic Regression")
    with cm_tabs[1]:
        plot_cm(outputs["y_test"], outputs["y_pred_decision_tree"], outputs["all_labels"], outputs["reverse_map"], "Matriz de confusión - Decision Tree")
    with cm_tabs[2]:
        plot_cm(outputs["y_test"], outputs["y_pred_random_forest"], outputs["all_labels"], outputs["reverse_map"], "Matriz de confusión - Random Forest")

    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Predicción exploratoria para 2027")
    st.write(
        f"De acuerdo con el notebook, el mejor modelo según F1-Score fue **{outputs['best_model_name']}**. "
        f"Con los datos hipotéticos definidos para 2027, la clasificación resultante fue **{outputs['pred_clase']}**."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Nivel de impacto predicho para 2027", outputs["pred_clase"])
    with c2:
        st.write(
            "Esta predicción debe interpretarse como un ejercicio exploratorio. No constituye una inferencia robusta ni una proyección definitiva, "
            "porque depende de un conjunto pequeño, de variables parcialmente estimadas y de una evaluación sobre un set de prueba muy reducido."
        )

    fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
    ax_pred.bar(outputs["proba_df"].index, outputs["proba_df"]["Probabilidad"], color=["#6a4c93", "#fb8500", "#d62828"][:len(outputs["proba_df"])])
    ax_pred.set_title(f"Predicciones para 2027 ({outputs['best_model_name']})")
    ax_pred.set_xlabel("Nivel de Impacto")
    ax_pred.set_ylabel("Probabilidad")
    ax_pred.set_ylim(0, 1)
    for i, v in enumerate(outputs["proba_df"]["Probabilidad"].values):
        ax_pred.text(i, v + 0.02, f"{v:.2f}", ha="center")
    st.pyplot(fig_pred)

    st.write(
        "El propio notebook advierte que esta predicción debe tomarse con cautela. De hecho, aunque el modelo seleccionado puede producir una salida formal para 2027, "
        "la tendencia observada en el EDA sugiere que el Carnaval viene en expansión y que, sustantivamente, podría esperarse un comportamiento alto si se mantienen las condiciones recientes."
    )
    st.write(
        "Por tanto, el valor de esta pestaña no es tanto afirmar una verdad definitiva sobre 2027, sino mostrar cómo un pipeline de datos puede apoyar una conversación anticipada sobre escenarios."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab6:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Conclusiones generales")
    st.write(
        "El análisis confirma que el Carnaval de Barranquilla puede estudiarse como un sistema turístico estructurado, con patrones de largo plazo, choques exógenos y diferencias claras entre ediciones. "
        "La evidencia reunida a lo largo del QUEST, del clustering y de la clasificación permite extraer varias conclusiones relevantes."
    )
    st.markdown(
        """
        **1. El Carnaval presenta crecimiento estructural de largo plazo**  
        Entre 2013 y 2025 el evento aumentó de forma marcada su capacidad de atracción, tanto en pasajeros como en visitantes y gasto turístico.

        **2. La pandemia fue un punto de quiebre real**  
        2021 aparece sistemáticamente como outlier estructural. No debe interpretarse como parte de una tendencia normal, sino como un evento extraordinario.

        **3. Las variables reales del sistema turístico explican mejor el impacto que las señales digitales aisladas**  
        El flujo de pasajeros, la ocupación hotelera, el gasto y el contexto macroeconómico muestran relaciones más fuertes que Google Trends por sí solo.

        **4. Existen grupos históricos diferenciables de ediciones del Carnaval**  
        Tanto K-Means como el clustering jerárquico muestran que el Carnaval no se comporta igual cada año. La segmentación por periodos tiene sentido empírico.

        **5. Es posible construir un modelo predictivo, pero con cautela**  
        Sí se puede entrenar una clasificación ALTO/MEDIO/BAJO, pero con 13 observaciones la interpretación debe ser prudente. El ejercicio es útil como prototipo analítico, no como sistema definitivo.

        **6. El proyecto tiene valor aplicado para stakeholders reales**  
        Alcaldía, hoteleros, organizadores, comerciantes y aerolíneas podrían beneficiarse de una herramienta que traduzca señales tempranas en una estimación de impacto.
        """
    )
    st.write(
        "En síntesis, el proyecto demuestra que la intuición y la experiencia pueden complementarse con evidencia cuantitativa. "
        "No reemplaza la decisión humana, pero sí la vuelve más informada."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Limitaciones del estudio")
    st.markdown(
        """
        - Tamaño de muestra reducido (**n = 13**).  
        - Presencia de años estimados en la primera parte de la serie.  
        - 2021 introduce un outlier estructural extremo.  
        - Algunas variables solo son útiles para EDA y no para predicción anticipada.  
        - La evaluación de modelos con tan pocas observaciones no permite inferencias fuertes de generalización.
        """
    )
    st.write(
        "Estas limitaciones no invalidan el análisis, pero sí obligan a ser muy transparentes en la forma de comunicar los resultados. "
        "El mayor valor del trabajo está en la integración de fuentes, la formalización del problema y la construcción de un pipeline explicable."
    )
    st.markdown("</div>", unsafe_allow_html=True)
