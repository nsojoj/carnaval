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

    if "trends_ene_co" in df.columns:
        df = df.rename(columns={"trends_ene_co": "trends_ene"})

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

    kmeans_k4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_model["cluster"] = kmeans_k4.fit_predict(df_scaled)

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)
    df_pca_clusters = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
    df_pca_clusters["año"] = df_model["año"].values
    df_pca_clusters["cluster"] = df_model["cluster"].values

    kmeans_k3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_model["cluster_kmeans"] = kmeans_k3.fit_predict(df_scaled)

    medias = df_model.groupby("cluster_kmeans")["pax_feb"].mean().sort_values()
    mapa_etiquetas = {
        medias.index[0]: "BAJO",
        medias.index[1]: "MEDIO",
        medias.index[2]: "ALTO"
    }
    df_model["nivel_impacto"] = df_model["cluster_kmeans"].map(mapa_etiquetas)

    df_scaled_for_hac = df_scaled.drop(columns=["cluster", "cluster_kmeans", "nivel_impacto"], errors="ignore")
    linked_data = linkage(df_scaled_for_hac, method="ward")
    df_model["cluster_jerarquico"] = fcluster(linked_data, 4, criterion="maxclust")

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

    rf_model_importance = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_importance.fit(X, y)
    feature_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model_importance.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    reverse_map = {0: "ALTO", 1: "BAJO", 2: "MEDIO"}
    all_labels = sorted(y_train.unique())

    # CORREGIDO AQUÍ
    log_reg_model = LogisticRegression(random_state=42, solver="liblinear", max_iter=1000)
    log_reg_model.fit(X_train, y_train)
    y_pred_log_reg = log_reg_model.predict(X_test)

    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train, y_train)
    y_pred_decision_tree = decision_tree_model.predict(X_test)

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
with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Q - Question: misión analítica")
    st.write(
        "Bajo la metodología QUEST, el análisis comienza con una misión analítica clara: "
        "entender si el Carnaval presenta patrones históricos distinguibles y si es posible anticipar su nivel de impacto antes del evento."
    )

    preguntas_df = pd.DataFrame({
        "Preguntas clave": [
            "¿Existen grupos naturales de años según su desempeño turístico?",
            "¿Qué variables distinguen mejor un Carnaval ALTO de uno MEDIO?",
            "¿Es posible predecir el nivel con variables disponibles antes del evento?",
            "¿Qué tan confiable es la predicción con solo 13 observaciones?",
            "¿Qué podría esperarse para el Carnaval 2027?"
        ]
    })
    st.dataframe(preguntas_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Restricciones del análisis**
        - Solo 13 observaciones (2013–2025)
        - 2021 fue un Carnaval cancelado
        - Parte de la serie temprana fue estimada
        - El modelo debe usar variables disponibles antes del evento
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("U - Understand: auditoría del dataset")

    missing_percentages = df.isnull().sum() / len(df) * 100
    missing_percentages = missing_percentages[missing_percentages > 0]

    if missing_percentages.empty:
        st.success("No se identificaron valores faltantes: el dataset está completo al 100%.")
    else:
        missing_df = missing_percentages.reset_index()
        missing_df.columns = ["Variable", "Porcentaje de nulos"]
        fig_missing, ax_missing = plt.subplots(figsize=(10, 5))
        sns.barplot(x="Porcentaje de nulos", y="Variable", data=missing_df, palette="viridis", ax=ax_missing)
        ax_missing.set_title("Valores faltantes por variable")
        ax_missing.grid(axis="x", linestyle="--", alpha=0.7)
        st.pyplot(fig_missing)

    st.dataframe(df.describe().T, use_container_width=True)

    st.markdown(
        """
        **Hallazgos principales**
        1. `visitantes_carnaval` tiene un outlier extremo en 2021 por COVID.  
        2. `crecimiento_pax_yoy` presenta una caída fuerte en 2018 por la remodelación del aeropuerto.  
        3. `trm_feb_usdcop` muestra una tendencia creciente estable.  
        4. `trends_ene` presenta comportamiento irregular por la normalización de Google Trends.  
        5. No hay valores faltantes.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("E - Explore: análisis univariado")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Selecciona una variable", numeric_cols)

    fig_dist, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df[selected_col], kde=True, ax=axes[0])
    axes[0].set_title(f"Histograma de {selected_col}")
    sns.boxplot(y=df[selected_col], ax=axes[1])
    axes[1].set_title(f"Boxplot de {selected_col}")
    st.pyplot(fig_dist)

    st.write(
        "El perfilado univariado muestra una estructura general estable, pero con alta variabilidad en los indicadores turísticos. "
        "`pax_feb`, `visitantes_carnaval` y `ocup_hotel_feb` conservan patrones relativamente consistentes, "
        "mientras que `crecimiento_pax_yoy`, `gasto_prom_cop` y `ratio_visitantes_pax` reflejan mayor inestabilidad y sensibilidad a choques."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Correlación entre variables")
    corr = df.select_dtypes(include=np.number).corr()
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax_corr)
    ax_corr.set_title("Matriz de correlación")
    st.pyplot(fig_corr)

    st.write(
        "La matriz de correlación evidencia un núcleo fuerte de variables económicas y turísticas interrelacionadas, "
        "especialmente entre `gasto_prom_cop`, `ocup_hotel_feb`, `visitantes_carnaval` y `tur_int_colombia_miles`. "
        "En contraste, `trends_ene` y `crecimiento_pax_yoy` muestran menor capacidad explicativa."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("S - Study: hipótesis y resultados principales")

    hip_df = pd.DataFrame({
        "Hipótesis": [
            "H1. Febrero supera el promedio anual de pasajeros",
            "H2. Google Trends predice pasajeros de febrero",
            "H3. Los años post-pandemia superan a los pre-pandemia",
            "H4. El gasto promedio crece de forma sostenida",
            "H6. Google Trends se relaciona con el gasto",
            "H7. Pasajeros y gasto están relacionados",
            "H8. Existe ruptura estructural post-pandemia",
            "H9. Hay tendencia creciente del flujo",
            "H10. Existen valores atípicos"
        ],
        "Resultado": [
            "Confirmada",
            "Rechazada",
            "Confirmada",
            "Confirmada",
            "Rechazada",
            "Confirmada",
            "Confirmada",
            "Confirmada",
            "Confirmada"
        ]
    })
    st.dataframe(hip_df, use_container_width=True, hide_index=True)

    htabs = st.tabs(["H1-H2", "H3-H4", "H6-H7", "H8-H10"])

    with htabs[0]:
        fig_h1, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(df))
        w = 0.35

        axes[0].bar(x - w/2, df["pax_feb"], width=w, label="Febrero", color="#E74C3C", alpha=0.85)
        axes[0].bar(x + w/2, df["pax_promedio_año"], width=w, label="Promedio año", color="#3498DB", alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df["año"], rotation=45)
        axes[0].set_title("H1: pax_feb vs promedio anual")
        axes[0].legend()

        axes[1].bar(df["año"], df["efecto_carnaval"], color="#F39C12")
        axes[1].axhline(y=1, color="black", linestyle="--")
        axes[1].set_title("Efecto Carnaval")
        axes[1].set_xticks(df["año"])
        axes[1].set_xticklabels(df["año"], rotation=45)
        st.pyplot(fig_h1)

        corr_h2 = df["trends_ene"].corr(df["pax_feb"])
        st.write(
            f"**H1 confirmada:** febrero supera sistemáticamente el nivel promedio del año. "
            f"**H2 rechazada:** la correlación entre `trends_ene` y `pax_feb` es **{corr_h2:.2f}**, lo que indica una relación débil."
        )

    with htabs[1]:
        fig_h34, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(df["año"], df["visitantes_carnaval"], color="#2ECC71")
        axes[0].set_title("H3: visitantes del Carnaval")
        axes[0].set_xticks(df["año"])
        axes[0].set_xticklabels(df["año"], rotation=45)

        df_gasto = df[df["gasto_normalizado"] > 0]
        axes[1].plot(df_gasto["año"], df_gasto["gasto_normalizado"], marker="o", color="#E67E22")
        axes[1].set_title("H4: tendencia del gasto")
        axes[1].set_xticks(df_gasto["año"])
        axes[1].set_xticklabels(df_gasto["año"], rotation=45)
        st.pyplot(fig_h34)

        st.write(
            "El bloque H3-H4 muestra recuperación post-pandemia y crecimiento del gasto turístico. "
            "Esto sugiere que la expansión reciente del Carnaval no es solo en volumen, sino también en intensidad económica."
        )

    with htabs[2]:
        fig_h67, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.scatterplot(x="trends_ene", y="gasto_normalizado", data=df, ax=axes[0], color="#6a4c93")
        axes[0].set_title("H6: Trends vs gasto")

        sns.scatterplot(x="pax_feb", y="gasto_normalizado", data=df, ax=axes[1], color="#fb8500")
        axes[1].set_title("H7: pasajeros vs gasto")

        st.pyplot(fig_h67)

        corr_h6 = df["trends_ene"].corr(df["gasto_normalizado"])
        corr_h7 = df["pax_feb"].corr(df["gasto_normalizado"])

        st.write(
            f"**H6 rechazada:** `trends_ene` y gasto tienen una relación débil (**r = {corr_h6:.2f}**). "
            f"**H7 confirmada:** pasajeros y gasto presentan asociación positiva (**r = {corr_h7:.2f}**)."
        )

    with htabs[3]:
        fig_h8, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.lineplot(x="año", y="pax_feb", data=df, marker="o", color="#28B463", ax=axes[0])
        axes[0].axvspan(2019.5, 2021.5, color="red", alpha=0.12)
        axes[0].set_title("H8-H9: cambio estructural y tendencia")

        sns.boxplot(y=df["visitantes_carnaval"], ax=axes[1], color="lightcoral")
        axes[1].set_title("H10: outliers en visitantes")

        st.pyplot(fig_h8)

        slope, intercept, r_value, p_value, std_err = linregress(df["año"], df["pax_feb"])
        st.write(
            f"**H8 confirmada:** la pandemia introduce una ruptura estructural. "
            f"**H9 confirmada:** la pendiente de la tendencia de pasajeros es **{slope:.2f}** con **p = {p_value:.3f}**. "
            "**H10 confirmada:** 2021 aparece como outlier estructural."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("T - Tell: hallazgos ejecutivos")
    st.markdown(
        """
        1. El Carnaval presenta crecimiento estructural de largo plazo.  
        2. La TRM emerge como una variable estratégica del contexto.  
        3. La ocupación hotelera refleja con claridad la intensidad del evento.  
        4. 2021 debe tratarse como outlier estructural, no como tendencia.  
        5. Google Trends aporta una señal limitada y contraintuitiva.  
        6. El EDA justifica avanzar a clustering y clasificación.
        """
    )
    st.write(
        "La lectura integrada del QUEST muestra que el Carnaval combina crecimiento, choques externos y heterogeneidad suficiente entre años. "
        "Eso justifica tanto la segmentación histórica como la construcción de un modelo predictivo exploratorio."
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
        "El método del codo sugiere una zona plausible entre 4 y 6 grupos, mientras que la silueta señala que k=3 y k=4 son soluciones razonables. "
        "En el notebook se prioriza k=4 para descripción histórica y k=3 para la construcción de la variable objetivo."
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

    st.dataframe(outputs["cluster_means_k4"], use_container_width=True)
    st.write(
        "La solución de 4 clusters separa con bastante claridad el periodo pre-pandemia temprano, la transición 2018–2020, "
        "el año extremo 2021 y la recuperación post-pandemia. Esto aporta una lectura histórica interpretable del sistema."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("K-Means (k=3) y variable objetivo")
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

    st.dataframe(df_model[["año", "pax_feb", "efecto_carnaval", "nivel_impacto"]], use_container_width=True, hide_index=True)

    st.write(
        "La variable `nivel_impacto` surge del clustering con k=3 y se etiqueta como BAJO, MEDIO y ALTO según el promedio de `pax_feb`. "
        "Este paso conecta el descubrimiento no supervisado con la etapa de clasificación supervisada."
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

    st.dataframe(outputs["cluster_means_hac"], use_container_width=True)
    st.write(
        "El clustering jerárquico refuerza la existencia de grupos diferenciados y confirma que 2021 queda aislado como observación singular."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Importancia de variables")
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
        "La jerarquía de importancia sugiere que el modelo se apoya en una combinación de factores turísticos, macroeconómicos y estructurales. "
        "No existe una única variable mágica, sino un sistema de señales."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Evaluación de modelos")
    st.dataframe(outputs["model_results"].round(2), use_container_width=True)
    st.write(
        "El split usado en el notebook favorece a Árbol de Decisión y Random Forest frente a Regresión Logística. "
        "Aun así, debe recordarse que con n=13 las métricas son sensibles a cualquier partición."
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
        f"Según el notebook, el mejor modelo en esta corrida fue **{outputs['best_model_name']}**. "
        f"Con los datos hipotéticos definidos para 2027, el nivel predicho es **{outputs['pred_clase']}**."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Nivel de impacto predicho", outputs["pred_clase"])
    with c2:
        st.write(
            "Esta salida no debe interpretarse como pronóstico definitivo, sino como una demostración del pipeline. "
            "La muestra es pequeña y la predicción depende de supuestos hipotéticos."
        )

    fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
    colors = ["#6a4c93", "#fb8500", "#d62828"][:len(outputs["proba_df"])]
    ax_pred.bar(outputs["proba_df"].index, outputs["proba_df"]["Probabilidad"], color=colors)
    ax_pred.set_title(f"Predicciones para 2027 ({outputs['best_model_name']})")
    ax_pred.set_xlabel("Nivel de Impacto")
    ax_pred.set_ylabel("Probabilidad")
    ax_pred.set_ylim(0, 1)
    for i, v in enumerate(outputs["proba_df"]["Probabilidad"].values):
        ax_pred.text(i, v + 0.02, f"{v:.2f}", ha="center")
    st.pyplot(fig_pred)

    st.write(
        "Más que afirmar una verdad cerrada sobre 2027, esta pestaña muestra cómo una organización podría convertir señales tempranas en una estimación formal de escenario."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab6:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Conclusiones generales")
    st.markdown(
        """
        - El Carnaval presenta crecimiento estructural de largo plazo.  
        - 2021 debe tratarse como outlier estructural.  
        - Las variables reales del sistema turístico explican mejor el impacto que las señales digitales aisladas.  
        - Existen grupos históricos diferenciables de ediciones del Carnaval.  
        - Es posible construir un modelo predictivo exploratorio, pero con cautela por el tamaño muestral.  
        - El proyecto tiene valor práctico para stakeholders reales de la ciudad.
        """
    )
    st.write(
        "El principal aporte del trabajo no es solo la métrica final del modelo, sino la construcción de una base integrada, una narrativa analítica clara y un pipeline que traduce un problema de ciudad en una herramienta basada en datos."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Limitaciones")
    st.markdown(
        """
        - n = 13 observaciones  
        - Años tempranos parcialmente estimados  
        - 2021 introduce una ruptura extrema  
        - Algunas variables son útiles para EDA pero no para predicción anticipada  
        - Las métricas deben interpretarse con prudencia
        """
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
    colors = ["#6a4c93", "#fb8500", "#d62828"][:len(outputs["proba_df"])]
    ax_pred.bar(outputs["proba_df"].index, outputs["proba_df"]["Probabilidad"], color=colors)
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
