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
        background: rgba(255,255,255,0.90);
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

    # Variables manuales usadas en el notebook
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
    df = df_indexed.reset_index()
    df["efecto_carnaval"] = df["pax_feb"] / df["pax_promedio_año"]

    return df


@st.cache_data
def prepare_model_outputs(df):
    df_model = df.copy()

    # ========= CLUSTERING =========
    df_clustering = df_model.copy()
    numerical_cols = df_clustering.select_dtypes(include=["number"]).columns

    cols_to_exclude = [
        "año",
        "cluster",
        "cluster_kmeans",
        "nivel_impacto_encoded",
        "cluster_k3",
        "cluster_k4",
        "cluster_jerarquico"
    ]

    features_for_scaling = [col for col in numerical_cols if col not in cols_to_exclude]
    df_clustering_numerical = df_clustering[features_for_scaling]

    scaler = StandardScaler()
    df_scaled_array = scaler.fit_transform(df_clustering_numerical)
    df_scaled = pd.DataFrame(df_scaled_array, columns=df_clustering_numerical.columns)

    inertia_values = []
    silhouette_scores = []
    max_clusters = len(df_scaled) - 1

    for k in range(1, max_clusters + 1):
        # En el notebook original está n_init='auto' para esta parte.
        # Se usa 10 por compatibilidad y estabilidad en GCP.
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertia_values.append(kmeans.inertia_)
        if k > 1:
            score = silhouette_score(df_scaled, kmeans.labels_)
            silhouette_scores.append({"k": k, "score": score})

    silhouette_df = pd.DataFrame(silhouette_scores)

    # KMeans k=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    df_model["cluster"] = kmeans.labels_

    # PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)

    df_pca_clusters = pd.DataFrame(data=pca_components, columns=["PC1", "PC2"])
    df_pca_clusters["año"] = df_model["año"]
    df_pca_clusters["cluster"] = df_model["cluster"]

    cluster_means_k4 = df_model.groupby("cluster").mean(numeric_only=True)

    # KMeans k=3 para nivel_impacto
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_model["cluster_kmeans"] = kmeans.fit_predict(df_scaled)

    medias = df_model.groupby("cluster_kmeans")["pax_feb"].mean().sort_values()
    mapa_etiquetas = {
        medias.index[0]: "BAJO",
        medias.index[1]: "MEDIO",
        medias.index[2]: "ALTO"
    }
    df_model["nivel_impacto"] = df_model["cluster_kmeans"].map(mapa_etiquetas)

    resultado_k3 = df_model[["año", "pax_feb", "efecto_carnaval", "nivel_impacto"]].copy()

    # HAC
    df_scaled_for_hac = df_scaled.drop(columns=["cluster", "cluster_kmeans", "nivel_impacto"], errors="ignore")
    linked_data = linkage(df_scaled_for_hac, method="ward")
    hierarchical_clusters = fcluster(linked_data, 4, criterion="maxclust")
    df_model["cluster_jerarquico"] = hierarchical_clusters

    hierarchical_cluster_means = df_model.drop(
        columns=["cluster", "cluster_kmeans", "nivel_impacto"],
        errors="ignore"
    ).groupby("cluster_jerarquico").mean(numeric_only=True)

    # ========= CLASIFICACIÓN =========
    encoder = LabelEncoder()
    df_model["nivel_impacto_encoded"] = encoder.fit_transform(df_model["nivel_impacto"])

    exclude_cols = ["año", "cluster", "cluster_kmeans", "cluster_jerarquico", "nivel_impacto_encoded"]
    numerical_cols = df_model.select_dtypes(include=["int64", "float64"]).columns.tolist()
    X_cols = [col for col in numerical_cols if col not in exclude_cols]
    X = df_model[X_cols]
    y = df_model["nivel_impacto_encoded"]

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    feature_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    reverse_map = {0: "ALTO", 1: "BAJO", 2: "MEDIO"}
    all_labels = sorted(y_train.unique())

    # Ajuste técnico para GCP: sin multi_class='auto'
    log_reg_model = LogisticRegression(random_state=42, solver="liblinear")
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

    # ========= PREDICCIÓN 2027 =========
    datos_2027_unscaled = pd.DataFrame({
        "pax_feb": [22000],
        "pax_enero": [28000],
        "pax_dic_anterior": [22000],
        "pax_promedio_año": [23000],
        "efecto_carnaval": [0.95],
        "tendencia_anual": [-10.0],
        "trends_dic_ant": [6],
        "trends_ene": [20],
        "visitantes_carnaval": [700000],
        "crecimiento_pax_yoy": [0.25],
        "momentum_trends": [3.5],
        "ratio_visitantes_pax": [33.0],
        "gasto_normalizado": [90.0]
    }, columns=X_train.columns)

    scaler_pred = StandardScaler()
    scaler_pred.fit(X)

    datos_2027_scaled_array = scaler_pred.transform(datos_2027_unscaled)
    datos_2027_scaled = pd.DataFrame(datos_2027_scaled_array, columns=X.columns)

    pred_clase_encoded = best_model.predict(datos_2027_scaled)
    pred_proba = best_model.predict_proba(datos_2027_scaled)

    pred_clase = [reverse_map[i] for i in pred_clase_encoded]
    model_classes = best_model.classes_

    proba_df = pd.DataFrame(pred_proba, columns=[reverse_map[i] for i in model_classes]).T
    proba_df.columns = ["Probabilidad"]
    proba_df = proba_df.sort_values(by="Probabilidad", ascending=False)

    return {
        "df_model": df_model,
        "df_scaled": df_scaled,
        "silhouette_df": silhouette_df,
        "inertia_values": inertia_values,
        "df_pca_clusters": df_pca_clusters,
        "cluster_means_k4": cluster_means_k4,
        "resultado_k3": resultado_k3,
        "linked_data": linked_data,
        "hierarchical_cluster_means": hierarchical_cluster_means,
        "feature_importance_df": feature_importance_df,
        "model_results": model_results,
        "best_model_name": best_model_name,
        "pred_clase": pred_clase[0],
        "proba_df": proba_df,
        "y_test": y_test,
        "y_pred_log_reg": y_pred_log_reg,
        "y_pred_decision_tree": y_pred_decision_tree,
        "y_pred_random_forest": y_pred_random_forest,
        "all_labels": all_labels,
        "reverse_map": reverse_map,
        "X_cols": X_cols
    }


def plot_conf_matrix(y_true, y_pred, labels, reverse_map, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tick_labels = [reverse_map[i] for i in labels]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    st.pyplot(fig)


df = load_data()
outputs = prepare_model_outputs(df)
df_model = outputs["df_model"]

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.3rem;">Carnaval de Barranquilla: analítica del impacto turístico</h1>
        <p style="font-size:1.05rem; margin-bottom:0.35rem;">
            Proyecto de maestría orientado a describir, segmentar y predecir el impacto turístico del Carnaval.
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
c3.metric("Variables del dataset base", 10)

st.markdown(
    """
    <div class="highlight">
        El objetivo es identificar patrones históricos del Carnaval y evaluar si es posible
        anticipar un nivel de impacto ALTO, MEDIO o BAJO antes del evento.
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
        "El Carnaval de Barranquilla es uno de los principales motores culturales y económicos de la ciudad. "
        "En 2025 generó más de **$880.000 millones**, alrededor de **193.000 empleos** y cerca de **800.000 visitantes**."
    )
    st.write(
        "El reto para actores como Alcaldía, hoteleros, organizadores, comerciantes y aerolíneas es tomar decisiones meses antes del evento, cuando todavía no se conoce su magnitud final."
    )
    st.write(
        "Este proyecto busca apoyar esa planeación con datos, mediante un pipeline de **clustering + clasificación**."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Pregunta y objetivo")
    st.markdown(
        """
        > **¿Podemos predecir si un Carnaval será ALTO, MEDIO o BAJO antes de que comience, usando señales turísticas, macroeconómicas y digitales observables previamente?**
        """
    )
    st.write(
        "El objetivo es identificar patrones históricos del Carnaval y construir un modelo que permita anticipar niveles de impacto turístico."
    )
    st.write(
        "El análisis comienza en 2013 porque antes de ese punto el aeropuerto BAQ operaba en una escala diferente, lo que afecta la comparabilidad histórica."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Fuentes y construcción de la base")
    fuentes = pd.DataFrame({
        "Variable": [
            "pax_feb",
            "visitantes_carnaval",
            "gasto_prom_cop",
            "ocup_hotel_feb",
            "trm_feb_usdcop",
            "tur_int_colombia_miles",
            "trends_ene"
        ],
        "Fuente principal": [
            "Aerocivil",
            "Alcaldía / Carnaval SAS BIC / estimación",
            "Alcaldía / CCB + IPC",
            "COTELCO Atlántico / prensa",
            "Banco de la República",
            "Migración Colombia",
            "Google Trends"
        ]
    })
    st.dataframe(fuentes, use_container_width=True, hide_index=True)
    st.write(
        "La base integra fuentes reales, oficiales y estimadas. Los años 2013–2016 requirieron reconstrucción parcial para variables como visitantes, gasto y ocupación hotelera."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Variables principales")
    variables = pd.DataFrame({
        "Variable": [
            "pax_feb", "crecimiento_pax_yoy", "visitantes_carnaval", "gasto_prom_cop",
            "ratio_visitantes_pax", "ocup_hotel_feb", "trm_feb_usdcop",
            "tur_int_colombia_miles", "trends_ene"
        ],
        "Significado": [
            "Pasajeros que llegaron al aeropuerto BAQ en febrero.",
            "Variación porcentual del tráfico aéreo respecto al año anterior.",
            "Total de personas de fuera de Barranquilla que llegaron al Carnaval.",
            "Gasto promedio por turista en COP. Se usa para EDA, no como predictor práctico.",
            "Visitantes por cada pasajero aéreo. Útil para entender estructura de transporte, pero con leakage para predicción.",
            "Porcentaje de habitaciones ocupadas durante Carnaval.",
            "Precio del dólar en COP en febrero.",
            "Turistas internacionales que visitaron Colombia ese año.",
            "Interés de búsqueda de 'carnaval barranquilla' en Google en enero."
        ]
    })
    st.dataframe(variables, use_container_width=True, hide_index=True)
    st.write(
        "Además del dataset base, el notebook añadió variables auxiliares para análisis y modelado: `pax_enero`, `pax_dic_anterior`, `pax_promedio_año`, `efecto_carnaval`, `tendencia_anual`, `trends_dic_ant`, `momentum_trends` y `gasto_normalizado`."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Q - Question")
    st.write(
        "La misión analítica fue determinar si existen patrones históricos de desempeño turístico y si es posible anticipar el impacto del Carnaval usando información previa al evento."
    )
    st.markdown(
        """
        **Preguntas guía**
        - ¿Existen grupos naturales de años?
        - ¿Qué variables distinguen mejor los niveles de impacto?
        - ¿Se puede predecir el nivel con información previa?
        - ¿Qué tan confiable es con n=13?
        - ¿Qué podría esperarse para 2027?
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("U - Understand")
    if df.isnull().sum().sum() == 0:
        st.success("El dataset no presenta valores faltantes.")
    st.dataframe(df.describe().T, use_container_width=True)
    st.write(
        "La auditoría confirma una base completa y coherente, aunque heterogénea: 2021 aparece como outlier estructural y variables como `crecimiento_pax_yoy` reflejan choques específicos del sistema."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("E - Explore")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Selecciona una variable", numeric_cols)

    fig_dist, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df[selected_col], kde=True, ax=axes[0])
    axes[0].set_title(f"Histograma de {selected_col}")
    sns.boxplot(y=df[selected_col], ax=axes[1])
    axes[1].set_title(f"Boxplot de {selected_col}")
    st.pyplot(fig_dist)

    corr = df.select_dtypes(include=np.number).corr()
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax_corr)
    ax_corr.set_title("Matriz de correlación")
    st.pyplot(fig_corr)

    st.write(
        "El análisis univariado y de correlación muestra alta variabilidad en el sistema turístico, relaciones fuertes entre visitantes, ocupación y gasto, y menor capacidad explicativa de Google Trends."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("S - Study")
    hip_df = pd.DataFrame({
        "Hipótesis": [
            "H1. Febrero supera el promedio anual",
            "H2. Google Trends predice pasajeros",
            "H3. Post-pandemia supera pre-pandemia",
            "H4. El gasto crece de forma sostenida",
            "H6. Google Trends se relaciona con gasto",
            "H7. Pasajeros y gasto se relacionan",
            "H8. Existe cambio estructural post-pandemia",
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

    subtabs = st.tabs(["H1", "H2", "H3-H4", "H6-H7", "H8-H10"])

    with subtabs[0]:
        st.markdown("<div class='small-note'>H1. Flujo de pasajeros en febrero vs promedio anual</div>", unsafe_allow_html=True)
        fig_h1, axes = plt.subplots(1, 2, figsize=(15, 5))

        x = np.arange(len(df))
        w = 0.35
        axes[0].bar(x - w/2, df["pax_feb"], width=w, label="Febrero (Carnaval)", color="#E74C3C", alpha=0.85)
        axes[0].bar(x + w/2, df["pax_promedio_año"], width=w, label="Promedio del año", color="#3498DB", alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df["año"], rotation=45)
        axes[0].set_title("Pasajeros febrero vs promedio anual")
        axes[0].legend()

        colores_barra = ["#E74C3C" if e > 1 else "#3498DB" for e in df["efecto_carnaval"]]
        axes[1].bar(df["año"], df["efecto_carnaval"], color=colores_barra, alpha=0.85)
        axes[1].axhline(y=1, color="black", linestyle="--")
        axes[1].set_title("Efecto Carnaval")
        axes[1].set_xticks(df["año"])
        axes[1].set_xticklabels(df["año"], rotation=45)

        st.pyplot(fig_h1)
        st.write("H1 se confirma: febrero supera el promedio anual y valida el uso de `pax_feb` como termómetro del impacto turístico.")

    with subtabs[1]:
        st.markdown("<div class='small-note'>H2. Interés en Google en enero vs flujo de pasajeros en febrero</div>", unsafe_allow_html=True)
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
        st.write(f"H2 se rechaza: la correlación `trends_ene` vs `pax_feb` es **{corr_h2:.2f}**, lo que indica una relación débil o nula.")

    with subtabs[2]:
        st.markdown("<div class='small-note'>H3. Recuperación post-pandemia y H4. crecimiento del gasto</div>", unsafe_allow_html=True)
        fig_h34, axes = plt.subplots(1, 2, figsize=(15, 7))

        colores_vis = ['#E74C3C' if a in [2021] else ('#F39C12' if a == 2020 else '#2ECC71') for a in df['año']]
        axes[0].bar(df['año'], df['visitantes_carnaval'], color=colores_vis, alpha=0.85)
        axes[0].set_title('Visitantes totales al Carnaval\n(H3 — crecimiento post-pandemia)')
        axes[0].set_ylabel('Visitantes')
        axes[0].set_xticks(df['año'])
        axes[0].set_xticklabels(df['año'], rotation=45)
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}K'))
        for año, vis in zip(df['año'], df['visitantes_carnaval']):
            if vis > 0:
                axes[0].text(año, vis + 5000, f'{vis/1000:.0f}K', ha='center', fontsize=8)
        axes[0].annotate('Cancelado', xy=(2021, 0), xytext=(2021, 30000),
                         ha='center', fontsize=8, color='#E74C3C',
                         arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=0.8))

        df_gasto = df[df['gasto_normalizado'] > 0]
        axes[1].plot(df_gasto['año'], df_gasto['gasto_normalizado'] / 1e6,
                     marker='o', color='#E67E22', linewidth=2.5, markersize=8)
        axes[1].fill_between(df_gasto['año'], df_gasto['gasto_normalizado'] / 1e6, alpha=0.15, color='#E67E22')
        axes[1].set_title('Gasto promedio del turista (millones COP)\n(H4 — crecimiento sostenido)')
        axes[1].set_ylabel('Millones COP')
        axes[1].set_xticks(df_gasto['año'])
        axes[1].set_xticklabels(df_gasto['año'], rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        for año, gasto in zip(df_gasto['año'], df_gasto['gasto_normalizado']):
            axes[1].text(año, gasto/1e6 + 0.03, f'${gasto/1e6:.1f}M', ha='center', fontsize=8)

        st.pyplot(fig_h34)
        st.write("H3 y H4 se confirman: existe recuperación post-pandemia y crecimiento sostenido del gasto turístico.")

    with subtabs[3]:
        st.markdown("<div class='small-note'>H6. Trends vs gasto y H7. pasajeros vs gasto</div>", unsafe_allow_html=True)
        fig_h67, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.scatterplot(x='trends_ene', y='gasto_normalizado', data=df, s=100, hue='año', palette='viridis', ax=axes[0])
        axes[0].set_title('Interés en Google en Enero vs. Gasto Normalizado (H6)')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        for i in range(len(df)):
            axes[0].text(df['trends_ene'].iloc[i] + 0.5, df['gasto_normalizado'].iloc[i], str(df['año'].iloc[i]), fontsize=8)

        sns.scatterplot(x='pax_feb', y='gasto_normalizado', data=df, s=100, hue='año', palette='viridis', ax=axes[1], legend=False)
        axes[1].set_title('Flujo de Pasajeros en Febrero vs. Gasto Normalizado (H7)')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        for i in range(len(df)):
            axes[1].text(df['pax_feb'].iloc[i] + 0.5, df['gasto_normalizado'].iloc[i], str(df['año'].iloc[i]), fontsize=8)

        st.pyplot(fig_h67)

        corr_h6 = df["trends_ene"].corr(df["gasto_normalizado"])
        corr_h7 = df["pax_feb"].corr(df["gasto_normalizado"])
        st.write(f"H6 se rechaza (**r = {corr_h6:.2f}**). H7 se confirma (**r = {corr_h7:.2f}**).")

    with subtabs[4]:
        st.markdown("<div class='small-note'>H8. Cambio estructural, H9. tendencia y H10. outliers</div>", unsafe_allow_html=True)

        fig_h8, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        fig_h8.suptitle('Análisis de Cambio Estructural Post-Pandemia (H8)', fontsize=16)

        sns.lineplot(x='año', y='pax_feb', data=df, marker='o', color='skyblue', ax=axes[0])
        sns.scatterplot(x='año', y='pax_feb', data=df, s=100, hue='año', palette='viridis', legend=False, ax=axes[0])
        axes[0].set_title('Flujo de Pasajeros en Febrero (pax_feb)')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].axvspan(2019.5, 2021.5, color='red', alpha=0.15)
        axes[0].text(2020.5, axes[0].get_ylim()[1] * 0.9, 'Pandemia', horizontalalignment='center', color='red', fontsize=10)

        sns.lineplot(x='año', y='visitantes_carnaval', data=df, marker='o', color='lightcoral', ax=axes[1])
        sns.scatterplot(x='año', y='visitantes_carnaval', data=df, s=100, hue='año', palette='viridis', legend=False, ax=axes[1])
        axes[1].set_title('Visitantes al Carnaval')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].axvspan(2019.5, 2021.5, color='red', alpha=0.15)
        axes[1].text(2020.5, axes[1].get_ylim()[1] * 0.9, 'Pandemia', horizontalalignment='center', color='red', fontsize=10)

        sns.lineplot(x='año', y='gasto_normalizado', data=df, marker='o', color='lightgreen', ax=axes[2])
        sns.scatterplot(x='año', y='gasto_normalizado', data=df, s=100, hue='año', palette='viridis', legend=False, ax=axes[2])
        axes[2].set_title('Gasto Normalizado del Turista')
        axes[2].set_xlabel('Año')
        axes[2].grid(True, linestyle='--', alpha=0.7)
        axes[2].axvspan(2019.5, 2021.5, color='red', alpha=0.15)
        axes[2].text(2020.5, axes[2].get_ylim()[1] * 0.9, 'Pandemia', horizontalalignment='center', color='red', fontsize=10)

        st.pyplot(fig_h8)

        fig_h9, ax_h9 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='año', y='pax_feb', data=df, marker='o', color='#28B463', ax=ax_h9)
        sns.scatterplot(x='año', y='pax_feb', data=df, s=100, hue='año', palette='viridis', legend=False, ax=ax_h9)
        ax_h9.set_title('Tendencia del Flujo de Pasajeros en Febrero a lo largo del Tiempo (H9)')
        ax_h9.grid(True, linestyle='--', alpha=0.7)
        ax_h9.set_xticks(df['año'])
        for i in range(len(df)):
            ax_h9.text(df['año'].iloc[i], df['pax_feb'].iloc[i] + 500, df['año'].iloc[i], ha='center', fontsize=9)
        st.pyplot(fig_h9)

        fig_h10, axes = plt.subplots(1, 3, figsize=(15, 6))
        columns_for_outliers = ['pax_feb', 'visitantes_carnaval', 'gasto_normalizado']
        for i, col in enumerate(columns_for_outliers):
            sns.boxplot(y=df[col], color='lightcoral', ax=axes[i])
            sns.stripplot(y=df[col], x=df['año'], jitter=True, color='black', size=5, ax=axes[i])
            axes[i].set_title(f'Boxplot de {col}')
            axes[i].set_ylabel(col)
            axes[i].set_xlabel('Año')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_h10)

        slope, intercept, r_value, p_value, std_err = linregress(df["año"], df["pax_feb"])
        st.write(f"H8, H9 y H10 se confirman. En H9, la pendiente de `pax_feb` es **{slope:.2f}** con **p = {p_value:.3f}**.")

        st.write(
            "En H10, los boxplots con puntos por año no buscan comparar una serie temporal, sino mostrar la distribución general de cada variable y resaltar valores extremos. "
            "El año 2021 aparece claramente como outlier en flujo, visitantes y gasto."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("T - Tell")
    st.markdown(
        """
        - El Carnaval presenta crecimiento estructural de largo plazo.  
        - La TRM emerge como variable estratégica del contexto turístico.  
        - La ocupación hotelera refleja con claridad la intensidad del evento.  
        - 2021 debe tratarse como outlier estructural, no como tendencia.  
        - Google Trends aporta una señal limitada y contraintuitiva.  
        - El EDA justifica avanzar a clustering y clasificación.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Número óptimo de clusters")
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
        "El análisis del codo y de silueta sugiere que **k=3** y **k=4** son soluciones plausibles. "
        "En el notebook se utiliza **k=4** para caracterizar etapas históricas y **k=3** para construir la variable objetivo del modelo."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("K-Means (k=4) con PCA")
    fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="cluster",
        size="año",
        sizes=(50, 350),
        palette="viridis",
        data=outputs["df_pca_clusters"],
        ax=ax_pca
    )
    for _, row in outputs["df_pca_clusters"].iterrows():
        ax_pca.text(row["PC1"] + 0.1, row["PC2"] + 0.1, str(row["año"]), fontsize=8)
    ax_pca.set_title("Clusters of Carnival Editions (PCA)")
    st.pyplot(fig_pca)

    st.dataframe(outputs["cluster_means_k4"], use_container_width=True)
    st.write(
        "El clustering con k=4 separa el periodo pre-pandemia temprano, una etapa de transición, el outlier pandémico y la recuperación posterior."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("K-Means (k=3) y nivel_impacto")
    fig_compare, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.scatterplot(
        x="año",
        y="nivel_impacto",
        hue="nivel_impacto",
        data=df_model,
        palette="viridis",
        s=100,
        ax=axes[0]
    )
    axes[0].set_title("K-Means Clustering (k=3)")
    axes[0].grid(True, linestyle="--", alpha=0.7)

    sns.scatterplot(
        x="año",
        y="cluster",
        hue="cluster",
        data=df_model,
        palette="viridis",
        s=100,
        ax=axes[1]
    )
    axes[1].set_title("K-Means Clustering (k=4)")
    axes[1].grid(True, linestyle="--", alpha=0.7)

    st.pyplot(fig_compare)

    st.dataframe(outputs["resultado_k3"], use_container_width=True, hide_index=True)
    st.write(
        "Con **k=3**, el notebook construye la variable `nivel_impacto` como BAJO, MEDIO y ALTO según el promedio de `pax_feb` por cluster."
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

    st.dataframe(outputs["hierarchical_cluster_means"], use_container_width=True)
    st.write(
        "El clustering jerárquico confirma la existencia de grupos diferenciados y refuerza la lectura histórica del Carnaval."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Importancia de variables")
    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=outputs["feature_importance_df"],
        palette="viridis",
        ax=ax_imp
    )
    ax_imp.set_title("Importancia de Variables para Predecir el Nivel de Impacto del Carnaval")
    st.pyplot(fig_imp)

    st.write(
        "Las variables que más pesan en la predicción están relacionadas con el comportamiento de llegada de personas a Barranquilla, en línea con lo señalado en el notebook."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Modelos de clasificación")
    st.dataframe(outputs["model_results"].round(2), use_container_width=True)

    st.write(
        "El notebook compara Regresión Logística, Árbol de Decisión y Random Forest. "
        "Aunque el mejor modelo se selecciona por F1-Score, estos resultados deben interpretarse con mucha cautela por el tamaño reducido del conjunto de prueba."
    )

    cm_tabs = st.tabs(["Logistic Regression", "Decision Tree", "Random Forest"])

    with cm_tabs[0]:
        plot_conf_matrix(outputs["y_test"], outputs["y_pred_log_reg"], outputs["all_labels"], outputs["reverse_map"], "Matriz de confusión - Logistic Regression")
    with cm_tabs[1]:
        plot_conf_matrix(outputs["y_test"], outputs["y_pred_decision_tree"], outputs["all_labels"], outputs["reverse_map"], "Matriz de confusión - Decision Tree")
    with cm_tabs[2]:
        plot_conf_matrix(outputs["y_test"], outputs["y_pred_random_forest"], outputs["all_labels"], outputs["reverse_map"], "Matriz de confusión - Random Forest")

    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Predicción 2027")
    st.metric("Nivel de impacto predicho", outputs["pred_clase"])

    fig_pred, ax_pred = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=outputs["proba_df"].index,
        y="Probabilidad",
        hue=outputs["proba_df"].index,
        data=outputs["proba_df"],
        palette="viridis",
        legend=False,
        ax=ax_pred
    )
    ax_pred.set_ylim(0, 1)
    ax_pred.set_title(f"Predicciones para 2027 ({outputs['best_model_name']})")
    for i, v in enumerate(outputs["proba_df"]["Probabilidad"].values):
        ax_pred.text(i, v + 0.02, f"{v:.2f}", ha="center")
    st.pyplot(fig_pred)

    st.write(
        "En el notebook, la predicción para 2027 se clasifica como **BAJO**, pero se aclara explícitamente que este resultado debe leerse solo como un ejercicio exploratorio."
    )
    st.write(
        "También se señala que, si se observa la tendencia reciente del Carnaval, la expectativa sustantiva sería una recepción alta. "
        "Por eso, esta salida no debe interpretarse como inferencia robusta."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab6:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Conclusiones")
    st.markdown(
        """
        - El Carnaval muestra crecimiento estructural entre 2013 y 2025.  
        - La pandemia introdujo un quiebre real en la serie.  
        - Existen grupos históricos distinguibles de ediciones del Carnaval.  
        - Las variables económicas y turísticas explican mejor el sistema que las señales digitales aisladas.  
        - El modelo predictivo es útil como prototipo analítico, pero debe interpretarse con cautela por el tamaño de muestra.
        """
    )
    st.write(
        "El principal valor del proyecto está en traducir un problema real de planeación en una herramienta basada en datos."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Limitaciones")
    st.markdown(
        """
        - n = 13 observaciones  
        - Años tempranos parcialmente estimados  
        - 2021 como outlier estructural  
        - Algunas variables sirven para EDA pero no para predicción anticipada  
        - Las métricas no deben interpretarse como evidencia definitiva de generalización
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
