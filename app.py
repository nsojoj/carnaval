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

    # Variables auxiliares usadas en el trabajo
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

    return df_indexed.reset_index()


@st.cache_data
def prepare_model_outputs(df):
    df_model = df.copy()

    # ---------- CLUSTERING ----------
    numerical_cols = df_model.select_dtypes(include=["number"]).columns
    cols_to_exclude = [
        "año", "cluster", "cluster_kmeans", "nivel_impacto_encoded",
        "cluster_jerarquico"
    ]
    features_for_scaling = [col for col in numerical_cols if col not in cols_to_exclude]
    df_scaled_input = df_model[features_for_scaling]

    scaler_cluster = StandardScaler()
    df_scaled_array = scaler_cluster.fit_transform(df_scaled_input)
    df_scaled = pd.DataFrame(df_scaled_array, columns=df_scaled_input.columns)

    inertia_values = []
    silhouette_scores = []
    max_clusters = len(df_scaled) - 1

    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertia_values.append(kmeans.inertia_)
        if k > 1:
            silhouette_scores.append({"k": k, "score": silhouette_score(df_scaled, kmeans.labels_)})

    silhouette_df = pd.DataFrame(silhouette_scores)

    # KMeans k=4
    kmeans_k4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_model["cluster"] = kmeans_k4.fit_predict(df_scaled)

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)
    df_pca_clusters = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
    df_pca_clusters["año"] = df_model["año"].values
    df_pca_clusters["cluster"] = df_model["cluster"].values

    # KMeans k=3 -> variable objetivo
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
    linked_data = linkage(df_scaled, method="ward")
    df_model["cluster_jerarquico"] = fcluster(linked_data, 4, criterion="maxclust")

    cluster_means_k4 = df_model.groupby("cluster").mean(numeric_only=True)
    cluster_means_hac = df_model.groupby("cluster_jerarquico").mean(numeric_only=True)

    # ---------- CLASIFICACIÓN ----------
    encoder = LabelEncoder()
    df_model["nivel_impacto_encoded"] = encoder.fit_transform(df_model["nivel_impacto"])

    exclude_cols = ["año", "cluster", "cluster_kmeans", "cluster_jerarquico", "nivel_impacto_encoded"]
    X_cols = [col for col in df_model.select_dtypes(include=["int64", "float64"]).columns if col not in exclude_cols]
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

    # Predicción 2027
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
        "reverse_map": reverse_map
    }


df = load_data()
outputs = prepare_model_outputs(df)
df_model = outputs["df_model"]

# ---------- HEADER ----------
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
        El objetivo es anticipar si una edición del Carnaval puede clasificarse como de impacto
        ALTO, MEDIO o BAJO antes de que el evento ocurra.
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
    st.subheader("Problema y propósito")
    st.write(
        "El Carnaval de Barranquilla es uno de los principales motores culturales y económicos de la ciudad. "
        "En 2025 generó más de **$880.000 millones**, alrededor de **193.000 empleos** y cerca de **800.000 visitantes**."
    )
    st.write(
        "El reto para actores como Alcaldía, hoteleros, organizadores, comerciantes y aerolíneas es tomar decisiones costosas meses antes del evento, cuando aún no se conoce su magnitud final."
    )
    st.write(
        "Este proyecto propone apoyar esa planeación con datos, usando un pipeline de **clustering + clasificación** que permita entender patrones históricos y anticipar niveles de impacto."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Objetivo y pregunta")
    st.write(
        "El objetivo es identificar patrones históricos del Carnaval y predecir si una futura edición tendrá un impacto **ALTO, MEDIO o BAJO**."
    )
    st.markdown(
        """
        > **¿Podemos anticipar el nivel de impacto turístico del Carnaval antes de que comience, usando señales turísticas, macroeconómicas y digitales observables previamente?**
        """
    )
    st.write(
        "El periodo analizado comienza en 2013, año a partir del cual el aeropuerto BAQ entra en una nueva escala operativa y la comparación histórica se vuelve más consistente."
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
        "La base integra múltiples fuentes y combina datos reales, oficiales y estimados. "
        "Los años 2013–2016 requirieron reconstrucción parcial para visitantes, gasto y ocupación hotelera."
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
        "Rol": [
            "Flujo turístico aéreo",
            "Feature engineering",
            "Target turístico total",
            "EDA",
            "Feature engineering / leakage",
            "Capacidad de ciudad",
            "Contexto macroeconómico",
            "Contexto internacional",
            "Señal digital"
        ]
    })
    st.dataframe(variables, use_container_width=True, hide_index=True)
    st.write(
        "Además, se construyeron variables auxiliares para análisis y modelado, entre ellas `efecto_carnaval`, `pax_promedio_año`, `momentum_trends` y `gasto_normalizado`."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Q - Question")
    st.write(
        "La misión analítica fue determinar si existen patrones históricos de desempeño turístico y si es posible anticipar el nivel de impacto del Carnaval con variables previas al evento."
    )
    st.markdown(
        """
        **Preguntas guía**
        - ¿Existen grupos naturales de años?
        - ¿Qué variables distinguen mejor los niveles de impacto?
        - ¿Se puede predecir el nivel con información previa?
        - ¿Qué tan confiable es la predicción con n=13?
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
        "La auditoría confirma una base completa y coherente, pero estadísticamente heterogénea: 2021 aparece como outlier estructural y ciertas variables reflejan choques operativos o macroeconómicos."
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
        "El análisis univariado y de correlación muestra alta variabilidad en el sistema turístico, una relación fuerte entre visitantes, ocupación y gasto, y menor poder explicativo de señales como Google Trends."
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

    subtabs = st.tabs(["H1-H2", "H3-H4", "H6-H7", "H8-H10"])

    with subtabs[0]:
        fig_h1, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(df))
        w = 0.35
        axes[0].bar(x - w/2, df["pax_feb"], width=w, label="Febrero", color="#E74C3C")
        axes[0].bar(x + w/2, df["pax_promedio_año"], width=w, label="Promedio año", color="#3498DB")
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
        st.write(f"H1 confirmada. H2 rechazada: correlación `trends_ene` vs `pax_feb` = **{corr_h2:.2f}**.")

    with subtabs[1]:
        fig_h34, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(df["año"], df["visitantes_carnaval"], color="#2ECC71")
        axes[0].set_title("H3: visitantes")
        axes[0].set_xticks(df["año"])
        axes[0].set_xticklabels(df["año"], rotation=45)

        df_gasto = df[df["gasto_normalizado"] > 0]
        axes[1].plot(df_gasto["año"], df_gasto["gasto_normalizado"], marker="o", color="#E67E22")
        axes[1].set_title("H4: gasto")
        axes[1].set_xticks(df_gasto["año"])
        axes[1].set_xticklabels(df_gasto["año"], rotation=45)
        st.pyplot(fig_h34)

        st.write("H3 y H4 confirmadas: existe recuperación post-pandemia y crecimiento económico del Carnaval.")

    with subtabs[2]:
        fig_h67, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.scatterplot(x="trends_ene", y="gasto_normalizado", data=df, ax=axes[0], color="#6a4c93")
        axes[0].set_title("H6: Trends vs gasto")

        sns.scatterplot(x="pax_feb", y="gasto_normalizado", data=df, ax=axes[1], color="#fb8500")
        axes[1].set_title("H7: pasajeros vs gasto")
        st.pyplot(fig_h67)

        corr_h6 = df["trends_ene"].corr(df["gasto_normalizado"])
        corr_h7 = df["pax_feb"].corr(df["gasto_normalizado"])
        st.write(f"H6 rechazada (**r = {corr_h6:.2f}**). H7 confirmada (**r = {corr_h7:.2f}**).")

    with subtabs[3]:
        fig_h8, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.lineplot(x="año", y="pax_feb", data=df, marker="o", color="#28B463", ax=axes[0])
        axes[0].axvspan(2019.5, 2021.5, color="red", alpha=0.12)
        axes[0].set_title("H8-H9: cambio estructural y tendencia")

        sns.boxplot(y=df["visitantes_carnaval"], ax=axes[1], color="lightcoral")
        axes[1].set_title("H10: outliers")
        st.pyplot(fig_h8)

        slope, intercept, r_value, p_value, std_err = linregress(df["año"], df["pax_feb"])
        st.write(f"H8, H9 y H10 confirmadas. Pendiente de `pax_feb`: **{slope:.2f}**, p = **{p_value:.3f}**.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("T - Tell")
    st.markdown(
        """
        - El Carnaval presenta crecimiento estructural de largo plazo.  
        - La TRM y la ocupación hotelera son variables estratégicas.  
        - 2021 debe tratarse como outlier estructural.  
        - Google Trends aporta una señal limitada.  
        - El EDA justifica avanzar a clustering y clasificación.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Clustering")
    fig_elbow, axes = plt.subplots(1, 2, figsize=(12, 5))
    max_clusters = len(outputs["df_scaled"]) - 1
    axes[0].plot(range(1, max_clusters + 1), outputs["inertia_values"], marker="o", linestyle="--")
    axes[0].set_title("Elbow")
    axes[0].grid(True)

    sil_df = outputs["silhouette_df"]
    axes[1].plot(sil_df["k"], sil_df["score"], marker="o", linestyle="--")
    axes[1].set_title("Silhouette")
    axes[1].grid(True)
    st.pyplot(fig_elbow)

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
    ax_pca.set_title("K-Means (k=4) con PCA")
    st.pyplot(fig_pca)

    st.dataframe(outputs["cluster_means_k4"], use_container_width=True)
    st.write(
        "Los clusters permiten diferenciar un bloque pre-pandemia, una transición, un outlier pandémico y una recuperación post-pandemia."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Clasificación")
    st.dataframe(outputs["model_results"].round(2), use_container_width=True)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=outputs["feature_importance_df"],
        palette="viridis",
        ax=ax_imp
    )
    ax_imp.set_title("Importancia de variables")
    st.pyplot(fig_imp)

    st.write(
        "Los resultados muestran mejor desempeño de Árbol de Decisión y Random Forest en esta corrida. "
        "Aun así, por el tamaño de muestra, las métricas deben leerse como evidencia exploratoria."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Predicción 2027")
    st.metric("Nivel de impacto predicho", outputs["pred_clase"])

    fig_pred, ax_pred = plt.subplots(figsize=(8, 5))
    colors = ["#6a4c93", "#fb8500", "#d62828"][:len(outputs["proba_df"])]
    ax_pred.bar(outputs["proba_df"].index, outputs["proba_df"]["Probabilidad"], color=colors)
    ax_pred.set_ylim(0, 1)
    ax_pred.set_title(f"Probabilidades 2027 ({outputs['best_model_name']})")
    for i, v in enumerate(outputs["proba_df"]["Probabilidad"].values):
        ax_pred.text(i, v + 0.02, f"{v:.2f}", ha="center")
    st.pyplot(fig_pred)

    st.write(
        "La predicción para 2027 debe interpretarse como un ejercicio de escenarios, no como una proyección definitiva."
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
        "El principal valor del proyecto está en traducir un problema real de planeación en una herramienta basada en datos, integrando múltiples fuentes y una lógica de análisis explicable."
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
