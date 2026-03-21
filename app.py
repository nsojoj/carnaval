import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    f1_score,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Carnaval de Barranquilla", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fff7e6 0%, #fffdf7 45%, #f8fbff 100%);
    }
    .hero {
        background: linear-gradient(135deg, #ffb703 0%, #fb8500 35%, #d62828 70%, #6a4c93 100%);
        padding: 1.5rem 1.8rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1rem;
    }
    .card {
        background: rgba(255,255,255,0.92);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_num(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)


@st.cache_data
def load_data():
    df = pd.read_csv("dataset_final_carnaval_fe-6.csv")

    if "trends_ene_co" in df.columns:
        df = df.rename(columns={"trends_ene_co": "trends_ene"})

    extra = {
        "año": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "pax_enero": [14000, 15000, 16000, 17000, 17894, 12446, 16433, 17923, 17312, 22000, 25000, 27000, 29000],
        "pax_dic_anterior": [13500, 14500, 15500, 16500, 17894, 13501, 16679, 19388, 14361, 21000, 24000, 26000, 28000],
        "pax_promedio_año": [10000, 10000, 10000, 15618, 13699, 12540, 16296, 6961, 15695, 24000, 20000, 22000, 23000],
        "tendencia_anual": [-5.0, 10.0, 15.0, 20.0, -30.83, 65.42, 22.65, -27.64, 43.58, 10.0, 12.0, 15.0, 18.0],
        "trends_dic_ant": [25, 20, 28, 32, 30, 7, 6, 7, 2, 10, 15, 18, 22],
        "momentum_trends": [1.0, 1.2, 1.5, 1.8, 1.0, 4.1429, 3.3333, 3.1429, 2.5, 3.0, 3.2, 3.5, 3.8],
    }

    dfi = df.set_index("año").copy()
    for col, vals in extra.items():
        if col != "año":
            dfi[col] = pd.Series(vals, index=extra["año"])

    dfi["gasto_normalizado"] = dfi["gasto_prom_cop"] / 10000
    df = dfi.reset_index()
    df["efecto_carnaval"] = df["pax_feb"] / df["pax_promedio_año"]
    return df


@st.cache_data
def run_models(df):
    work = df.copy()

    numerical_cols = work.select_dtypes(include=["number"]).columns.tolist()
    exclude = ["año", "cluster", "cluster_kmeans", "cluster_jerarquico", "nivel_impacto_encoded"]
    scale_cols = [c for c in numerical_cols if c not in exclude]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(work[scale_cols]), columns=scale_cols)

    inertia = []
    sil = []
    for k in range(1, len(X_scaled)):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
        if k > 1:
            sil.append({"k": k, "score": silhouette_score(X_scaled, km.labels_)})

    sil_df = pd.DataFrame(sil)

    km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    work["cluster"] = km4.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "año": work["año"], "cluster": work["cluster"]})

    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    work["cluster_kmeans"] = km3.fit_predict(X_scaled)
    medias = work.groupby("cluster_kmeans")["pax_feb"].mean().sort_values()
    mapa = {medias.index[0]: "BAJO", medias.index[1]: "MEDIO", medias.index[2]: "ALTO"}
    work["nivel_impacto"] = work["cluster_kmeans"].map(mapa)

    linked = linkage(X_scaled, method="ward")
    work["cluster_jerarquico"] = fcluster(linked, 4, criterion="maxclust")

    le = LabelEncoder()
    work["nivel_impacto_encoded"] = le.fit_transform(work["nivel_impacto"])

    X_cols = [c for c in work.select_dtypes(include=["int64", "float64"]).columns if c not in exclude]
    X = work[X_cols]
    y = work["nivel_impacto_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, solver="liblinear", max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    preds = {}
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        rows.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        })

    results = pd.DataFrame(rows).set_index("Modelo")
    best_name = results["F1-Score"].idxmax()
    best_model = models[best_name]

    rf_imp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_imp.fit(X, y)
    imp_df = pd.DataFrame({"Feature": X.columns, "Importance": rf_imp.feature_importances_}).sort_values(
        "Importance", ascending=False
    )

    data_2027 = pd.DataFrame({
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
        "gasto_normalizado": [90.0],
    }, columns=X_train.columns)

    pred_scaler = StandardScaler()
    pred_scaler.fit(X)
    X_2027 = pd.DataFrame(pred_scaler.transform(data_2027), columns=X.columns)

    reverse_map = {0: "ALTO", 1: "BAJO", 2: "MEDIO"}
    pred_class = reverse_map[best_model.predict(X_2027)[0]]
    pred_probs = best_model.predict_proba(X_2027)
    proba_df = pd.DataFrame(pred_probs, columns=[reverse_map[i] for i in best_model.classes_]).T
    proba_df.columns = ["Probabilidad"]
    proba_df = proba_df.sort_values("Probabilidad", ascending=False)

    return {
        "work": work,
        "scaled": X_scaled,
        "inertia": inertia,
        "sil_df": sil_df,
        "pca_df": pca_df,
        "linked": linked,
        "results": results,
        "preds": preds,
        "y_test": y_test,
        "best_name": best_name,
        "pred_class": pred_class,
        "proba_df": proba_df,
        "imp_df": imp_df,
        "reverse_map": reverse_map,
        "all_labels": sorted(y_train.unique()),
    }


def plot_cm(y_true, y_pred, labels, reverse_map, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tick_labels = [reverse_map[i] for i in labels]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    st.pyplot(fig)


df = load_data()

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.3rem;">Carnaval de Barranquilla: analítica del impacto turístico</h1>
        <p style="font-size:1.03rem; margin-bottom:0.35rem;">
            Proyecto de maestría orientado a describir, segmentar y predecir el impacto turístico del Carnaval.
        </p>
        <p style="margin-bottom:0.2rem;"><b>Programa:</b> Maestría en Analítica de Datos</p>
        <p style="margin-bottom:0;"><b>Integrantes:</b> Mario Orozco · Rosa Mora · Natalia Sojo · Donnys Torres</p>
    </div>
    """,
    unsafe_allow_html=True,
)

a, b, c = st.columns(3)
a.metric("Años analizados", int(df["año"].nunique()))
b.metric("Máx. pasajeros en febrero", fmt_num(df["pax_feb"].max()))
c.metric("Variables base", 10)

tabs = st.tabs(["Presentación", "Datos y variables", "EDA QUEST", "Modelo", "Predicción", "Conclusiones"])

with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Problema")
    st.write(
        "El Carnaval de Barranquilla es uno de los principales motores culturales y económicos de la ciudad. "
        "El reto de planeación consiste en estimar su magnitud antes del evento para apoyar decisiones públicas y privadas."
    )
    st.write(
        "Este proyecto busca responder si es posible anticipar un nivel de impacto **ALTO, MEDIO o BAJO** usando señales turísticas, macroeconómicas y digitales."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Base de datos")
    st.write(
        "La base integra fuentes oficiales, reales y estimadas. Incluye variables turísticas, económicas, hoteleras, macroeconómicas y digitales para 2013–2025."
    )
    vars_df = pd.DataFrame({
        "Variable": [
            "pax_feb", "crecimiento_pax_yoy", "visitantes_carnaval", "gasto_prom_cop",
            "ratio_visitantes_pax", "ocup_hotel_feb", "trm_feb_usdcop",
            "tur_int_colombia_miles", "trends_ene"
        ],
        "Descripción": [
            "Flujo aéreo de febrero",
            "Variación interanual del flujo",
            "Visitantes totales",
            "Gasto promedio del turista",
            "Visitantes por pasajero aéreo",
            "Ocupación hotelera",
            "TRM de febrero",
            "Turismo internacional en Colombia",
            "Interés de búsqueda en Google"
        ]
    })
    st.dataframe(vars_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("QUEST")
    st.write("El EDA se organizó bajo QUEST: Question, Understand, Explore, Study y Tell.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Question + Understand")
    st.markdown(
        """
        - ¿Existen grupos naturales de años?  
        - ¿Qué variables distinguen mejor los niveles de impacto?  
        - ¿Es posible predecir el nivel con variables previas?  
        - El dataset no presenta valores faltantes.  
        - 2021 es un outlier estructural.
        """
    )
    st.dataframe(df.describe().T, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Explore")
    selected = st.selectbox("Selecciona una variable", df.select_dtypes(include=np.number).columns.tolist())

    fig_dist, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df[selected], kde=True, ax=axes[0])
    axes[0].set_title(f"Histograma de {selected}")
    sns.boxplot(y=df[selected], ax=axes[1])
    axes[1].set_title(f"Boxplot de {selected}")
    st.pyplot(fig_dist)

    fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.4, ax=ax_corr)
    ax_corr.set_title("Matriz de correlación")
    st.pyplot(fig_corr)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Study")
    htabs = st.tabs(["H1", "H2", "H3-H4", "H6-H7", "H8-H10"])

    with htabs[0]:
        fig_h1, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(df))
        w = 0.35
        axes[0].bar(x - w/2, df["pax_feb"], width=w, label="Febrero", color="#E74C3C")
        axes[0].bar(x + w/2, df["pax_promedio_año"], width=w, label="Promedio año", color="#3498DB")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df["año"], rotation=45)
        axes[0].legend()
        axes[0].set_title("H1: pax_feb vs promedio anual")

        axes[1].bar(df["año"], df["efecto_carnaval"], color="#F39C12")
        axes[1].axhline(y=1, color="black", linestyle="--")
        axes[1].set_xticks(df["año"])
        axes[1].set_xticklabels(df["año"], rotation=45)
        axes[1].set_title("Efecto Carnaval")
        st.pyplot(fig_h1)

    with htabs[1]:
        fig_h2, ax_h2 = plt.subplots(figsize=(9, 5))
        sns.scatterplot(x="trends_ene", y="pax_feb", data=df, s=100, hue="año", palette="viridis", ax=ax_h2)
        for i in range(len(df)):
            ax_h2.text(df["trends_ene"].iloc[i] + 0.5, df["pax_feb"].iloc[i], str(df["año"].iloc[i]), fontsize=8)
        ax_h2.set_title("H2: trends_ene vs pax_feb")
        st.pyplot(fig_h2)

    with htabs[2]:
        fig_h34, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(df["año"], df["visitantes_carnaval"], color="#2ECC71")
        axes[0].set_title("H3: visitantes")
        axes[0].set_xticks(df["año"])
        axes[0].set_xticklabels(df["año"], rotation=45)

        gasto_df = df[df["gasto_normalizado"] > 0]
        axes[1].plot(gasto_df["año"], gasto_df["gasto_normalizado"], marker="o", color="#E67E22")
        axes[1].set_title("H4: gasto")
        axes[1].set_xticks(gasto_df["año"])
        axes[1].set_xticklabels(gasto_df["año"], rotation=45)
        st.pyplot(fig_h34)

    with htabs[3]:
        fig_h67, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.scatterplot(x="trends_ene", y="gasto_normalizado", data=df, ax=axes[0], color="#6a4c93")
        axes[0].set_title("H6: Trends vs gasto")
        sns.scatterplot(x="pax_feb", y="gasto_normalizado", data=df, ax=axes[1], color="#fb8500")
        axes[1].set_title("H7: pasajeros vs gasto")
        st.pyplot(fig_h67)

    with htabs[4]:
        fig_h8, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.lineplot(x="año", y="pax_feb", data=df, marker="o", color="#28B463", ax=axes[0])
        axes[0].axvspan(2019.5, 2021.5, color="red", alpha=0.12)
        axes[0].set_title("H8-H9: cambio estructural y tendencia")

        fig_h10, ax_h10 = plt.subplots(figsize=(12, 5))
        cols = ["pax_feb", "visitantes_carnaval", "gasto_normalizado"]
        tmp = df[cols].melt(var_name="Variable", value_name="Valor")
        sns.boxplot(x="Variable", y="Valor", data=tmp, ax=ax_h10, color="lightcoral")
        ax_h10.set_title("H10: outliers principales")
        st.pyplot(fig_h8)
        st.pyplot(fig_h10)

    st.markdown(
        """
        **Tell**
        - El Carnaval muestra crecimiento estructural.
        - 2021 es un outlier estructural.
        - La TRM y la ocupación hotelera son variables clave.
        - Google Trends tiene capacidad explicativa limitada.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Modelo")
    if st.button("Cargar resultados del modelo"):
        out = run_models(df)

        fig_elbow, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(range(1, len(out["scaled"])), out["inertia"], marker="o", linestyle="--")
        axes[0].set_title("Elbow")
        axes[1].plot(out["sil_df"]["k"], out["sil_df"]["score"], marker="o", linestyle="--")
        axes[1].set_title("Silhouette")
        st.pyplot(fig_elbow)

        fig_pca, ax_pca = plt.subplots(figsize=(9, 6))
        sns.scatterplot(x="PC1", y="PC2", hue="cluster", size="año", sizes=(50, 300), palette="viridis", data=out["pca_df"], ax=ax_pca)
        st.pyplot(fig_pca)

        st.dataframe(out["results"].round(2), use_container_width=True)

        fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
        sns.barplot(x="Importance", y="Feature", data=out["imp_df"], palette="viridis", ax=ax_imp)
        ax_imp.set_title("Importancia de variables")
        st.pyplot(fig_imp)

        cm_tabs = st.tabs(["Logistic Regression", "Decision Tree", "Random Forest"])
        with cm_tabs[0]:
            plot_cm(out["y_test"], out["preds"]["Logistic Regression"], out["all_labels"], out["reverse_map"], "Logistic Regression")
        with cm_tabs[1]:
            plot_cm(out["y_test"], out["preds"]["Decision Tree"], out["all_labels"], out["reverse_map"], "Decision Tree")
        with cm_tabs[2]:
            plot_cm(out["y_test"], out["preds"]["Random Forest"], out["all_labels"], out["reverse_map"], "Random Forest")
    else:
        st.info("Haz clic en el botón para cargar clustering y clasificación.")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[4]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Predicción")
    if st.button("Cargar predicción 2027"):
        out = run_models(df)
        st.metric("Nivel de impacto predicho", out["pred_class"])

        fig_pred, ax_pred = plt.subplots(figsize=(8, 5))
        ax_pred.bar(out["proba_df"].index, out["proba_df"]["Probabilidad"])
        ax_pred.set_ylim(0, 1)
        ax_pred.set_title(f"Probabilidades 2027 ({out['best_name']})")
        for i, v in enumerate(out["proba_df"]["Probabilidad"].values):
            ax_pred.text(i, v + 0.02, f"{v:.2f}", ha="center")
        st.pyplot(fig_pred)
    else:
        st.info("Haz clic en el botón para cargar la predicción.")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[5]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Conclusiones")
    st.markdown(
        """
        - El Carnaval presenta crecimiento estructural de largo plazo.  
        - La pandemia introdujo una ruptura real en la serie.  
        - Existen grupos históricos distinguibles.  
        - Las variables turísticas y económicas explican mejor el sistema que las señales digitales aisladas.  
        - El modelo es útil como prototipo analítico, pero debe leerse con cautela por el tamaño de muestra.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
