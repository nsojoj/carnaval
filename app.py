import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Carnaval de Barranquilla",
    page_icon="🎭",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("dataset_final_carnaval_fe.csv")


def plot_line(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df[x], df[y], marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    st.pyplot(fig)


st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Ir a:",
    [
        "Inicio",
        "Datos y variables",
        "EDA inicial",
    ]
)

st.title("Análisis del impacto turístico del Carnaval de Barranquilla")
st.caption("Clustering y clasificación del flujo turístico aéreo (2017–2025)")

df = load_data()

if page == "Inicio":
    col1, col2, col3 = st.columns(3)
    col1.metric("Años analizados", int(df["año"].nunique()))
    col2.metric("Máx. pasajeros en febrero", f"{int(df['pax_feb'].max()):,}".replace(",", "."))
    col3.metric("Variables del dataset", df.shape[1])

    st.markdown("""
    ## Objetivo
    Analizar el comportamiento histórico del flujo turístico asociado al Carnaval de Barranquilla
    y construir modelos de clustering y clasificación que permitan identificar niveles de impacto
    y apoyar la predicción de futuras ediciones.

    ## Contexto
    El dataset consolidado integra variables de demanda turística aérea, señales de interés digital
    y variables económicas construidas para estudiar el Carnaval de Barranquilla entre 2017 y 2025.

    ## Qué muestra esta app
    - Resumen del proyecto
    - Exploración inicial del dataset
    - Visualizaciones clave del EDA
    - Luego se podrán integrar los módulos de clustering y clasificación
    """)

    st.info("Primer MVP: esta versión cubre introducción, datos y EDA inicial. Luego añadimos clustering y clasificación.")

elif page == "Datos y variables":
    st.subheader("Vista general del dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Dimensiones")
    c1, c2 = st.columns(2)
    c1.write(f"**Filas:** {df.shape[0]}")
    c2.write(f"**Columnas:** {df.shape[1]}")

    st.subheader("Diccionario de variables")
    variables = pd.DataFrame(
        {
            "Variable": [
                "año", "pax_feb", "pax_enero", "pax_dic_anterior", "pax_promedio_año",
                "efecto_carnaval", "tendencia_anual", "trends_dic_ant", "trends_ene",
                "visitantes_carnaval", "crecimiento_pax_yoy", "momentum_trends",
                "ratio_visitantes_pax", "gasto_normalizado"
            ],
            "Descripción": [
                "Año de análisis",
                "Pasajeros de febrero",
                "Pasajeros de enero",
                "Pasajeros de diciembre del año anterior",
                "Promedio anual de pasajeros",
                "Relación entre febrero y el promedio anual",
                "Tendencia anual del flujo de pasajeros",
                "Interés en Google Trends en diciembre previo",
                "Interés en Google Trends en enero",
                "Visitantes estimados del Carnaval",
                "Crecimiento interanual de pasajeros",
                "Cambio reciente en interés digital",
                "Relación entre visitantes y pasajeros",
                "Indicador económico normalizado"
            ]
        }
    )
    st.dataframe(variables, use_container_width=True, hide_index=True)

elif page == "EDA inicial":
    st.subheader("Flujo turístico en febrero")
    plot_line(df, "año", "pax_feb", "Pasajeros en febrero por año", "Pasajeros")

    st.subheader("Interés digital previo al Carnaval")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["año"], df["trends_dic_ant"], marker="o", label="Trends diciembre")
    ax.plot(df["año"], df["trends_ene"], marker="o", label="Trends enero")
    ax.set_title("Google Trends previo al Carnaval")
    ax.set_xlabel("Año")
    ax.set_ylabel("Índice de interés")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Relación entre flujo turístico y visitantes")
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.scatter(df["pax_feb"], df["visitantes_carnaval"], s=80)
    for _, row in df.iterrows():
        ax2.annotate(str(int(row["año"])), (row["pax_feb"], row["visitantes_carnaval"]))
    ax2.set_title("Pasajeros de febrero vs visitantes del Carnaval")
    ax2.set_xlabel("Pasajeros febrero")
    ax2.set_ylabel("Visitantes del Carnaval")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    st.subheader("Hallazgos iniciales")
    st.markdown("""
    - El flujo de pasajeros de febrero cambia de forma importante entre años.
    - Las señales digitales previas al evento pueden servir como predictor complementario.
    - La relación entre pasajeros y visitantes ayuda a contextualizar el nivel de impacto del Carnaval.
    - Este módulo es la base visual antes de pasar a clustering y clasificación.
    """)
