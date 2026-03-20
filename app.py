import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Carnaval de Barranquilla",
    page_icon="🎭",
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
        background: rgba(255,255,255,0.85);
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

@st.cache_data
def load_data():
    return pd.read_csv("dataset_final_carnaval_fe.csv")


def plot_line(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, color: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df[x], df[y], marker="o", linewidth=2.5, color=color)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    st.pyplot(fig)


# ========= DATA =========
df = load_data()

# ========= HERO =========
st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.3rem;">🎭 Carnaval de Barranquilla: analítica del impacto turístico</h1>
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
c2.metric("Máx. pasajeros en febrero", f"{int(df['pax_feb'].max()):,}".replace(",", "."))
c3.metric("Variables del dataset", df.shape[1])

st.markdown(
    """
    <div class="highlight">
        Esta app presenta un primer producto funcional del proyecto final. Se prioriza la comprensión del problema,
        la trazabilidad de los datos y el análisis exploratorio como base para los módulos posteriores de modelado.
    </div>
    """,
    unsafe_allow_html=True,
)

# ========= TABS =========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Presentación",
    "🗂️ Datos y variables",
    "📊 EDA",
    "🧠 Modelo",
    "🔮 Predicción"
])

with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Contexto del estudio")
    st.write(
        "El Carnaval de Barranquilla es uno de los eventos culturales y turísticos más relevantes de Colombia, "
        "con efectos visibles sobre la movilidad de visitantes, la ocupación de servicios y la dinámica económica local. "
        "En la mayoría de los informes públicos, su impacto suele resumirse en cifras agregadas de visitantes o derrama económica. "
        "Sin embargo, esos reportes no siempre permiten identificar patrones históricos, comparar ediciones del evento ni anticipar el comportamiento "
        "de futuras celebraciones a partir de señales observables en los datos."
    )
    st.write(
        "Por ello, este proyecto propone una aproximación de analítica de datos centrada en el flujo turístico aéreo hacia Barranquilla. "
        "La lógica del estudio parte de que la llegada de pasajeros por vía aérea constituye un indicador consistente, disponible y específico para la ciudad, "
        "lo que lo convierte en una base metodológicamente sólida para estudiar el impacto del Carnaval en el período 2017–2025."
    )
    st.write(
        "Además del flujo de pasajeros, se incorporan variables de interés digital y variables económicas construidas a partir de fuentes públicas, "
        "con el fin de enriquecer la interpretación del fenómeno y construir una visión más integral del evento."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Objetivo general")
    st.write(
        "Analizar y caracterizar el impacto turístico del Carnaval de Barranquilla entre 2017 y 2025, empleando herramientas de análisis exploratorio, "
        "clustering y clasificación sobre variables de movilidad aérea, interés digital y contexto económico."
    )

    st.subheader("Objetivos específicos")
    st.markdown(
        """
        1. Explorar el comportamiento histórico del flujo de pasajeros asociado al Carnaval.
        2. Identificar grupos o tipos de ediciones del Carnaval según su nivel de impacto turístico.
        3. Construir una base analítica que sirva de insumo para un modelo de clasificación del nivel de impacto.
        4. Relacionar el comportamiento del flujo turístico con señales de interés digital y variables económicas complementarias.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Recolección y construcción de la base de datos")
    st.write(
        "La base consolidada se construyó integrando diferentes fuentes públicas. La fuente principal corresponde a los registros de Aerocivil, "
        "de donde se obtuvo el número de pasajeros mensuales del aeropuerto Ernesto Cortissoz (BAQ). Esta fuente fue priorizada porque ofrece continuidad temporal, "
        "especifidad geográfica para Barranquilla y una frecuencia mensual útil para comparar el comportamiento del mes de Carnaval frente al resto del año."
    )
    st.write(
        "Como complemento, se incorporaron datos de Google Trends asociados al término de búsqueda relacionado con el Carnaval de Barranquilla. "
        "Estas variables permiten aproximarse al interés digital previo al evento y funcionan como señales tempranas de atención o intención de viaje."
    )
    st.write(
        "Finalmente, se añadieron variables económicas estimadas, como visitantes del Carnaval y algunos indicadores derivados, con el propósito de contextualizar "
        "el impacto del evento más allá de la sola movilidad aérea. A partir de estas fuentes se realizó un proceso de limpieza, consolidación y generación de variables "
        "ingenierizadas para construir el dataset final del proyecto."
    )
    st.markdown(
        "<p class='small-note'><b>Decisión metodológica clave:</b> se priorizó el flujo de pasajeros como eje del estudio debido a que otras fuentes turísticas no mantenían la misma disponibilidad y granularidad para Barranquilla en todo el período de análisis.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Vista general del dataset")
    st.dataframe(df, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.write(f"**Filas:** {df.shape[0]}")
    c2.write(f"**Columnas:** {df.shape[1]}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Explicación de variables")
    variables = pd.DataFrame(
        {
            "Variable": [
                "año", "pax_feb", "pax_enero", "pax_dic_anterior", "pax_promedio_año",
                "efecto_carnaval", "tendencia_anual", "trends_dic_ant", "trends_ene",
                "visitantes_carnaval", "crecimiento_pax_yoy", "momentum_trends",
                "ratio_visitantes_pax", "gasto_normalizado"
            ],
            "Tipo": [
                "Temporal", "Numérica", "Numérica", "Numérica", "Numérica",
                "Derivada", "Derivada", "Digital", "Digital",
                "Económica", "Derivada", "Derivada",
                "Derivada", "Económica"
            ],
            "Descripción": [
                "Año de análisis de la edición del Carnaval.",
                "Número de pasajeros registrados en febrero, mes asociado al Carnaval.",
                "Número de pasajeros en enero, usado como referencia previa al evento.",
                "Pasajeros de diciembre del año anterior, útil como señal de cierre de ciclo turístico.",
                "Promedio mensual de pasajeros del año, usado como línea base comparativa.",
                "Razón entre pasajeros de febrero y el promedio anual; aproxima la intensidad relativa del Carnaval.",
                "Indicador de comportamiento o dirección general del flujo de pasajeros durante el año.",
                "Índice de búsquedas en Google Trends en diciembre previo al Carnaval.",
                "Índice de búsquedas en Google Trends en enero, inmediatamente antes del evento.",
                "Estimación de visitantes al Carnaval en cada año.",
                "Crecimiento interanual del flujo de pasajeros respecto al año anterior.",
                "Cambio reciente del interés digital entre periodos cercanos.",
                "Relación entre visitantes del Carnaval y pasajeros de febrero.",
                "Variable económica transformada para facilitar comparación entre años."
            ]
        }
    )
    st.dataframe(variables, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Interpretación analítica")
    st.write(
        "Las variables del dataset combinan tres dimensiones: movilidad turística, interés digital y contexto económico. "
        "Esto permite que el análisis no se limite a observar un solo indicador, sino que incorpore señales complementarias para entender mejor las diferencias entre ediciones del Carnaval."
    )
    st.write(
        "Las variables derivadas, como efecto_carnaval o crecimiento_pax_yoy, fueron construidas para mejorar la capacidad explicativa de la base y preparar el terreno para los modelos posteriores."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Flujo turístico en febrero")
    plot_line(df, "año", "pax_feb", "Pasajeros en febrero por año", "Pasajeros", "#d62828")
    st.write(
        "Esta gráfica resume el comportamiento del principal indicador del estudio. Permite observar la variación del flujo turístico aéreo en el mes más importante del análisis y comparar qué años presentan mayor o menor intensidad."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Interés digital previo al Carnaval")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["año"], df["trends_dic_ant"], marker="o", linewidth=2.2, label="Trends diciembre", color="#6a4c93")
        ax.plot(df["año"], df["trends_ene"], marker="o", linewidth=2.2, label="Trends enero", color="#219ebc")
        ax.set_title("Google Trends previo al Carnaval")
        ax.set_xlabel("Año")
        ax.set_ylabel("Índice de interés")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Relación entre pasajeros y visitantes")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.scatter(df["pax_feb"], df["visitantes_carnaval"], s=95, color="#fb8500")
        for _, row in df.iterrows():
            ax2.annotate(str(int(row["año"])), (row["pax_feb"], row["visitantes_carnaval"]))
        ax2.set_title("Pasajeros de febrero vs visitantes del Carnaval")
        ax2.set_xlabel("Pasajeros febrero")
        ax2.set_ylabel("Visitantes del Carnaval")
        ax2.grid(alpha=0.25)
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Lectura preliminar del EDA")
    st.markdown(
        """
        - Existen diferencias visibles entre años, lo que sugiere que no todas las ediciones del Carnaval tienen el mismo nivel de impacto turístico.
        - Las búsquedas en Google Trends antes del evento pueden aportar valor como señal temprana de interés.
        - La relación entre pasajeros y visitantes permite interpretar el flujo aéreo dentro de un contexto turístico más amplio.
        - Estas primeras visualizaciones justifican continuar hacia una etapa de segmentación con clustering y luego hacia clasificación supervisada.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Modelo")
    st.write("")
    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Predicción")
    st.write("")
    st.markdown("</div>", unsafe_allow_html=True)
