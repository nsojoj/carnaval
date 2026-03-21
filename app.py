import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
c2.metric("Máx. pasajeros en febrero", f"{int(df['pax_feb'].max()):,}".replace(",", "."))
c3.metric("Variables del dataset", df.shape[1])

# ========= TABS =========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Presentación",
    "Datos y variables",
    "EDA",
    "Modelo",
    "Predicción"
])

with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("1.1 Contexto")
    st.write(
        "El Carnaval de Barranquilla es uno de los eventos culturales, turísticos y económicos más importantes de Colombia. "
        "Además de ser una de las celebraciones más emblemáticas del país, fue declarado Patrimonio Oral e Inmaterial "
        "de la Humanidad por la UNESCO en 2003 y es reconocido como el segundo carnaval más grande del mundo."
    )
    st.write(
        "Cada año, durante febrero, Barranquilla experimenta una transformación visible en su dinámica urbana y económica: "
        "aumenta la llegada de visitantes, se activan sectores como transporte, comercio, hotelería, gastronomía y entretenimiento, "
        "y la ciudad adquiere una visibilidad nacional e internacional excepcional. En ese sentido, el Carnaval no solo representa "
        "una expresión cultural, sino también un fenómeno turístico que puede analizarse cuantitativamente."
    )
    st.write(
        "Sin embargo, no todas las ediciones del Carnaval presentan el mismo comportamiento. El Carnaval de 2020, por ejemplo, "
        "se realizó pocos días antes de la llegada del COVID-19 a Colombia; el de 2021 se celebró sin público; y el de 2025 "
        "tuvo un contexto especialmente favorable por la alta atención mediática y la participación de Shakira como invitada especial."
    )
    st.write(
        "En 2025, el Carnaval generó más de $840.000 millones de pesos en impacto económico y atrajo cerca de 800.000 visitantes. "
        "Estas diferencias entre ediciones plantean una pregunta central para este estudio: si el impacto del Carnaval cambia cada año, "
        "¿es posible identificar patrones históricos y anticipar su nivel de impacto turístico antes de que llegue febrero?"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("1.2 Objetivo del estudio")
    st.write(
        "Desarrollar un pipeline de clustering + clasificación que permita analizar históricamente las ediciones del Carnaval "
        "de Barranquilla entre 2017 y 2025 y predecir si el Carnaval 2027 tendrá un nivel de impacto turístico ALTO, MEDIO o BAJO."
    )
    st.write(
        "El enfoque metodológico combina técnicas no supervisadas y supervisadas. En una primera etapa, se busca agrupar las ediciones "
        "históricas del Carnaval según su perfil de impacto turístico. En una segunda etapa, se entrena un modelo de clasificación "
        "capaz de anticipar la categoría de impacto de una futura edición a partir de información disponible antes del evento."
    )
    st.markdown(
        """
        **Objetivos específicos**
        
        1. Agrupar históricamente las ediciones del Carnaval (2017–2025) según su nivel de impacto turístico usando **K-Means** y **Clustering Jerárquico**.  
        2. Entrenar un modelo de **clasificación** capaz de predecir si el Carnaval 2027 tendrá un impacto turístico **ALTO, MEDIO o BAJO**, usando información disponible antes de febrero.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("1.3 Pregunta de investigación")
    st.markdown(
        """
        > **¿Es posible identificar patrones históricos en el flujo turístico del Carnaval de Barranquilla (2017–2025) y predecir si el Carnaval 2027 tendrá un nivel de impacto turístico ALTO, MEDIO o BAJO?**
        """
    )
    st.write(
        "Esta pregunta orienta todo el proyecto, ya que conecta el análisis exploratorio, la segmentación histórica de los años de Carnaval "
        "y la construcción posterior de un modelo predictivo útil para la toma de decisiones."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("1.4 Fuentes de datos")
    st.write(
        "La base final del proyecto fue construida mediante la integración de tres fuentes públicas que representan dimensiones distintas "
        "del fenómeno analizado: movilidad turística, interés digital y contexto económico. Esta combinación permite construir una visión "
        "más rica del Carnaval que la que se obtendría usando una sola fuente."
    )

    fuentes = pd.DataFrame({
        "Fuente": [
            "Aerocivil — datos.gov.co",
            "Google Trends",
            "Alcaldía de Barranquilla / Carnaval S.A.S."
        ],
        "Variable principal": [
            "Pasajeros llegados al aeropuerto BAQ por mes",
            "Interés de búsqueda 'Carnaval Barranquilla' en Colombia",
            "Visitantes totales y gasto promedio por edición"
        ],
        "Descripción": [
            "Fuente principal del estudio. Permite medir el flujo turístico aéreo mensual hacia Barranquilla.",
            "Permite capturar el interés digital previo al evento como posible señal temprana de demanda.",
            "Aporta variables de contexto económico y turístico para complementar la lectura del impacto."
        ]
    })

    st.dataframe(fuentes, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Enlaces de consulta**
        - [Aerocivil — datos.gov.co](https://www.datos.gov.co/Transporte/Llegada-pasajeros-mensual-por-aeropuerto-origen-na/vy2b-zbv7)
        - [Google Trends](https://trends.google.com/trends/explore?q=Carnaval+Barranquilla&geo=CO)
        - [Barranquilla en cifras](https://www.barranquilla.gov.co/gerencia-de-ciudad/barranquilla-en-cifras)
        """
    )

    st.write(
        "La fuente central del estudio es Aerocivil, ya que ofrece datos mensuales, consistentes y específicos para Barranquilla "
        "a través del aeropuerto Ernesto Cortissoz. Google Trends se incorporó para representar señales previas de interés en el evento, "
        "mientras que las cifras de visitantes y gasto complementan la interpretación desde la dimensión económica."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Vista general del dataset")
    st.dataframe(df, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.write(f"**Filas:** {df.shape[0]}")
    c2.write(f"**Columnas:** {df.shape[1]}")
    st.write(
        "El dataset consolidado contiene una fila por año de Carnaval entre 2017 y 2025, integrando variables originales y variables derivadas "
        "que permiten caracterizar mejor el comportamiento turístico de cada edición."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Variables del dataset y feature engineering")

    variables = pd.DataFrame({
        "Variable": [
            "pax_feb",
            "pax_enero",
            "pax_dic_anterior",
            "pax_promedio_año",
            "efecto_carnaval",
            "tendencia_anual",
            "trends_dic_ant",
            "trends_ene",
            "visitantes_carnaval",
            "crecimiento_pax_yoy",
            "momentum_trends",
            "ratio_visitantes_pax",
            "gasto_normalizado",
            "nivel_impacto"
        ],
        "Fuente": [
            "Aerocivil",
            "Aerocivil",
            "Aerocivil",
            "Aerocivil",
            "Feature Engineering",
            "Feature Engineering",
            "Google Trends",
            "Google Trends",
            "Carnaval S.A.S.",
            "Feature Engineering",
            "Feature Engineering",
            "Feature Engineering",
            "Feature Engineering",
            "Generada por K-Means"
        ],
        "Tipo": [
            "Numérica",
            "Numérica",
            "Numérica",
            "Numérica",
            "Numérica",
            "Numérica",
            "Numérica",
            "Numérica",
            "Numérica",
            "Numérica",
            "Numérica",
            "Numérica",
            "Numérica",
            "Categórica"
        ],
        "Descripción": [
            "Pasajeros llegados a Barranquilla en febrero; base para generar Y.",
            "Pasajeros en enero; contexto del mes previo al Carnaval.",
            "Pasajeros en diciembre del año anterior; tendencia de fin de año.",
            "Promedio mensual de pasajeros del año; nivel base del mercado.",
            "pax_feb / pax_promedio_año; cuánto aumenta el Carnaval sobre el comportamiento normal del año.",
            "(Q4 - Q1) / Q1 × 100; indica si el sector venía creciendo durante el año.",
            "Índice de búsqueda en diciembre anterior (0–100).",
            "Índice de búsqueda en enero (0–100).",
            "Total estimado de visitantes a la edición del Carnaval.",
            "pax_feb / pax_feb_año_anterior - 1; momentum interanual del Carnaval.",
            "trends_ene / trends_dic_ant; aceleración reciente del interés digital.",
            "visitantes / pax_feb; aproxima la proporción relativa de visitantes que llegan por tierra frente a aire.",
            "gasto_prom_cop / pax_promedio_año; gasto ajustado por tamaño del mercado.",
            "Variable objetivo Y: ALTO / MEDIO / BAJO."
        ]
    })

    st.dataframe(variables, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Nota metodológica")
    st.write(
        "Las variables marcadas como 'Feature Engineering' no existían de forma directa en las fuentes originales. "
        "Fueron construidas a partir de variables crudas para aportar más información al modelo y mejorar su capacidad explicativa."
    )
    st.write(
        "Las variables **efecto_carnaval** y **tendencia_anual** corresponden a un feature engineering básico realizado durante la consolidación del dataset. "
        "Las variables **crecimiento_pax_yoy**, **momentum_trends**, **ratio_visitantes_pax** y **gasto_normalizado** corresponden a un feature engineering más avanzado."
    )
    st.write(
        "Adicionalmente, dos variables originales fueron eliminadas por multicolinealidad: **trends_feb**, por su alta correlación "
        "con **efecto_carnaval** (r = 0.94), y **gasto_prom_cop**, por su alta correlación con **visitantes_carnaval** (r = 0.99)."
    )
    st.markdown(
        "<p class='small-note'><b>Interpretación:</b> esta etapa no solo consolida fuentes, sino que transforma variables para que el modelo capture mejor la intensidad, la tendencia y el contexto de cada edición del Carnaval.</p>",
        unsafe_allow_html=True,
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
