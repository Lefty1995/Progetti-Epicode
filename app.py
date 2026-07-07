
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Synthetic Data Sandbox - VAE Edition",
    page_icon="🧬",
    layout="wide"
)

# -------------------------------------------------
# Funzioni di caricamento
# -------------------------------------------------

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

real_data = load_csv("real_clean_data.csv")
synthetic_data = load_csv("synthetic_data.csv")
statistics_comparison = load_csv("statistics_comparison.csv")
privacy_report = load_csv("privacy_check_report.csv")

# -------------------------------------------------
# Titolo principale
# -------------------------------------------------

st.title("Synthetic Data Sandbox - VAE Edition")
st.write(
    "Dashboard dimostrativa per generare, confrontare e analizzare dati sintetici "
    "creati con un modello VAE."
)

st.info(
    "Il progetto usa un dataset medico pubblico come esempio di dati sensibili. "
    "Il dataset sintetico viene generato per mantenere proprietà simili ai dati originali, "
    "senza copiare direttamente le righe reali."
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("Menu")
section = st.sidebar.radio(
    "Scegli una sezione",
    [
        "Overview",
        "Dataset",
        "Validazione",
        "Privacy Check",
        "Natural Language Analyst",
        "Upload CSV Demo"
    ]
)

# -------------------------------------------------
# Overview
# -------------------------------------------------

if section == "Overview":
    st.header("Overview del progetto")

    st.subheader("Numeri principali")

    if real_data is not None and synthetic_data is not None:
        summary_df = pd.DataFrame({
            "Voce": [
                "Righe dataset reale",
                "Colonne dataset reale",
                "Righe dataset sintetico"
            ],
            "Valore": [
                real_data.shape[0],
                real_data.shape[1],
                synthetic_data.shape[0]
            ]
        })

        st.dataframe(summary_df, use_container_width=True)
    else:
        st.warning("Dataset non caricati correttamente.")

    st.subheader("Obiettivo")
    st.write(
        "L'obiettivo del progetto è creare una piattaforma che parta da un dataset tabellare, "
        "addestri un Variational Autoencoder e generi un nuovo dataset sintetico. "
        "Successivamente i dati sintetici vengono confrontati con quelli reali tramite statistiche, "
        "grafici e controlli di privacy."
    )

    st.subheader("Componenti principali")
    st.markdown(
        """
        - **Data Engine:** preparazione e normalizzazione del dataset.
        - **Generative Core:** modello VAE per generare dati sintetici.
        - **Validation:** confronto tra dati reali e sintetici.
        - **Privacy Check:** controllo di righe identiche o troppo simili.
        - **User Interface:** dashboard Streamlit per visualizzare i risultati.
        - **Natural Language Analyst:** piccola funzione per interrogare i dati sintetici.
        """
    )

# -------------------------------------------------
# Dataset
# -------------------------------------------------

elif section == "Dataset":
    st.header("Dataset reale e sintetico")

    if real_data is None or synthetic_data is None:
        st.error("File del dataset non trovati. Assicurati di aver eseguito gli step precedenti.")
    else:
        tab1, tab2, tab3 = st.tabs(
            ["Dataset reale pulito", "Dataset sintetico", "Statistiche"]
        )

        with tab1:
            st.subheader("Dataset reale pulito")
            st.dataframe(real_data.head(50), use_container_width=True)
            st.write("Dimensione:", real_data.shape)

        with tab2:
            st.subheader("Dataset sintetico")
            st.dataframe(synthetic_data.head(50), use_container_width=True)
            st.write("Dimensione:", synthetic_data.shape)

            csv = synthetic_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Scarica dataset sintetico CSV",
                data=csv,
                file_name="synthetic_data.csv",
                mime="text/csv"
            )

        with tab3:
            st.subheader("Statistiche dataset reale")
            st.dataframe(real_data.describe(), use_container_width=True)

            st.subheader("Statistiche dataset sintetico")
            st.dataframe(synthetic_data.describe(), use_container_width=True)

# -------------------------------------------------
# Validazione
# -------------------------------------------------

elif section == "Validazione":
    st.header("Validazione dati reali vs sintetici")

    if statistics_comparison is not None:
        st.subheader("Confronto statistiche principali")
        st.dataframe(statistics_comparison, use_container_width=True)
    else:
        st.warning("File statistics_comparison.csv non trovato.")

    st.subheader("Grafici di validazione")

    image_paths = [
        "images/outcome_distribution_comparison.png",
        "images/distribution_comparison_Glucose.png",
        "images/distribution_comparison_BMI.png",
        "images/distribution_comparison_Age.png",
        "images/correlation_real.png",
        "images/correlation_synthetic.png",
        "images/correlation_difference.png"
    ]

    for path in image_paths:
        if os.path.exists(path):
            st.image(path, caption=path)
        else:
            st.warning(f"Immagine non trovata: {path}")

# -------------------------------------------------
# Privacy Check
# -------------------------------------------------

elif section == "Privacy Check":
    st.header("Privacy Check")

    st.write(
        "Questa sezione controlla se il dataset sintetico contiene righe identiche "
        "o troppo vicine alle righe reali."
    )

    if privacy_report is not None:
        st.subheader("Report privacy")
        st.dataframe(privacy_report, use_container_width=True)
    else:
        st.warning("File privacy_check_report.csv non trovato.")

    privacy_image = "images/privacy_distance_distribution.png"

    if os.path.exists(privacy_image):
        st.subheader("Distribuzione delle distanze")
        st.image(privacy_image)
    else:
        st.warning("Grafico privacy non trovato.")

    st.info(
        "Nel progetto finale questo controllo viene usato come verifica base. "
        "Non sostituisce una valutazione completa di privacy, ma aiuta a capire "
        "se il modello sta copiando direttamente i dati reali."
    )

# -------------------------------------------------
# Natural Language Analyst
# -------------------------------------------------

elif section == "Natural Language Analyst":
    st.header("Natural Language Analyst")

    if synthetic_data is None:
        st.error("Dataset sintetico non trovato.")
    else:
        st.write(
            "Fai una domanda semplice sul dataset sintetico. "
            "Esempi: correlazione, media, minimo, massimo o distribuzione."
        )

        st.markdown(
            """
            **Esempi di domande:**

            - Qual è la correlazione tra Glucose e BMI?
            - Qual è la media di Age?
            - Mostrami la distribuzione di Outcome.
            - Qual è il massimo di Insulin?
            - Fammi un riassunto del dataset.
            """
        )

        question = st.text_input(
            "Scrivi la tua domanda",
            value="Qual è la correlazione tra Glucose e BMI?"
        )

        use_llm = st.checkbox(
            "Usa LLM per migliorare la risposta se la chiave API è disponibile",
            value=False
        )

        def find_columns_in_question(question, columns):
            question_lower = question.lower()
            found_columns = []

            for col in columns:
                if col.lower() in question_lower:
                    found_columns.append(col)

            return found_columns

        def local_analysis(question, data):
            question_lower = question.lower()
            columns = data.columns.tolist()
            found_columns = find_columns_in_question(question, columns)

            result_text = ""
            fig = None

            if "correlazione" in question_lower or "correlation" in question_lower:
                if len(found_columns) >= 2:
                    col1 = found_columns[0]
                    col2 = found_columns[1]
                    corr_value = data[col1].corr(data[col2])

                    result_text = (
                        f"La correlazione tra {col1} e {col2} nel dataset sintetico "
                        f"è pari a {corr_value:.4f}."
                    )

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(data[col1], data[col2], alpha=0.6)
                    ax.set_title(f"Correlazione tra {col1} e {col2}")
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    ax.grid(alpha=0.3)

                else:
                    result_text = (
                        "Hai chiesto una correlazione, ma non ho trovato due colonne valide. "
                        "Esempio: Qual è la correlazione tra Glucose e BMI?"
                    )

            elif "media" in question_lower or "mean" in question_lower or "valore medio" in question_lower:
                if len(found_columns) >= 1:
                    col = found_columns[0]
                    mean_value = data[col].mean()
                    result_text = f"La media della colonna {col} è pari a {mean_value:.4f}."
                else:
                    result_text = "Non ho trovato una colonna valida per calcolare la media."

            elif "minimo" in question_lower or "minimum" in question_lower:
                if len(found_columns) >= 1:
                    col = found_columns[0]
                    min_value = data[col].min()
                    result_text = f"Il valore minimo della colonna {col} è pari a {min_value:.4f}."
                else:
                    result_text = "Non ho trovato una colonna valida per calcolare il minimo."

            elif "massimo" in question_lower or "maximum" in question_lower:
                if len(found_columns) >= 1:
                    col = found_columns[0]
                    max_value = data[col].max()
                    result_text = f"Il valore massimo della colonna {col} è pari a {max_value:.4f}."
                else:
                    result_text = "Non ho trovato una colonna valida per calcolare il massimo."

            elif "distribuzione" in question_lower or "distribution" in question_lower:
                if len(found_columns) >= 1:
                    col = found_columns[0]
                    result_text = f"Distribuzione della colonna {col}."

                    fig, ax = plt.subplots(figsize=(6, 4))

                    if data[col].nunique() <= 10:
                        data[col].value_counts().sort_index().plot(kind="bar", ax=ax)
                        ax.set_ylabel("Numero di record")
                    else:
                        ax.hist(data[col], bins=30)
                        ax.set_ylabel("Frequenza")

                    ax.set_title(f"Distribuzione di {col}")
                    ax.set_xlabel(col)
                    ax.grid(alpha=0.3)

                else:
                    result_text = "Non ho trovato una colonna valida per mostrare la distribuzione."

            elif "riassunto" in question_lower or "summary" in question_lower or "descrivi" in question_lower:
                result_text = (
                    f"Il dataset sintetico contiene {data.shape[0]} righe e {data.shape[1]} colonne. "
                    f"Le colonne sono: {', '.join(columns)}. "
                    "È una versione sintetica del dataset medico originale generata tramite VAE."
                )

            else:
                result_text = (
                    "Non ho riconosciuto completamente la domanda. "
                    "Prova con una domanda su correlazione, media, minimo, massimo o distribuzione."
                )

            return result_text, fig

        def improve_with_llm(question, local_answer, data):
            api_key = os.environ.get("OPENAI_API_KEY")

            if not api_key:
                return local_answer + "\n\nNota: chiave API non disponibile nell'ambiente Streamlit."

            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key)

                summary_stats = data.describe().round(3).to_string()
                corr_matrix = data.corr().round(3).to_string()

                response = client.responses.create(
                    model="gpt-4.1-mini",
                    instructions=(
                        "Sei un assistente di analisi dati. "
                        "Rispondi in italiano, in modo semplice e chiaro. "
                        "Non inventare dati. Usa solo il risultato locale e il contesto aggregato."
                    ),
                    input=f"""
Domanda:
{question}

Risultato calcolato localmente:
{local_answer}

Statistiche aggregate:
{summary_stats}

Correlazioni aggregate:
{corr_matrix}

Rispondi in modo breve e comprensibile.
"""
                )

                return response.output_text

            except Exception as e:
                return local_answer + f"\n\nNota: LLM non disponibile. Errore: {str(e)}"

        if st.button("Analizza domanda"):
            local_answer, fig = local_analysis(question, synthetic_data)

            st.subheader("Risposta")
            st.write(local_answer)

            if fig is not None:
                st.pyplot(fig)

            if use_llm:
                st.subheader("Risposta migliorata con LLM")
                final_answer = improve_with_llm(question, local_answer, synthetic_data)
                st.write(final_answer)

# -------------------------------------------------
# Upload CSV Demo
# -------------------------------------------------

elif section == "Upload CSV Demo":
    st.header("Upload CSV Demo")

    st.write(
        "Questa sezione mostra la parte di caricamento file richiesta dalla piattaforma. "
        "L'utente può caricare un CSV e visualizzarne una prima analisi."
    )

    uploaded_file = st.file_uploader("Carica un file CSV", type=["csv"])

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)

        st.success("File CSV caricato correttamente.")
        st.write("Dimensione del file:", uploaded_df.shape)

        st.subheader("Prime righe")
        st.dataframe(uploaded_df.head(), use_container_width=True)

        st.subheader("Statistiche descrittive")
        st.dataframe(uploaded_df.describe(), use_container_width=True)
    else:
        st.info("Carica un file CSV per visualizzarlo nella dashboard.")
