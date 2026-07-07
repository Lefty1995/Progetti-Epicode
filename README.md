# Synthetic Data Sandbox - VAE Edition

## Descrizione del progetto

Questo progetto ha l'obiettivo di generare dati sintetici partendo da un dataset tabellare reale.

Il sistema utilizza un modello VAE, cioè un Variational Autoencoder, per imparare la struttura generale del dataset originale e generare un nuovo dataset sintetico con caratteristiche statistiche simili.

Il dataset utilizzato è il Pima Indians Diabetes Dataset, un dataset medico pubblico. Anche se il dataset è pubblico, nel progetto viene trattato come esempio di dato sensibile, perché contiene informazioni legate alla salute.

## Obiettivo

L'obiettivo principale è creare una piccola piattaforma in grado di:

- caricare e preparare un dataset tabellare;
- addestrare un modello VAE;
- generare un dataset sintetico;
- confrontare dati reali e dati sintetici;
- controllare che i dati sintetici non siano copie dirette dei dati reali;
- visualizzare i risultati tramite una dashboard Streamlit;
- esporre alcuni endpoint tramite FastAPI;
- permettere una semplice analisi in linguaggio naturale sui dati sintetici.

## Struttura del progetto

Synthetic_Data_Sandbox_VAE/

- Synthetic_Data_Sandbox_VAE.ipynb
- app.py
- api.py
- README.md
- PROJECT_SUMMARY.md
- requirements.txt
- .gitignore
- vae_model.pth
- real_clean_data.csv
- synthetic_data.csv
- statistics_comparison.csv
- privacy_check_report.csv
- real_correlation_matrix.csv
- synthetic_correlation_matrix.csv
- correlation_difference_matrix.csv
- images/

## Tecnologie utilizzate

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Matplotlib
- Seaborn
- Streamlit
- FastAPI
- OpenAI API opzionale

## Dataset

Il dataset utilizzato è il Pima Indians Diabetes Dataset.

Contiene 768 righe e 9 colonne:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

## Fasi principali

### 1. Preparazione dei dati

Il dataset viene caricato, controllato e pulito.

Alcuni valori pari a 0 in colonne mediche vengono trattati come valori mancanti e sostituiti con la mediana della colonna.

Successivamente i dati vengono normalizzati tra 0 e 1.

### 2. Creazione del modello VAE

Il modello VAE è composto da:

- Encoder;
- spazio latente;
- Decoder;
- Reparameterization Trick.

Il modello impara a ricostruire i dati originali e a generare nuovi dati partendo dallo spazio latente.

### 3. Addestramento

Il modello viene addestrato sui dati normalizzati.

La funzione di loss è composta da:

- Reconstruction Loss;
- KL Divergence.

### 4. Generazione dati sintetici

Dopo l'addestramento, il decoder del VAE viene usato per generare un nuovo dataset sintetico.

Il dataset generato viene salvato nel file synthetic_data.csv.

### 5. Validazione

I dati reali e sintetici vengono confrontati tramite:

- statistiche descrittive;
- distribuzioni delle variabili principali;
- confronto della variabile Outcome;
- matrici di correlazione;
- differenza tra correlazioni reali e sintetiche.

### 6. Privacy Check

Il progetto controlla che il dataset sintetico non contenga righe identiche al dataset reale.

Viene anche calcolata la distanza tra ogni riga sintetica e la riga reale più vicina.

Nel test eseguito:

- non sono state trovate righe sintetiche identiche a righe reali;
- non sono stati trovati duplicati nel dataset sintetico;
- una riga sintetica è risultata molto vicina a una riga reale secondo la soglia scelta.

Questo viene indicato come limite del progetto.

### 7. Natural Language Analyst

Il progetto include una piccola funzione che permette di fare domande semplici sul dataset sintetico.

Esempi:

- Qual è la correlazione tra Glucose e BMI?
- Qual è la media di Age?
- Mostrami la distribuzione di Outcome.

La risposta viene calcolata localmente con Python.

Se è disponibile una chiave API OpenAI, la risposta può essere resa più chiara tramite LLM.

## Interfaccia Streamlit

La dashboard Streamlit permette di visualizzare:

- overview del progetto;
- dataset reale e sintetico;
- statistiche;
- grafici di validazione;
- privacy check;
- Natural Language Analyst;
- demo upload CSV.

Per avviare la dashboard in locale:

streamlit run app.py

Su Google Colab è necessario usare un tunnel, ad esempio Cloudflare Tunnel.

## Backend FastAPI

Il progetto include anche un backend FastAPI con questi endpoint:

- /
- /upload
- /train
- /generate
- /privacy-report
- /statistics
- /dataset-info

Per avviare FastAPI:

uvicorn api:app --host 0.0.0.0 --port 8000

La documentazione automatica è disponibile su:

/docs

## Sicurezza API Key

La chiave API OpenAI non deve essere mai scritta nel codice e non deve essere caricata su GitHub.

Nel progetto è stata usata la funzione Secrets di Google Colab con il nome:

OPENAI_API_KEY

Se la chiave non è disponibile, il progetto funziona comunque usando solo l'analisi locale.

## Come eseguire il progetto

1. Aprire il notebook su Google Colab.
2. Attivare la GPU.
3. Eseguire gli step del notebook in ordine.
4. Generare il dataset sintetico.
5. Controllare i grafici di validazione.
6. Avviare la dashboard Streamlit.
7. Avviare FastAPI e controllare la pagina /docs.

## Risultato finale

Il progetto finale produce:

- un modello VAE addestrato;
- un dataset sintetico;
- grafici di confronto;
- report privacy;
- dashboard Streamlit;
- backend FastAPI;
- funzione di analisi in linguaggio naturale.

## Conclusione

Questo progetto mostra come un modello VAE possa essere usato per generare dati sintetici partendo da dati tabellari reali.

Il sistema non si limita alla generazione dei dati, ma include anche preparazione, validazione, controllo privacy, interfaccia grafica e backend API.