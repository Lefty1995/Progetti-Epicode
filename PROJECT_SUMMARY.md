# Project Summary - Synthetic Data Sandbox VAE

## Titolo

Synthetic Data Sandbox - VAE Edition

## Obiettivo

Creare una piattaforma dimostrativa per generare dati sintetici da un dataset tabellare usando un Variational Autoencoder.

## Dataset

Il dataset utilizzato è il Pima Indians Diabetes Dataset.

È stato scelto perché:

- è tabellare;
- contiene dati numerici;
- rappresenta un caso vicino ai dati medici;
- è semplice da gestire in Google Colab;
- permette una buona dimostrazione del concetto di dati sintetici.

## Modello

Il modello usato è un VAE realizzato in PyTorch.

Componenti principali:

- Encoder;
- Reparameterization Trick;
- Decoder;
- Loss composta da Reconstruction Loss e KL Divergence.

## Output generati

Il progetto genera:

- synthetic_data.csv;
- statistics_comparison.csv;
- privacy_check_report.csv;
- matrici di correlazione;
- grafici di validazione;
- dashboard Streamlit;
- API FastAPI.

## Validazione

Sono stati confrontati dati reali e sintetici usando:

- statistiche descrittive;
- distribuzioni;
- confronto della colonna Outcome;
- matrici di correlazione;
- differenze tra correlazioni.

## Privacy Check

Il controllo privacy ha verificato:

- righe identiche tra reale e sintetico;
- duplicati nel dataset sintetico;
- distanza tra righe sintetiche e righe reali.

Risultato principale:

- nessuna riga sintetica identica a una riga reale;
- nessun duplicato nel dataset sintetico;
- una riga sintetica molto vicina a una riga reale, indicata come limite del progetto.

## Interfaccia

È stata creata una dashboard Streamlit con sezioni per:

- overview;
- dataset;
- validazione;
- privacy check;
- Natural Language Analyst;
- upload CSV demo.

## Backend

È stato creato un backend FastAPI con endpoint per:

- upload CSV;
- train;
- generate;
- privacy report;
- statistics;
- dataset info.

## Nota finale

Il progetto è stato pensato come demo didattica.

Non sostituisce una soluzione professionale completa per la privacy dei dati, ma mostra una pipeline completa e funzionante per generare e analizzare dati sintetici.