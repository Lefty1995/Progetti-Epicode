import numpy as np

# ===== PARTE 1 – Variabili e Tipi di Dati =====

# Paziente 1
nome1 = "Mario"
cognome1 = "Rossi"
codice_fiscale1 = "RSSMRA45A01H501Z"
eta1 = 45
peso1 = 78.5
analisi1 = ["emocromo", "glicemia", "colesterolo"]

# Paziente 2
nome2 = "Anna"
cognome2 = "Verdi"
codice_fiscale2 = "VRDNNA72B02L219X"
eta2 = 32
peso2 = 62.3
analisi2 = ["glicemia", "colesterolo", "trigliceridi"]

# Paziente 3
nome3 = "Luca"
cognome3 = "Bianchi"
codice_fiscale3 = "BNCLCU88C03M833Y"
eta3 = 58
peso3 = 85.0
analisi3 = ["emocromo", "trigliceridi", "creatinina"]

print("PARTE 1 – variabili di 3 pazienti definita.\n")

# ===== PARTE 2 – Classi e OOP =====

class Paziente:
    def __init__(self, nome, cognome, codice_fiscale, eta, peso, analisi_effettuate):
        self.nome = nome
        self.cognome = cognome
        self.codice_fiscale = codice_fiscale
        self.eta = eta
        self.peso = peso
        self.analisi_effettuate = analisi_effettuate  # lista di stringhe
        self.risultati_analisi = np.array([]) 
    
    def scheda_personale(self):
        return (f"Paziente: {self.nome} {self.cognome}\n"
                f"CF: {self.codice_fiscale} | Età: {self.eta} | Peso: {self.peso} kg\n"
                f"Analisi effettuate: {', '.join(self.analisi_effettuate)}")
    
    def statistiche_analisi(self):
        """Calcola statistiche con NumPy sui risultati analisi (Parte 4)."""
        if self.risultati_analisi is None or len(self.risultati_analisi) == 0:
            return "Nessun risultato analisi disponibile"
        
        media = np.mean(self.risultati_analisi)
        minimo = np.min(self.risultati_analisi)
        massimo = np.max(self.risultati_analisi)
        dev_std = np.std(self.risultati_analisi)
        
        return (f"Statistiche analisi ({len(self.risultati_analisi)} valori):\n"
                f"Media: {media:.1f}\n"
                f"Min: {minimo:.1f}\n"
                f"Max: {massimo:.1f}\n"
                f"Dev. std: {dev_std:.1f}")

class Medico:
    def __init__(self, nome, cognome, specializzazione):
        self.nome = nome
        self.cognome = cognome
        self.specializzazione = specializzazione
    
    def visita_paziente(self, paziente):
        print(f"Il Dr. {self.nome} {self.cognome} ({self.specializzazione}) "
            f"sta visitando il paziente {paziente.nome} {paziente.cognome}")

class Analisi:
    def __init__(self, tipo, risultato):
        self.tipo = tipo
        self.risultato = risultato  # valore numerico (float o int)
    
    def valuta(self):
        """Valuta il risultato in base a range realistici standard."""
        if self.tipo == "glicemia":
            # valori indicativi: 70-110 mg/dl normale
            return "NORMALE" if 70 <= self.risultato <= 110 else "ANOMALO"
        elif self.tipo == "colesterolo":
            # normale < 200 mg/dl
            return "NORMALE" if self.risultato < 200 else "ALTO"
        elif self.tipo == "trigliceridi":
            # normale < 150 mg/dl
            return "NORMALE" if self.risultato < 150 else "ALTO"
        elif self.tipo == "emocromo":
            # valore di esempio: emoglobina 12–17 g/dL
            return "NORMALE" if 12 <= self.risultato <= 17 else "ANOMALO"
        elif self.tipo == "creatinina":
            # valori indicativi adulti: 0.6–1.2 mg/dL
            return "NORMALE" if 0.6 <= self.risultato <= 1.2 else "ANOMALO"
        else:
            return "NON DEFINITO"

print("PARTE 2 – classi Paziente, Medico, Analisi definite.\n")

# ===== PARTE 3 – Uso di NumPy su 10 pazienti =====

valori_glicemia_10 = np.array([95, 112, 88, 145, 102, 78, 130, 99, 115, 92])

media_glicemia = np.mean(valori_glicemia_10)
max_glicemia = np.max(valori_glicemia_10)
min_glicemia = np.min(valori_glicemia_10)
std_glicemia = np.std(valori_glicemia_10)

print("PARTE 3 – Statistiche NumPy su glicemia 10 pazienti:")
print("Valori glicemia:", valori_glicemia_10)
print(f"Media: {media_glicemia:.1f}")
print(f"Max: {max_glicemia}")
print(f"Min: {min_glicemia}")
print(f"Deviazione standard: {std_glicemia:.1f}\n")

# ===== PARTE 5 – Applicazione completa (include Parte 4) =====

print("=" * 60)
print("GESTIONE CENTRO ANALISI MEDICHE - PROGRAMMA COMPLETO")
print("=" * 60)

# 3 medici
medico1 = Medico("Giovanni", "Ferrari", "Endocrinologo")
medico2 = Medico("Laura", "Galli", "Cardiologo")
medico3 = Medico("Marco", "Neri", "Ematologo")

# 5 pazienti, ognuno con almeno 3 analisi (nomi delle analisi)
pazienti = [
    Paziente("Mario", "Rossi", "RSSMRA45A01H501Z", 45, 78.5,
            ["emocromo", "glicemia", "colesterolo"]),
    Paziente("Anna", "Verdi", "VRDNNA72B02L219X", 32, 62.3,
            ["glicemia", "colesterolo", "trigliceridi"]),
    Paziente("Luca", "Bianchi", "BNCLCU88C03M833Y", 58, 85.0,
            ["emocromo", "trigliceridi", "creatinina"]),
    Paziente("Giulia", "Romani", "RMNGIU28D04F205W", 41, 70.2,
            ["glicemia", "emocromo", "colesterolo"]),
    Paziente("Francesco", "Moretti", "MRTFNC65E05G501V", 67, 92.1,
            ["trigliceridi", "creatinina", "glicemia"])
]

# Parte 4 – integrazione OOP + NumPy:
# assegniamo a ogni paziente un array NumPy con 3 risultati numerici
pazienti[0].risultati_analisi = np.array([14.5, 95, 185])   # emocromo, glicemia, colesterolo
pazienti[1].risultati_analisi = np.array([88, 210, 165])    # glicemia, colesterolo, trigliceridi
pazienti[2].risultati_analisi = np.array([13.2, 180, 1.1])  # emocromo, trigliceridi, creatinina
pazienti[3].risultati_analisi = np.array([112, 15.0, 195])  # glicemia, emocromo, colesterolo
pazienti[4].risultati_analisi = np.array([220, 1.4, 145])   # trigliceridi, creatinina, glicemia

# 1) Stampa scheda di ogni paziente + statistiche analisi
print("\n1. SCHEDE PAZIENTI E STATISTICHE ANALISI:")
for i, paziente in enumerate(pazienti, 1):
    print(f"\n--- Paziente {i} ---")
    print(paziente.scheda_personale())
    print(paziente.statistiche_analisi())

# 2) Mostra quale medico visita quale paziente (stampe sequenziali)
print("\n2. VISITE MEDICHE:")
visite = [
    (medico1, pazienti[0]),
    (medico2, pazienti[1]),
    (medico3, pazienti[2]),
    (medico1, pazienti[3]),
    (medico2, pazienti[4]),
]
for medico, paziente in visite:
    medico.visita_paziente(paziente)

# 3) Esempio uso classe Analisi con metodo valuta()
print("\n3. ESEMPIO SINGOLA ANALISI:")
analisi_esempio = Analisi("glicemia", 130)
print(f"Analisi: {analisi_esempio.tipo} = {analisi_esempio.risultato} → {analisi_esempio.valuta()}")

print("\nPROGRAMMA TERMINATO ✔")
