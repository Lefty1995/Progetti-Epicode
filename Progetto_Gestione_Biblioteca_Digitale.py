# ============================================
# PROGETTO: Gestione Biblioteca Digitale
# ============================================
# In questo file implemento un piccolo sistema per gestire
# libri, utenti e prestiti usando:
# - variabili e tipi di dati base
# - strutture dati (lista, dizionario, set)
# - classi e oggetti (OOP)
# - una funzione che simula i prestiti


# ============================
# PARTE 1 – Variabili e tipi di dati
# ============================
# Qui dichiaro alcune variabili di esempio legate a un libro,
# usando tipi di dati diversi: stringa, intero, float e booleano.

titolo = "Il Signore degli Anelli"      # stringa
copie = 5                               # intero
prezzo_medio = 15.99                    # float
disponibile = True                      # booleano

print("=== Parte 1: Variabili e tipi di dati ===")
print("Titolo:", titolo)
print("Copie disponibili:", copie)
print("Prezzo medio:", prezzo_medio)
print("Disponibile:", disponibile)
print()  # Riga vuota per separare l'output


# ============================
# PARTE 2 – Strutture dati
# ============================
# In questa parte uso tre strutture dati fondamentali di Python:
# - lista: per memorizzare più titoli di libri in ordine
# - dizionario: per collegare ogni titolo al numero di copie
# - set: per rappresentare l'insieme degli utenti registrati
#   (senza duplicati).

# Lista con almeno 5 titoli di libri
lista_titoli = [
    "Il Signore degli Anelli",
    "1984",
    "Il Nome della Rosa",
    "Harry Potter e la Pietra Filosofale",
    "Il Piccolo Principe"
]

# Dizionario: titolo -> numero di copie disponibili
dizionario_copie = {
    "Il Signore degli Anelli": 5,
    "1984": 3,
    "Il Nome della Rosa": 2,
    "Harry Potter e la Pietra Filosofale": 4,
    "Il Piccolo Principe": 6
}

# Insieme (set) di utenti registrati
utenti_registrati = {"Luca", "Giulia", "Marco", "Anna"}

print("=== Parte 2: Strutture dati ===")
print("Lista titoli:", lista_titoli)
print("Dizionario copie:", dizionario_copie)
print("Utenti registrati:", utenti_registrati)
print()


# ============================
# PARTE 3 – Classi e OOP
# ============================
# In questa sezione definisco le classi principali del dominio:
# - Libro: rappresenta un libro con titolo, autore, anno e copie.
# - Utente: rappresenta una persona registrata in biblioteca.
# - Prestito: collega un Utente a un Libro per un certo numero di giorni.
# Ogni classe ha un metodo che restituisce/stampa le informazioni
# principali dell'oggetto.


class Libro:
    def __init__(self, titolo, autore, anno, copie_disponibili):
        self.titolo = titolo
        self.autore = autore
        self.anno = anno
        self.copie_disponibili = copie_disponibili

    def info(self):
        # Restituisce una stringa descrittiva del libro
        return (f"'{self.titolo}' di {self.autore} "
                f"({self.anno}) - Copie disponibili: {self.copie_disponibili}")


class Utente:
    def __init__(self, nome, eta, id_utente):
        self.nome = nome
        self.eta = eta
        self.id_utente = id_utente

    def scheda(self):
        # Stampa i dati principali dell'utente
        print(f"Utente: {self.nome} - Età: {self.eta} - ID: {self.id_utente}")


class Prestito:
    def __init__(self, utente, libro, giorni):
        self.utente = utente    # oggetto Utente
        self.libro = libro      # oggetto Libro
        self.giorni = giorni    # numero di giorni del prestito

    def dettagli(self):
        # Stampa tutte le informazioni sul prestito
        print(
            f"Prestito: {self.utente.nome} ha preso "
            f"'{self.libro.titolo}' per {self.giorni} giorni."
        )


# ============================
# PARTE 4 – Funzionalità
# ============================
# Qui definisco la logica di base della "biblioteca":
# la funzione presta_libro controlla se il libro ha copie
# disponibili, aggiorna il numero di copie e crea un oggetto
# Prestito in caso di successo.


def presta_libro(utente, libro, giorni):
    """
    Verifica se il libro ha almeno 1 copia disponibile.
    Se sì: riduce le copie e crea un nuovo oggetto Prestito.
    Se no: stampa un messaggio di errore.
    """
    if libro.copie_disponibili >= 1:
        libro.copie_disponibili -= 1
        prestito = Prestito(utente, libro, giorni)
        return prestito
    else:
        print(f"Il libro '{libro.titolo}' non ha copie disponibili.")
        return None


# ============================
# BLOCCO PRINCIPALE DI ESECUZIONE
# ============================
# Nel main simulo la creazione di alcuni libri e utenti,
# effettuo almeno 3 prestiti e alla fine stampo:
# - le copie aggiornate per ogni libro
# - i dettagli di ogni prestito effettuato.


if __name__ == "__main__":
    print("=== Parte 3 e 4: OOP e Funzionalità ===")

    # Creo alcuni libri (coerenti con la parte 2)
    libro1 = Libro("Il Signore degli Anelli", "J.R.R. Tolkien", 1954, 5)
    libro2 = Libro("1984", "George Orwell", 1949, 3)
    libro3 = Libro("Il Nome della Rosa", "Umberto Eco", 1980, 2)

    # Creo alcuni utenti registrati
    utente1 = Utente("Luca", 30, "U001")
    utente2 = Utente("Giulia", 25, "U002")
    utente3 = Utente("Marco", 28, "U003")

    # Simulo almeno 3 prestiti con utenti e libri diversi
    prestiti_effettuati = []

    prestito1 = presta_libro(utente1, libro1, 7)
    if prestito1 is not None:
        prestiti_effettuati.append(prestito1)

    prestito2 = presta_libro(utente2, libro2, 10)
    if prestito2 is not None:
        prestiti_effettuati.append(prestito2)

    prestito3 = presta_libro(utente3, libro3, 3)
    if prestito3 is not None:
        prestiti_effettuati.append(prestito3)

    # Elenco aggiornato delle copie disponibili per ciascun libro
    print("\nCopie aggiornate dei libri:")
    for libro in [libro1, libro2, libro3]:
        print(libro.info())

    # Dettagli di ogni prestito effettuato
    print("\nDettagli prestiti effettuati:")
    for prestito in prestiti_effettuati:
        prestito.dettagli()
