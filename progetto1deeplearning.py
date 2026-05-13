import numpy as np
import requests
from PIL import Image
from io import BytesIO

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# ─── CONFIGURAZIONE ───────────────────────────────────────────
IMMAGINE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/1200px-African_Bush_Elephant.jpg"
INPUT_SIZE = (224, 224)

# ─── 1. CARICAMENTO MODELLO ───────────────────────────────────
def carica_modello():
    print("Caricamento VGG16 (prima volta: ~528MB, attendere)...")
    modello = VGG16(weights='imagenet', include_top=True)
    print("Modello caricato.")
    return modello

# ─── 2. CARICAMENTO E PREPROCESSING IMMAGINE ─────────────────
def carica_immagine(url):
    risposta = requests.get(url)
    immagine = Image.open(BytesIO(risposta.content)).convert('RGB')
    immagine = immagine.resize(INPUT_SIZE)
    array = img_to_array(immagine)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return array

# ─── 3. CLASSIFICAZIONE ──────────────────────────────────────
def classifica(modello, immagine_preprocessata):
    predizioni = modello.predict(immagine_preprocessata)
    risultati = decode_predictions(predizioni, top=5)[0]
    return risultati

# ─── 4. STAMPA RISULTATI ─────────────────────────────────────
def stampa_risultati(risultati):
    print("\n===== TOP-5 PREDIZIONI =====")
    for i, (id_classe, nome_classe, probabilita) in enumerate(risultati):
        print(f"{i+1}. {nome_classe:<25} {probabilita*100:.2f}%")

# ─── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    modello = carica_modello()
    immagine = carica_immagine(IMMAGINE_URL)
    risultati = classifica(modello, immagine)
    stampa_risultati(risultati)