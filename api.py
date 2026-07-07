
import os
import pandas as pd

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(
    title="Synthetic Data Sandbox API",
    description="Backend API per il progetto Synthetic Data Sandbox - VAE Edition",
    version="1.0"
)

# -------------------------------------------------
# Endpoint base
# -------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Synthetic Data Sandbox API attiva",
        "project": "VAE Edition",
        "available_endpoints": [
            "/upload",
            "/train",
            "/generate",
            "/privacy-report",
            "/statistics"
        ]
    }


# -------------------------------------------------
# Upload CSV
# -------------------------------------------------

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Endpoint per caricare un file CSV.
    Il file viene salvato come uploaded_dataset.csv.
    """

    if not file.filename.endswith(".csv"):
        return JSONResponse(
            status_code=400,
            content={"error": "Il file deve essere in formato CSV."}
        )

    file_path = "uploaded_dataset.csv"

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        df = pd.read_csv(file_path)

        return {
            "message": "File CSV caricato correttamente.",
            "filename": file.filename,
            "saved_as": file_path,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist()
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Errore durante la lettura del CSV: {str(e)}"}
        )


# -------------------------------------------------
# Train
# -------------------------------------------------

@app.post("/train")
def train_model():
    """
    Endpoint dimostrativo per rappresentare l'avvio dell'addestramento del VAE.

    Nel notebook Colab l'addestramento vero è già stato eseguito nello Step 5.
    Questo endpoint serve a mostrare come il backend gestirebbe la richiesta.
    """

    model_path = "vae_model.pth"

    if os.path.exists(model_path):
        return {
            "message": "Modello VAE già addestrato e disponibile.",
            "model_file": model_path,
            "status": "ready"
        }

    return {
        "message": "Il modello non è ancora stato trovato.",
        "status": "missing_model",
        "suggestion": "Esegui prima lo Step 5 nel notebook Colab."
    }


# -------------------------------------------------
# Generate
# -------------------------------------------------

@app.get("/generate")
def generate_synthetic_data():
    """
    Endpoint per scaricare il dataset sintetico generato.
    """

    synthetic_path = "synthetic_data.csv"

    if not os.path.exists(synthetic_path):
        return JSONResponse(
            status_code=404,
            content={
                "error": "Dataset sintetico non trovato.",
                "suggestion": "Esegui prima lo Step 6 nel notebook Colab."
            }
        )

    return FileResponse(
        path=synthetic_path,
        filename="synthetic_data.csv",
        media_type="text/csv"
    )


# -------------------------------------------------
# Privacy Report
# -------------------------------------------------

@app.get("/privacy-report")
def get_privacy_report():
    """
    Endpoint per visualizzare il report del Privacy Check.
    """

    report_path = "privacy_check_report.csv"

    if not os.path.exists(report_path):
        return JSONResponse(
            status_code=404,
            content={
                "error": "Privacy report non trovato.",
                "suggestion": "Esegui prima lo Step 8 nel notebook Colab."
            }
        )

    report = pd.read_csv(report_path)

    return {
        "message": "Privacy report caricato correttamente.",
        "report": report.to_dict(orient="records")
    }


# -------------------------------------------------
# Statistics
# -------------------------------------------------

@app.get("/statistics")
def get_statistics_comparison():
    """
    Endpoint per visualizzare il confronto statistico tra dati reali e sintetici.
    """

    stats_path = "statistics_comparison.csv"

    if not os.path.exists(stats_path):
        return JSONResponse(
            status_code=404,
            content={
                "error": "File statistics_comparison.csv non trovato.",
                "suggestion": "Esegui prima lo Step 7 nel notebook Colab."
            }
        )

    stats = pd.read_csv(stats_path)

    return {
        "message": "Statistiche caricate correttamente.",
        "statistics": stats.to_dict(orient="records")
    }


# -------------------------------------------------
# Dataset Info
# -------------------------------------------------

@app.get("/dataset-info")
def get_dataset_info():
    """
    Endpoint per leggere informazioni base sui dataset del progetto.
    """

    real_path = "real_clean_data.csv"
    synthetic_path = "synthetic_data.csv"

    result = {}

    if os.path.exists(real_path):
        real_df = pd.read_csv(real_path)
        result["real_dataset"] = {
            "rows": real_df.shape[0],
            "columns": real_df.shape[1],
            "column_names": real_df.columns.tolist()
        }
    else:
        result["real_dataset"] = "File real_clean_data.csv non trovato."

    if os.path.exists(synthetic_path):
        synthetic_df = pd.read_csv(synthetic_path)
        result["synthetic_dataset"] = {
            "rows": synthetic_df.shape[0],
            "columns": synthetic_df.shape[1],
            "column_names": synthetic_df.columns.tolist()
        }
    else:
        result["synthetic_dataset"] = "File synthetic_data.csv non trovato."

    return result
