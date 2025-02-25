from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import ollama
import PyPDF2

app = Flask(__name__)

# Carpeta para almacenar temporalmente los PDFs
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Endpoint para subir un PDF y extraer su texto
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Extraer texto del PDF
    text = extract_text_from_pdf(filepath)
    return jsonify({"text": text})


# Función para extraer texto del PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join(
            [page.extract_text() for page in reader.pages if page.extract_text()]
        )
    return text if text else "No text found"


# Endpoint para hacer consultas y obtener respuestas de Ollama
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Llamar a Ollama para generar respuesta
    response = ollama.chat(
        model="mistral", messages=[{"role": "user", "content": question}]
    )
    return jsonify({"response": response["message"]["content"]})


if __name__ == "__main__":
    app.run(debug=True)
