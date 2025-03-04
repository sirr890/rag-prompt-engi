from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import ollama
import PyPDF2
import shutil
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from text_process.text_extraction import TextExtractor
from text_process.text_index import Indexfaiss
import text_process.integration_ollama_llm as ollama_llm

app = Flask(__name__)
extractor = TextExtractor(chunk_size=1000, overlap=500)
indexfaiss = Indexfaiss()

# Carpeta para almacenar temporalmente los PDFs
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def delete_all_pdfs():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def upload_all_pdfs():
    directory = request.form.get("directory")
    print(directory)
    if not directory or not os.path.isdir(directory):
        return jsonify({"error": "Ruta inválida o no es un directorio"}), 400

    for pdf in os.listdir(directory):
        if pdf.endswith(".pdf"):
            src_path = os.path.join(directory, pdf)
            dest_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf)
            shutil.copy(src_path, dest_path)  # Copia el archivo
            print(f"📂 Copied: {src_path} -> {dest_path}")

    return jsonify({"message": "✅ All files uploaded."}), 200


# Endpoint para subir PDFs y crear indice FAISS
@app.route("/upload_pdfs", methods=["POST"])
def upload_pdfs():
    # delete all files from UPLOAD_FOLDER
    delete_all_pdfs()

    # Upload all files
    response, tuple = upload_all_pdfs()  # Subir los PDFs
    if tuple != 200:
        return response
    print(response)
    # Load documents from PDFs
    print("\n🔄 Extracting text from PDFs...")
    documents = extractor.extract_text_from_pdf_directory(UPLOAD_FOLDER)
    # saving documents to UPLOAD_FOLDER
    with open(f"{UPLOAD_FOLDER}/documents.json", "w") as outfile:
        json.dump(documents, outfile)
    print(f"✅ Documents saved to {UPLOAD_FOLDER}/documents.json.")

    # ✅ Generate embeddings
    print("\n🔄 Generating embeddings...")
    embeddings, document_names = indexfaiss.generate_embeddings(documents)
    faiss_path, metadata_path = (
        f"{UPLOAD_FOLDER}/faiss_index.index",
        f"{UPLOAD_FOLDER}/metadata.json",
    )
    index = indexfaiss.index_embeddings(
        embeddings,
    )
    indexfaiss.save_index(faiss_path, index)
    indexfaiss.save_metadata(metadata_path, document_names)
    print(f"✅ FAISS index and metadata saved.")
    return (
        jsonify(
            {
                "message": f"PDFs uploaded and FAISS index created successfully in {UPLOAD_FOLDER}."
            }
        ),
        200,
    )


# Endpoint para hacer consultas y obtener respuestas de Ollama
def get_context(faiss_path, metadata_path, query):
    # ✅ Load FAISS index and metadata
    index = indexfaiss.load_index(faiss_path)
    document_names = indexfaiss.load_metadata(metadata_path)
    similar_documents = indexfaiss.search_similar_documents(
        query, index, document_names, num_results=4
    )

    if not similar_documents:
        print("⚠ No similar documents found.")
    else:
        # ✅ Display search results
        print(f"\n📌 Top {len(similar_documents)} similar documents to '{query}':\n")
        for idx, doc in enumerate(similar_documents):
            print(
                f"🔹 {idx + 1}. Document: {doc['document_chunk']} | Distance: {doc['distance']:.4f}"
            )

        # ✅ Retrieve context from similar documents

        # Load documents from UPLOAD_FOLDER
        with open(f"{UPLOAD_FOLDER}/documents.json", "r") as infile:
            documents = json.load(infile)
        retrieved_texts = "\n\n".join(
            [
                documents.get(doc["document_chunk"], "⚠ Document not found.")
                for doc in similar_documents
            ]
        )

    return retrieved_texts


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Llamar a Ollama para generar respuesta
    context = get_context(
        f"{UPLOAD_FOLDER}/faiss_index.index", f"{UPLOAD_FOLDER}/metadata.json", question
    )
    response = ollama_llm.query_ollama(question, context)
    print(response)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
