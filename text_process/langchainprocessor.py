import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, TypedDict, Tuple
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from sklearn.preprocessing import normalize
import numpy as np


# Función para extraer textos de diferentes fuentes
def extract_texts(pdf_directory: str) -> List[Document]:
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    return documents


# Función para dividir el texto en chunks
def chunk_text(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=1000)
    texts = text_splitter.split_documents(documents)
    return texts


# Función para crear y guardar el índice FAISS
def create_and_save_faiss_index(texts: List[Document]):
    # Crear embeddings y almacenarlos en un vectorstore
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    embeddings = embeddings_model.embed_documents([doc.page_content for doc in texts])

    # Normalizar los embeddings
    normalized_embeddings = normalize(np.array(embeddings), norm="l2")

    # Crear el índice FAISS con los embeddings normalizados
    vectorstore = FAISS.from_embeddings(
        list(zip([doc.page_content for doc in texts], normalized_embeddings)),
        embeddings_model,
    )
    return vectorstore, embeddings_model


# Función para formatear la respuesta
def format_response(response: str) -> str:
    lines = response.strip().split("\n")
    formatted_lines = [f"> {line.strip()}" for line in lines if line.strip()]
    return "\n".join(formatted_lines)


# Define el estado de la aplicación
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    scores: List[float]  # Añadimos las puntuaciones de similitud al estado


# Función para recuperar documentos relevantes con puntuaciones
def retrieve(state: State, vector_store) -> State:
    # Recuperar los k documentos más similares con sus puntuaciones
    retrieved_docs_with_scores = vector_store.similarity_search_with_score(
        state["question"], k=2
    )
    # Separar documentos y puntuaciones
    retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
    scores = [score for _, score in retrieved_docs_with_scores]
    return {"context": retrieved_docs, "scores": scores}


# Función para generar una respuesta usando el LLM
def generate(state: State, prompt, llm) -> State:
    # Combinar el contenido de los documentos recuperados
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # Invocar el prompt con la pregunta y el contexto
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    print(messages.text, len(messages.text))
    # Generar la respuesta usando el LLM
    response = llm.invoke(messages)
    return {"answer": response}


# Define el prompt template
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Keep the answer concise.

    Question: {question}

    Context: {context}

    Answer:
    """,
)


# Función principal
def main():
    # Directorios y rutas
    pdf_directory = "data"

    # Extraer textos
    documents = extract_texts(pdf_directory)

    # Dividir el texto en chunks
    texts = chunk_text(documents)

    # Crear y guardar el índice FAISS
    faiss_index, embeddings = create_and_save_faiss_index(texts)

    # Crear el modelo LLM
    llm = OllamaLLM(model="llama3.2", temperature=1)

    # Compilar el grafo de estado
    graph_builder = StateGraph(State)

    # Añadir nodos
    graph_builder.add_node("retrieve", lambda state: retrieve(state, faiss_index))
    graph_builder.add_node(
        "generate", lambda state: generate(state, prompt_template, llm)
    )

    # Definir las conexiones entre nodos
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")

    # Establecer el punto de entrada
    graph_builder.set_entry_point("retrieve")

    # Compilar el grafo
    graph = graph_builder.compile()

    # Invocar el grafo con una pregunta
    response = graph.invoke(
        {
            # "question": "Parmi les personnes admises de 2012 à 2021, dit-moi taux de cuba en 2023 qui dépasse le seuil de 80"
            "question": "programme pilot intelligence artificielle"
        }
    )

    # Mostrar la respuesta generada
    print("Respuesta generada:")
    print(response["answer"])


if __name__ == "__main__":
    main()
