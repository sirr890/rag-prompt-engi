import requests
from text_index import Indexfaiss
from text_extraction import TextExtractor
from text_extraction import clean_text


def query_ollama(query, context, model="llama3.2", temperature=0.9):
    """Send the query and retrieved context to Ollama for response generation."""
    try:
        response = requests.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an imigration AI assistant.",
                    },
                    {
                        "role": "user",
                        "content": f" You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Keep the answer concise.\n\n Question:{query}\n\n Context: {context}\n\n Answer",
                    },
                ],
                "temperature": temperature,
            },
        )

        if response.status_code == 200:
            ollama_data = response.json()
            return (
                ollama_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "⚠ No valid response received from Ollama.")
            )
        return f"❌ Error querying Ollama: {response.status_code}"

    except requests.exceptions.RequestException as e:
        return f"❌ Connection error querying Ollama: {str(e)}"


if __name__ == "__main__":
    # ✅ Initialize TextExtractor
    extractor = TextExtractor(chunk_size=1000, overlap=500)
    indexfaiss = Indexfaiss()

    # ✅ Load documents from PDFs, URLs, and Local HTML files
    print("\n🔄 Extracting text from PDFs...")
    pdf_directory = "data"
    documents = extractor.extract_text_from_pdf_directory(pdf_directory)

    print("\n🔄 Extracting text from online URLs...")
    # local_documents = extractor.extract_text_from_local_html("data/sites_local.json")

    # documents.update(local_documents)
    print("\n✅ All documents extracted.")

    # ✅ Generate embeddings
    print("\n🔄 Generating embeddings...")
    embeddings, document_names = indexfaiss.generate_embeddings(documents)
    indexfaiss.index_embeddings(
        embeddings,
        document_names,
        faiss_path="faiss_index.index",
        metadata_path="metadata.json",
    )
    print("✅ FAISS index and metadata saved.")

    # ✅ Example Query
    # query = "taux de présence des immigrants de cuba au Québec entre 2012 et 2021."
    # query = "Parmi les personnes admises de 2012 à 2021, dit-moi taux de Cuba en 2023 qui dépasse le seuil de 80"
    query = "programme pilot intelligence artificielle"
    print(f"\n🔍 Searching for documents related to: '{query}'")

    # # ✅ Clean query text if needed
    # if "clean_text" in globals():
    #     query = clean_text(query)
    # else:
    #     print("⚠ Warning: `clean_text` function not found. Using raw query.")

    # ✅ Search for similar documents
    try:
        similar_documents = indexfaiss.search_similar_documents(
            query, "faiss_index.index", "metadata.json", num_results=4
        )

        if not similar_documents:
            print("⚠ No similar documents found.")
        else:
            # ✅ Display search results
            print(
                f"\n📌 Top {len(similar_documents)} similar documents to '{query}':\n"
            )
            for idx, doc in enumerate(similar_documents):
                print(
                    f"🔹 {idx + 1}. Document: {doc['document_chunk']} | Distance: {doc['distance']:.4f}"
                )

            # ✅ Retrieve context from similar documents
            retrieved_texts = "\n\n".join(
                [
                    documents.get(doc["document_chunk"], "⚠ Document not found.")
                    for doc in similar_documents
                ]
            )

            # ✅ Query Ollama with retrieved context
            print("\n🤖 Querying Ollama for response...")
            print(query, retrieved_texts)
            ollama_response = query_ollama(
                query, retrieved_texts
            )  # , model="deepseek-r1")

            print("\n📌 Ollama Response:\n")
            print(ollama_response)

    except Exception as e:
        print(f"❌ Error during search: {str(e)}")
