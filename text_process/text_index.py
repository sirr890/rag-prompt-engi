import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Indexfaiss:
    """A class for using FAISS to generate and search embeddings."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initializes the FAISS index and loads the embedding model."""
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, documents):
        """
        Generates embeddings for a dictionary {document_name: text}.

        Args:
            documents (dict): Dictionary where keys are document names and values are text chunks.

        Returns:
            np.array: Embeddings as a NumPy array.
            list: Document names corresponding to the embeddings.
        """
        if not documents:
            raise ValueError("⚠ Error: No documents to process for embeddings.")

        # Extract text values
        texts = list(documents.values())

        # Generate embeddings
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )

        # Normalize embeddings to improve FAISS performance
        faiss.normalize_L2(embeddings)

        return embeddings, list(documents.keys())

    def index_embeddings(self, embeddings, document_names, faiss_path, metadata_path):
        """
        Indexes embeddings in FAISS and saves metadata for retrieval.

        Args:
            embeddings (np.array): Numpy array of document embeddings.
            document_names (list): List of document names corresponding to embeddings.
            faiss_path (str): Path to save FAISS index.
            metadata_path (str): Path to save metadata (document names).
        """
        try:
            # Initialize FAISS index
            index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity

            # Add embeddings to the index
            index.add(embeddings)

            # Save FAISS index
            faiss.write_index(index, faiss_path)
            print(f"✅ FAISS index saved at {faiss_path}.")

            # Save metadata (document names)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(document_names, f, ensure_ascii=False, indent=4)

            print(f"✅ Metadata saved at {metadata_path}.")

        except Exception as e:
            print(f"❌ Error during FAISS indexing: {str(e)}")

    def load_index(self, index_path):
        """
        Loads a FAISS index from a file.

        Args:
            index_path (str): Path to the FAISS index file.

        Returns:
            faiss.Index: Loaded FAISS index or None if loading fails.
        """
        if not os.path.exists(index_path):
            print(f"❌ Error: FAISS index file '{index_path}' not found.")
            return None

        try:
            index = faiss.read_index(index_path)
            return index
        except Exception as e:
            print(f"❌ Error loading FAISS index: {str(e)}")
            return None

    def load_metadata(self, metadata_path):
        """
        Loads document metadata from a file.

        Args:
            metadata_path (str): Path to the metadata file.

        Returns:
            list: List of document names or an empty list if loading fails.
        """
        if not os.path.exists(metadata_path):
            print(f"❌ Error: Metadata file '{metadata_path}' not found.")
            return []

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading metadata: {str(e)}")
            return []

    def search_similar_documents(self, query, index_path, metadata_path, num_results=5):
        """
        Searches for the most similar documents to a given query.

        Args:
            query (str): The query string.
            index_path (str): Path to the FAISS index file.
            metadata_path (str): Path to the metadata file containing document names.
            num_results (int): Number of top similar documents to retrieve.

        Returns:
            list[dict]: A list of dictionaries containing document names and their similarity distances.
        """
        # ✅ Load FAISS index and metadata
        index = self.load_index(index_path)
        document_names = self.load_metadata(metadata_path)

        if index is None or not document_names:
            return []

        try:
            # ✅ Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)

            # Normalize query embedding for better similarity comparison
            faiss.normalize_L2(query_embedding)

            # ✅ Perform the search
            distances, indices = index.search(query_embedding, num_results)

            # ✅ Retrieve the most relevant documents
            similar_documents = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if 0 <= idx < len(document_names):  # Validate index range
                    similar_documents.append(
                        {
                            "document_chunk": document_names[idx],
                            "distance": float(
                                distances[0][i]
                            ),  # Convert to Python float
                        }
                    )

            return similar_documents

        except Exception as e:
            print(f"❌ Error during FAISS search: {str(e)}")
            return []
