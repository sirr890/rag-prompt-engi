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

    def index_embeddings(self, embeddings):
        """
        Indexes embeddings in FAISS

        Args:
            embeddings (np.array): Numpy array of document embeddings.
        """
        try:
            # Initialize FAISS index
            index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity

            # Add embeddings to the index
            index.add(embeddings)

            return index
        except Exception as e:
            print(f"❌ Error during FAISS indexing: {str(e)}")

    def save_index(self, path_index, index):
        """
        Saves a FAISS index to a file.

        Args:
            path_index (str): Path to save the FAISS index.
            index (faiss.Index): FAISS index to save.
        """
        try:
            faiss.write_index(index, path_index)
            print(f"✅ FAISS index saved at {path_index}.")
        except Exception as e:
            print(f"❌ Error saving FAISS index: {str(e)}")

    def save_metadata(self, path_metadata, documents):
        """
        Saves document metadata to a file.

        Args:
            path_metadata (str): Path to save the metadata file.
            documents (list): List of document names.
        """
        try:
            with open(path_metadata, "w", encoding="utf-8") as f:
                json.dump(documents, f, ensure_ascii=False, indent=4)
            print(f"✅ Metadata saved at {path_metadata}.")
        except Exception as e:
            print(f"❌ Error saving metadata: {str(e)}")

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

    def search_similar_documents(self, query, index, document_names, num_results=5):
        """
        Searches for the most similar documents to a given query.

        Args:
            query (str): The query string.
            index (str): FAISS index.
            metadata (str): Metadata containing document names.
            num_results (int): Number of top similar documents to retrieve.

        Returns:
            list[dict]: A list of dictionaries containing document names and their similarity distances.
        """

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
