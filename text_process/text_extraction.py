import os
import json
import re
import unicodedata
import requests
from bs4 import BeautifulSoup
import fitz


def chunk_text(text, chunk_size=500, overlap=20):
    """Splits the text into chunks of `chunk_size` words with an `overlap` between chunks."""
    words = text.split()
    if not words:
        return []

    chunks = [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]
    return chunks


def clean_text(text):
    """
    Cleans a given text by removing special characters, URLs, extra spaces, and normalizing accents.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"\S+@\S+\.\S+", "", text)  # Remove emails
    # text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    # text = re.sub(
    # r"[^\w\s,.!?-]", "", text
    # )  # Keep only letters, numbers, and basic punctuation
    return text


class TextExtractor:
    """Extracts and processes text from PDFs, URLs, and HTML files."""

    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_from_pdf(self, pdf_file_path):
        if pdf_file_path:
            """Extracts text from a PDF file, cleans it, and splits it into chunks."""
            if not os.path.isfile(pdf_file_path):
                raise FileNotFoundError(f"❌ PDF file '{pdf_file_path}' not found.")

            with fitz.open(pdf_file_path) as doc:
                extracted_text = "\n".join([page.get_text("text") for page in doc])

            cleaned_text = clean_text(extracted_text)
            return chunk_text(cleaned_text, self.chunk_size, self.overlap)

    def extract_text_from_online_urls(self, json_path):
        if json_path:
            """Extracts text from a JSON file containing a list of online URLs."""
            if not os.path.isfile(json_path):
                raise FileNotFoundError(f"❌ JSON file '{json_path}' not found.")

            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    raise ValueError(f"❌ Error: Invalid JSON format in '{json_path}'.")

            urls = data.get("urls", {})
            if not urls:
                raise KeyError(f"❌ Error: No 'urls' found in JSON file '{json_path}'.")

            extracted_texts = {}

            for label, url in urls.items():
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, "html.parser")
                    text = " ".join([p.get_text() for p in soup.find_all("p")]).strip()

                    if not text:
                        print(
                            f"⚠ Warning: No text extracted from URL '{url}'. Skipping..."
                        )
                        continue

                    cleaned_text = clean_text(text)
                    chunks = chunk_text(cleaned_text, self.chunk_size, self.overlap)

                    for i, chunk in enumerate(chunks):
                        extracted_texts[f"{label}_chunk{i}"] = chunk

                    print(f"✅ Extracted text from online URL: {url}")

                except requests.exceptions.RequestException as e:
                    print(f"❌ Error fetching URL '{url}': {str(e)}")

            return extracted_texts

    def extract_text_from_local_html(self, json_path):
        if json_path:
            """Extracts text from multiple local HTML files listed in a JSON file."""
            if not os.path.isfile(json_path):
                raise FileNotFoundError(f"❌ JSON file '{json_path}' not found.")

            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    raise ValueError(f"❌ Error: Invalid JSON format in '{json_path}'.")

            html_files = data.get("urls", {})
            if not html_files:
                raise KeyError(f"❌ Error: No 'urls' found in JSON file '{json_path}'.")

            extracted_texts = {}

            for label, html_path in html_files.items():
                if not os.path.isfile(html_path):
                    print(
                        f"⚠ Warning: Local HTML file '{html_path}' not found. Skipping..."
                    )
                    continue

                with open(html_path, "r", encoding="utf-8") as file:
                    content = file.read()

                soup = BeautifulSoup(content, "html.parser")
                text = " ".join([p.get_text() for p in soup.find_all("p")]).strip()

                if not text:
                    print(
                        f"⚠ Warning: No text extracted from '{html_path}'. Skipping..."
                    )
                    continue

                cleaned_text = clean_text(text)
                chunks = chunk_text(cleaned_text, self.chunk_size, self.overlap)

                for i, chunk in enumerate(chunks):
                    extracted_texts[f"{label}_chunk{i}"] = chunk

                print(f"✅ Extracted text from local HTML file: {html_path}")

            return extracted_texts

    def extract_text_from_pdf_directory(self, pdfs_directory_path):
        """Processes all PDF files in a directory, extracting, cleaning, and chunking text."""
        if not os.path.isdir(pdfs_directory_path):
            raise NotADirectoryError(f"❌ Directory '{pdfs_directory_path}' not found.")

        documents = {}
        for pdf_file in os.listdir(pdfs_directory_path):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdfs_directory_path, pdf_file)
                chunks = self.extract_text_from_pdf(pdf_path)

                for i, chunk in enumerate(chunks):
                    documents[f"{pdf_file}_chunk{i}"] = chunk

                print(f"✅ Extracted text from {pdf_file}")

        return documents
