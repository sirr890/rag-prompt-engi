import os
import requests
import json
from bs4 import BeautifulSoup


file_path = "data/sites.json"
output_folder = "data/html_files"

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Leer las URLs desde el archivo JSON
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)
    urls = data.get("urls", [])

# Descargar y guardar cada página
for i, url in enumerate(urls, 1):
    try:
        response = requests.get(urls[url], timeout=10)
        response.raise_for_status()  # Lanza un error si la respuesta no es 200

        # Analizar el HTML con BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Nombre del archivo basado en el índice y dominio
        domain = url.split("//")[-1].split("/")[0]  # Extraer el dominio
        file_name = f"{i}_{domain}.html"
        file_path = os.path.join(output_folder, file_name)

        # Guardar el contenido analizado en un archivo
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(soup))

        print(f"[✔] Guardado: {file_name}")

    except requests.exceptions.RequestException as e:
        print(f"[✘] Error con {url}: {e}")
