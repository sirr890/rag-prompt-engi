import os
import requests
import json
from bs4 import BeautifulSoup

file_path = "data/sites.json"
output_folder = "data/html_files"

# Create the output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Read the URLs from the JSON file
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)
    urls = data.get("urls", [])

# Download and save each webpage
for i, url in enumerate(urls, 1):
    try:
        response = requests.get(urls[url], timeout=10)
        response.raise_for_status()  # Raise an error if the response is not 200

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Generate the filename based on the index and domain
        domain = url.split("//")[-1].split("/")[0]  # Extract the domain
        file_name = f"{i}_{domain}.html"
        file_path = os.path.join(output_folder, file_name)

        # Save the parsed content to a file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(soup))

        print(f"✅ Saved: {file_name}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error with {url}: {e}")
