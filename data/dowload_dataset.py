import os
DATA_DIR = "data"
FILE_PATH = os.path.join(DATA_DIR, "Shakespeare.txt")
url = "https://www.gutenberg.org/files/100/100-0.txt"
def download_dataset(url, file_path):
    import requests
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Dataset downloaded and saved to {file_path}")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_dataset(url, FILE_PATH)    
    