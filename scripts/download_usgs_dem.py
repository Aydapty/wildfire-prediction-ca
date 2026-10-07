import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Make sure destination directory exists
os.makedirs("data/raw/dem", exist_ok=True)

# Read URLs, stripping out invalid lines or headers
with open("data/raw/usgs_dem/data.txt", "r") as f:
    urls = [
        line.strip()
        for line in f
        if line.strip().startswith("http")
    ]

def download_file(url):
    filename = url.split("/")[-1]
    dest_path = os.path.join("data/raw/dem", filename)
    
    # Skip if already downloaded
    if os.path.exists(dest_path):
        print(f"Skipping (already exists): {filename}")
        return
        
    print(f"Downloading: {filename}")
    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as e:
        print(f"Failed {filename}: {e}")

# Download 5 files at a time in parallel
with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(download_file, urls)

print("All downloads complete!")