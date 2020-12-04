import requests
from src.utils import download_file

import os
from zipfile import ZipFile

# Retrieve list of files from zenodo
tmp = requests.get("https://zenodo.org/api/records/4291940").json()
# Proceed files individually
for file in tmp["files"]:
    # Extract link
    link = file["links"]["self"]
    file_name = link.split("/")[-1]
    dst = "./"
    download_file(link, dst)
    # Unzip files
    with ZipFile(file_name, "r") as zip:
        zip.extractall(dst)

    # Remove zip files
    os.remove(dst + file_name)
