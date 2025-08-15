"""Download Brain-to-Text 2025 dataset from Dryad."""

import os
import sys
import urllib.request
import json
import zipfile


def display_progress_bar(block_num, block_size, total_size, message=""):
    bytes_downloaded_so_far = block_num * block_size
    MB_downloaded_so_far = bytes_downloaded_so_far / 1e6
    MB_total = total_size / 1e6
    sys.stdout.write(
        f"\r{message}\t\t{MB_downloaded_so_far:.1f} MB / {MB_total:.1f} MB"
    )
    sys.stdout.flush()


def download_dataset(data_dir="data", doi="10.5061/dryad.dncjsxm85"):
    """Download all files associated with the given Dryad DOI."""
    DRYAD_ROOT = "https://datadryad.org"
    data_dirpath = os.path.abspath(data_dir)
    os.makedirs(data_dirpath, exist_ok=True)

    urlified_doi = doi.replace("/", "%2F")
    versions_url = f"{DRYAD_ROOT}/api/v2/datasets/doi:{urlified_doi}/versions"
    with urllib.request.urlopen(versions_url) as response:
        versions_info = json.loads(response.read().decode())

    files_url_path = versions_info["_embedded"]["stash:versions"][-1]["_links"]["stash:files"]["href"]
    files_url = f"{DRYAD_ROOT}{files_url_path}"
    with urllib.request.urlopen(files_url) as response:
        files_info = json.loads(response.read().decode())

    for file_info in files_info["_embedded"]["stash:files"]:
        filename = file_info["path"]
        if filename == "README.md":
            continue

        download_path = file_info["_links"]["stash:download"]["href"]
        download_url = f"{DRYAD_ROOT}{download_path}"
        download_to = os.path.join(data_dirpath, filename)
        urllib.request.urlretrieve(
            download_url,
            download_to,
            reporthook=lambda *args: display_progress_bar(*args, message=f"Downloading {filename}")
        )
        sys.stdout.write("\n")

        if file_info["mimeType"] == "application/zip":
            print(f"Extracting files from {filename} ...")
            with zipfile.ZipFile(download_to, "r") as zf:
                zf.extractall(data_dirpath)

    print(f"\nDownload complete. See data files in {data_dirpath}\n")


if __name__ == "__main__":
    download_dataset()
