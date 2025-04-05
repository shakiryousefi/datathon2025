import zipfile
import json
from pathlib import Path
from typing import List, Dict, Any

ZIP_PATHS = ["data/datathon_part1.zip", "data/datathon_part2.zip", "data/datathon_part3.zip", "data/datathon_part4.zip"]

REQUIRED_FILES = [
    "passport.json",
    "client_profile.json",
    "account_form.json",
    "client_description.json"
]
OPTIONAL_FILES = ["label.json"]

def extract_client_data(zip_path: Path) -> List[Dict]:
    clients = []
    # For every data part
    with zipfile.ZipFile(zip_path, 'r') as outer_zip:
        for client_zip_name in outer_zip.namelist():
            if not client_zip_name.endswith('.zip'):
                continue
            
            with outer_zip.open(client_zip_name) as client_zip_bytes:
                with zipfile.ZipFile(client_zip_bytes) as client_zip:
                    client_data = {}
                    for filename in REQUIRED_FILES + OPTIONAL_FILES:
                        stem = Path(filename).stem
                        try:
                            with client_zip.open(filename) as f:
                                client_data[stem] = json.load(f)
                        except KeyError:
                            if filename in REQUIRED_FILES:
                                raise FileNotFoundError(f"{filename} missing in {client_zip_name}")
                            else:
                                client_data[stem] = None
                    clients.append(client_data)
    
    return clients

def process_zips(paths: List[str]) -> List[Dict]:
    all_clients = []
    # For every data part
    for path in paths:
        path_obj = Path(path)
        if path_obj.exists() and path_obj.suffix == ".zip":
            clients = extract_client_data(path_obj)
            all_clients.extend(clients)
        else:
            print(f"Skipping invalid path: {path}")
    return all_clients

def get_all():
    return process_zips(ZIP_PATHS)