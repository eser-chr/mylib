import pandas as pd
from pathlib import Path


def retrieve_string_enmax_enmin(potcar_file_path:Path):
    with open(potcar_file_path, 'r') as file:
        for i, line in enumerate(file):
            if i == 18:
                return line.strip()
    raise ValueError("POTCAR has fewer than 19 lines.")


def retrieve_eniminmax_of_potcar(element_potcar_file: Path) -> dict:


    stringify = retrieve_string_enmax_enmin(element_potcar_file/"POTCAR")
    parts = stringify.replace(';', '').split()

    return {
        "potential": element_potcar_file.name,
        "enmax": float(parts[2]),
        "enmin": float(parts[5])
    }


def get_enminmax(potcar_folder: Path) -> pd.DataFrame:
    res = []
    for folder in potcar_folder.iterdir():
        if not folder.is_dir():
            continue
        try:
            res.append(
                retrieve_eniminmax_of_potcar(folder)
                )
            
            
        except Exception as e:
            print(f"Error processing {str(folder)}: {e}")
            continue

    df = pd.DataFrame(res)
    return df.sort_values(by=["potential", "enmax"], ignore_index=True)
