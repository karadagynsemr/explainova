import pandas as pd
from pathlib import Path


def load_dataset(uploaded_file):
   # csv, xlsx, xls , tsv 

    file_name = uploaded_file.name
    file_extension = Path(file_name).suffix.lower()

    if file_extension == ".csv":
        df = pd.read_csv(uploaded_file)

    elif file_extension in [".xlsx", ".xls"]:
        df = pd.read_excel(uploaded_file)

    elif file_extension == ".tsv":
        df = pd.read_csv(uploaded_file, sep="\t")

    else:
        raise ValueError(
            "Unsupported file format. Please upload a CSV, XLSX, XLS, or TSV file."
        )

    return df
