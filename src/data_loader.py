import pandas as pd
import numpy as np
from pathlib import Path


MISSING_VALUE_MARKERS = {
    "?",
    "??",
    "-",
    "--",
    "---",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "missing",
    "unknown",
}


def normalize_missing_markers(df):
    data = df.copy()

    object_cols = data.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        stripped = data[col].astype(str).str.strip().str.lower()
        data.loc[stripped.isin(MISSING_VALUE_MARKERS), col] = np.nan

    return data


def load_dataset(uploaded_file):
   # csv, xlsx, xls , tsv 

    file_name = uploaded_file.name
    file_extension = Path(file_name).suffix.lower()

    if file_extension == ".csv":
        df = pd.read_csv(uploaded_file, na_values=list(MISSING_VALUE_MARKERS), keep_default_na=True)

    elif file_extension in [".xlsx", ".xls"]:
        df = pd.read_excel(uploaded_file, na_values=list(MISSING_VALUE_MARKERS), keep_default_na=True)

    elif file_extension == ".tsv":
        df = pd.read_csv(uploaded_file, sep="\t", na_values=list(MISSING_VALUE_MARKERS), keep_default_na=True)

    else:
        raise ValueError(
            "Unsupported file format. Please upload a CSV, XLSX, XLS, or TSV file."
        )

    return normalize_missing_markers(df)
