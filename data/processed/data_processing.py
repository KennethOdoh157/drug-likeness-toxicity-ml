"""
Stage 3: Data Cleaning & Label Preparation
------------------------------------------

This script cleans the raw Tox21 dataset by:
- Removing invalid SMILES
- Dropping molecules with missing assay labels
- Converting labels to integer format

Output:
- data/processed/tox21_clean.csv
"""

import pandas as pd
from rdkit import Chem
from pathlib import Path


# -----------------------------
# Paths (relative to this file)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_TOX21_PATH = BASE_DIR / "data" / "raw" / "tox21.csv"
PROCESSED_TOX21_PATH = BASE_DIR / "data" / "processed" / "tox21_clean.csv"


# -----------------------------
# Load Raw Data
# -----------------------------
tox21 = pd.read_csv(RAW_TOX21_PATH)
print(f"Original dataset shape: {tox21.shape}")


# -----------------------------
# Validate SMILES
# -----------------------------
def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


tox21["valid_smiles"] = tox21["smiles"].apply(is_valid_smiles)
print("\nSMILES validity:")
print(tox21["valid_smiles"].value_counts())


# -----------------------------
# Remove Invalid Molecules
# -----------------------------
tox21 = tox21[tox21["valid_smiles"]].copy()
print(f"\nAfter removing invalid SMILES: {tox21.shape}")


# -----------------------------
# Assay Columns
# -----------------------------
ASSAY_COLUMNS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]


# -----------------------------
# Drop Missing Labels
# -----------------------------
tox21_clean = tox21.dropna(subset=ASSAY_COLUMNS)
print(f"After dropping missing labels: {tox21_clean.shape}")


# -----------------------------
# Convert Labels to Integers
# -----------------------------
tox21_clean[ASSAY_COLUMNS] = tox21_clean[ASSAY_COLUMNS].astype(int)


# -----------------------------
# Final Sanity Check
# -----------------------------
print("\nFinal dataset info:")
print(tox21_clean.info())


# -----------------------------
# Save Clean Dataset
# -----------------------------
tox21_clean.to_csv(PROCESSED_TOX21_PATH, index=False)
print(f"\nClean dataset saved to: {PROCESSED_TOX21_PATH}")
