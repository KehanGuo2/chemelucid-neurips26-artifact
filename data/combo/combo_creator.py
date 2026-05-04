import json
import csv
import re
from pathlib import Path

#these have correct and incorrect labels

dag_data = {
    "nodes": [
        {"id": "BIG QUESTION", "label": """Molecule X has a molecular formula of {Molecular Formula}. The text carbon NMR spectra for Molecule X is {CNMR Spectra}. Molecule X’s carbon NMR was taken in {Solvent} solvent, on a machine with a frequency of {MHz Value}. From prior analysis of the carbon NMR, a list of potential structural possibilities and impossibilities for each carbon NMR peak has been created, here it is: {Blackboard Summarization of CNMR}. The text proton NMR of molecule X with each peak's shift, integration, multiplicity, and J value, listed respectively is: {HNMR Spectra: δ,n,m,J}. Molecule X’s proton NMR was taken in {Solvent} solvent, on a machine with a frequency of {MHz Value}. From prior analysis of the proton NMR, a list of potential structural possibilities and impossibilities for each proton NMR peak has been created, here it is: {Blackboard Summarization of HNMR}.

Given Molecule X’s molecular formula, carbon NMR, proton NMR, and potential peak possibilities provide 5 potential SMILES strings which could represent Molecule X. Rank these 1-5 with position 1 being the most likely and 5 being the least likely"""},

        {"id": "F1", "label": "For each SMILES string listed above, provide a Tannimoto Similarity score between the molecule and this SMILES string {GT SMILES}. Provide the answer in the format: Prediction1: Similarity Score, Prediction2: Similarity Score"},

    ],
  "edges": [
        {"source": "BIG QUESTION", "target": "F1"}, #JUMP
            ]
}
file_path = "COMBO_HRE_v1.json"
with open(file_path, "w") as f:
    json.dump(dag_data, f, indent=4)

print("Combo HRE Has Been Created")



