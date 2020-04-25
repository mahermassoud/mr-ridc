#%% Imports
# Script to stack dosages onto drug response
import pandas as pd
from os import path
import numpy as np

#%% Read in data
data_fp =  "/Users/massoudmaher/Documents/Code/mr-ridc/data"

drug_info = pd.read_csv(path.join(data_fp, "secondary-screen-replicate-treatment-info.csv"))
drug_response = pd.read_csv(path.join(data_fp, "secondary-screen-replicate-collapsed-logfold-change.csv"))
other_cell_lines = np.load(path.join(data_fp, "clean_cell_lines.npy"), allow_pickle=True)
drug_response = drug_response.rename(columns={"Unnamed: 0": "cell_line"})