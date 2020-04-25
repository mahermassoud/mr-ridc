#%% Imports
import pandas as pd
from os import path

#%% Read in data
data_fp =  "/Users/massoudmaher/Documents/Code/mr-ridc/data"

drug_info = pd.read_csv(path.join(data_fp, "secondary-screen-replicate-treatment-info.csv"))
drug_response = pd.read_csv(path.join(data_fp, "secondary-screen-replicate-collapsed-logfold-change.csv"))

#%% Rename to cell line
drug_response = drug_info.rename(columns={"Unnamed: 0": "cell_line"})

#%% Get only the cell lines we have crispr+rna-seq for