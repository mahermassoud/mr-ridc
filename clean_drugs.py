#%% Imports
import pandas as pd
from os import path
import numpy as np

#%% Read in data
data_fp =  "/Users/massoudmaher/Documents/Code/mr-ridc/data"

drug_info = pd.read_csv(path.join(data_fp, "secondary-screen-replicate-treatment-info.csv"))
drug_response = pd.read_csv(path.join(data_fp, "secondary-screen-replicate-collapsed-logfold-change.csv"))
other_cell_lines = np.load(path.join(data_fp, "clean_cell_lines.npy"), allow_pickle=True)
#%% Rename to cell line
drug_response = drug_response.rename(columns={"Unnamed: 0": "cell_line"})

#%% Cell lines we have crispr+rna-seq for
overlap_cells = set(drug_response["cell_line"]) & set(other_cell_lines)
overlap_cells = np.sort(np.array(list(overlap_cells)))
print("cell-lines we have drug response, crispr, rna for")
print("overlap_cells.shape")
print(overlap_cells.shape)
drug_response_only_cells = [drug for drug in drug_response["cell_line"]
                            if drug not in set(other_cell_lines)]
drug_response_only_cells = np.sort(np.array(drug_response_only_cells))
print("cell-lines we have drug response for, but not crispr+rna")
print("drug_response_only_cells.shape")
print(drug_response_only_cells.shape)

# %% Make drug_response columns (drug IDs) match "broad_id" in drug_info
rn_drug_resp = drug_response
rn_drug_resp.columns= (["cell_line"] +
                       [c.split("::")[0] for c in rn_drug_resp.columns[1:]])

"""
We're gonna have two datasets:
one with only the cell lines that overlap between drug response, crispr, rna-seq
and one with all cell lines we have drug response for
"""
# %% Function we will use to get both datasets once we sort and subset
def get_out_files(drug_response, drug_info, cell_lines, drugs):
  """
  drug response and info should already be subsetted to same order as 
  cell lines, drugs
  """
  drug_resp_np = drug_response.iloc[:,1:].to_numpy()

  return drug_resp_np
#%% all cell lines we have drug response for, clean drug response
info_resp_drug = set(rn_drug_resp.columns[1:]) & set(drug_info["broad_id"])
info_resp_drug = np.sort(np.array(list(info_resp_drug)))
print("n drugs in info and response")
print(info_resp_drug.shape)
print("n drugs in info")
print(drug_info["broad_id"].unique().shape)
print("n drugs in response")
print(rn_drug_resp.shape[1]-1)

ov_drug_resp = rn_drug_resp[["cell_line"] + list(info_resp_drug)]
ov_drug_resp = ov_drug_resp.sort_values("cell_line")
ov_drug_resp_np = ov_drug_resp.iloc[:,1:].to_numpy()

cells_ov_drug = ov_drug_resp["cell_line"].to_numpy()
drugs_ov_drug = ov_drug_resp.columns[1:].to_numpy()

# %% all cell lines we have drug response for, clean drug_info
ov_drug_info = drug_info[drug_info["broad_id"] in drugs_ov_drug]














# %%
