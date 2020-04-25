#%% Imports
import pandas as pd
from os import path
import numpy as np

#%% Read in data
data_fp =  "/Users/massoudmaher/Documents/Code/mr-ridc/data"

drug_info = pd.read_csv(path.join(data_fp, "secondary-screen-replicate-treatment-info.csv"))
drug_response = pd.read_csv(path.join(data_fp, "secondary-screen-replicate-collapsed-logfold-change.csv"))
other_cell_lines = np.load(path.join(data_fp, "clean_cell_lines.npy"), allow_pickle=True)
drug_response = drug_response.rename(columns={"Unnamed: 0": "cell_line"})

#%% Cell lines we have crispr+rna-seq for
overlap_cells = set(drug_response["cell_line"]) & set(other_cell_lines)
overlap_cells = np.sort(np.array(list(overlap_cells)))
drug_response_only_cells = [drug for drug in drug_response["cell_line"]
                            if drug not in set(other_cell_lines)]
drug_response_only_cells = np.sort(np.array(drug_response_only_cells))


#%% Drugs we don't have moa for
drug_w_moa = drug_info.drop_duplicates("broad_id").dropna(subset=["moa"])
drug_w_moa = np.sort(np.array(drug_w_moa))
#%% Get drug ids in both info and response
response_drugs = [c.split("::")[0] for c in drug_response.columns[1:]]
info_drugs = drug_info["broad_id"].drop_duplicates()
ov_drugs = np.array(np.sort(set(response_drugs) & set(info_drugs)))

#%% Get dosages for drug response
drug_response_doses = [[np.nan] + [c.split("::")[1] for c in drug_response.columns[1:]]]
drug_response_doses = np.array(drug_response_doses).astype(float)

"""
We're gonna have two datasets:
one with only the cell lines that overlap between drug response, crispr, rna-seq
and one with all cell lines we have drug response for
"""
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

# %% all cell lines we have drug response f
uniq_drug_info = drug_info.drop_duplicates("broad_id")
ov_drug_info = uniq_drug_info[uniq_drug_info["broad_id"].isin(drugs_ov_drug)]
ov_drug_info = ov_drug_info.sort_values("broad_id")

#%% Remove drugs for which we have no MOA
ov_d_info_moa = ov_drug_info

#%% n hot encode the moa
def get_n_hot_moa(uniq_drug_info):
  uniq_moa = uniq_drug_info["moa"].str.split(",").dropna()
  uniq_moa = uniq_moa.apply(pd.Series).stack().reset_index(drop=True)
  uniq_moa = uniq_moa.apply(lambda s: s.strip()).unique()
  uniq_moa = np.sort(np.array(uniq_moa))
  n_uniq_moa = uniq_moa.shape[0]

  moa2ind = dict(zip(uniq_moa, np.arange(uniq_moa.shape[0])))
  n_hot_moa = np.zeros((uniq_moa.shape[0], n_uniq_moa))
  list_moas = list(uniq_drug_info["moa"].str.split(","))
  for i in range(n_hot_moa.shape[0]):
    new_row = np.zeros((1, n_uniq_moa))
    moas = [m.strip() for m in list_moas[i] ]
    one_inds = [moa2ind[m] for m in moas]
    new_row[0,one_inds] = 1
    n_hot_moa[i,:] = new_row

  return uniq_moa, n_hot_moa
uniq_moa, ov_moa = get_n_hot_moa(ov_drug_info)


# %% Function we will use to get both datasets once we sort and subset
def get_out_files(drug_response, drug_info, cell_lines, drugs):
  """
  drug response and info should already be subsetted to same order as 
  cell lines, drugs
  """
  drug_resp_np = drug_response.iloc[:,1:].to_numpy()

  return drug_resp_np






# %%
