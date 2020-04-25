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

#%% Drugs missing MOA
drug_w_moa = drug_info.drop_duplicates("broad_id").dropna(subset=["moa"])["broad_id"]
drugs_w_moa = np.sort(np.array(drug_w_moa))

info_w_moa = drug_info[drug_info["broad_id"].isin(drugs_w_moa)]

#%% Drugs present in both drug response, info
response_drugs = [c.split("::")[0] for c in drug_response.columns[1:]]
info_drugs = info_w_moa["broad_id"].drop_duplicates()
ov_drugs = np.sort(list(set(response_drugs) & set(info_drugs)))

ov_info = info_w_moa[info_w_moa["broad_id"].isin(ov_drugs)]
ov_response = drug_response # They actually completely overlap
#resp_cols = np.where(np.isin(response_drugs, ov_drugs)) + 1
#ov_response = drug_response[[0]+resp_cols]

#%% Drug response one where the cell lines are in crispr+rna
feat_ov_response = ov_response[ov_response["cell_line"].isin(other_cell_lines)]

#%% Drop duplicates from drug info and N-hot encode MOAs
def get_n_hot_moa(uniq_drug_info):
  uniq_moa_list = uniq_drug_info["moa"].str.split(",")
  uniq_moa = uniq_moa_list.apply(pd.Series).stack().reset_index(drop=True)
  uniq_moa = uniq_moa.apply(lambda s: s.strip()).unique()
  uniq_moa = np.sort(np.array(uniq_moa))
  n_uniq_moa = uniq_moa.shape[0]

  moa2ind = dict(zip(uniq_moa, np.arange(uniq_moa.shape[0])))
  n_hot_moa = np.zeros((uniq_moa.shape[0], n_uniq_moa))
  list_moas = list(uniq_moa_list)
  for i in range(n_hot_moa.shape[0]):
    new_row = np.zeros((1, n_uniq_moa))
    moas = [m.strip() for m in list_moas[i] ]
    one_inds = [moa2ind[m] for m in moas]
    new_row[0,one_inds] = 1
    n_hot_moa[i,:] = new_row

  return uniq_drug_info["broad_id"].to_numpy(), uniq_moa, n_hot_moa

uniq_ov_info = ov_info.drop_duplicates("broad_id")
broad_id_np, uniq_moa_np, n_hot_moa = get_n_hot_moa(uniq_ov_info)


# %%
