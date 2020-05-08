#%% Imports
import pandas as pd
from os import path
import numpy as np
import pickle

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
resp_cols = [None] + [c.split("::")[0] for c in drug_response.columns[1:]]
keep_cols = np.where(np.array([rc in ov_drugs for rc in resp_cols]))[0]
keep_cols = np.concatenate([[0], keep_cols])
ov_response = drug_response.iloc[:,list(keep_cols)]
#resp_cols = np.where(np.isin(response_drugs, ov_drugs)) + 1
#ov_response = drug_response[[0]+resp_cols]

#%% Drug response one where the cell lines are in crispr+rna
feat_ov_response = ov_response[ov_response["cell_line"].isin(other_cell_lines)]

#%% Make order of drugs match between info and response
uniq_ov_info = ov_info.drop_duplicates("broad_id").sort_values("broad_id")
#feat_ov_response = feat_ov_response.sort_values("broad_id")
#ov_response = ov_response.sort_values("broad_id")

#%% Drop duplicates from drug info and N-hot encode MOAs
def get_n_hot_moa(uniq_drug_info):
  uniq_moa_list = uniq_drug_info["moa"].str.split(",")
  uniq_moa = uniq_moa_list.apply(pd.Series).stack().reset_index(drop=True)
  uniq_moa = uniq_moa.apply(lambda s: s.strip()).unique()
  uniq_moa = np.sort(np.array(uniq_moa))
  n_uniq_moa = uniq_moa.shape[0]

  moa2ind = dict(zip(uniq_moa, np.arange(uniq_moa.shape[0])))
  n_hot_moa = np.zeros((uniq_drug_info.shape[0], n_uniq_moa))
  list_moas = list(uniq_moa_list)
  drug2moa_n_hot, drug2moa_inds = {}, {}
  for i in range(n_hot_moa.shape[0]):
    new_row = np.zeros((1, n_uniq_moa))
    moas = [m.strip() for m in list_moas[i] ]
    one_inds = [moa2ind[m] for m in moas]
    new_row[0,one_inds] = 1
    n_hot_moa[i,:] = new_row
    
    drug = uniq_drug_info["broad_id"].iloc[i]
    drug2moa_inds[drug] = one_inds
    drug2moa_n_hot[drug] = new_row

  out = [uniq_drug_info["broad_id"].to_numpy(), uniq_moa, n_hot_moa, 
         drug2moa_inds, drug2moa_n_hot]
  return out

broad_id_np, uniq_moa_np, n_hot_moa, drug2moa_inds, drug2moa_n_hot = (
  get_n_hot_moa(uniq_ov_info))

# %% Stack drug dosage onto drug_response
def stack_response_dosage(drug_response):
  drugs = [c.split("::")[0] for c in drug_response.columns[1:]]
  drugs = np.array(drugs)
  dosages = [c.split("::")[1] for c in drug_response.columns[1:]]
  dosages = np.array(dosages).astype(float)

  resp_mat_np = drug_response.iloc[:,1:].to_numpy()
  resp_mat_np = np.transpose(resp_mat_np)
  resp_dosage_np = np.concatenate([dosages[:,np.newaxis], resp_mat_np], axis=1)

  return drug_response["cell_line"].to_numpy(), drugs, resp_dosage_np

ov_resp_cell, ov_resp_drug, ov_dose_resp = stack_response_dosage(ov_response)

#%% Normalize dosages and drug responses
#exp_normalizer = StandardScaler().fit(imp_expression_np)
#z_imp_expression_mp = exp_normalizer.transform(imp_expression_np)
from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler().fit(ov_dose_resp)
norm_ov_dose_resp = normalizer.transform(ov_dose_resp)

from sklearn.preprocessing import MinMaxScaler
min_maxer = MinMaxScaler().fit(ov_dose_resp)
mm_ov_dose_resp = min_maxer.transform(ov_dose_resp)

# %% Save files
np.save(path.join(data_fp, "drug/drug_only_overlap/moa_broad_ids.npy"), broad_id_np)
np.save(path.join(data_fp, "drug/drug_only_overlap/moas.npy"), uniq_moa_np)
np.save(path.join(data_fp, "drug/drug_only_overlap/n_hot_moa.npy"), broad_id_np)
pickle.dump(drug2moa_inds, 
            open(path.join(data_fp, "drug/drug_only_overlap/drug2moa_inds.pi"), "wb"))
pickle.dump(drug2moa_n_hot, 
            open(path.join(data_fp, "drug/drug_only_overlap/drug2moa_n_hot.pi"), "wb"))

np.save(path.join(data_fp, "drug/drug_only_overlap/ov_resp_cell.npy"), ov_resp_cell)
np.save(path.join(data_fp, "drug/drug_only_overlap/ov_resp_drug.npy"), ov_resp_drug)
np.save(path.join(data_fp, "drug/drug_only_overlap/ov_dose_resp.npy"), ov_dose_resp)
np.save(path.join(data_fp, "drug/drug_only_overlap/norm_ov_dose_resp.npy"), norm_ov_dose_resp)
np.save(path.join(data_fp, "drug/drug_only_overlap/mm_ov_dose_resp.npy"), mm_ov_dose_resp)

# %%
