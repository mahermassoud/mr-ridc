"""
Data loader for achilles dataset run this file and look at __main__ for 
example of how to use
"""
import torch
from os import path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pickle
from torch.utils.data._utils.collate import default_collate
import gc

"""
Specify files here
axis_val : dict( include_val:path_in_data_dir )
"""
DATASETS = {
  "cell_line": {
    "crispr": "imputed_crispr.npy", 
    "gene_expr": "norm_imputed_expression.npy", 
    #"pca_crispr": , # TODO
  },
  "drug": {
    "response": "drug/drug_only_overlap/ov_dose_resp.npy",
    #"moa": "" # TODO
  }
}
GENE_LIST_FN = "clean_gene_list.npy"
CELL_LINES_FN = "clean_cell_lines.npy"
to_tensor = transforms.ToTensor()

class AchillesDataset(Dataset):

  def __init__(
    self, 
    data_fp="/content/drive/My Drive/Mr RIDC/Data",
    axis="cell_line",
    datasets=["crispr"],
    drug_feature_fp=None, # TODO replace with drug_features enum
    drug_feature_cell_fp=None
  ):
    """
    data_fp: path to directory that holds all of the data_files.
             Default is for collab. If running locally, must change
    axis: what one item is, must be "cell_line" or "drug"
          eg. if "cell_line", __getitem__ returns the crispr, rna for 1 cell line
    datasets: list of data types to include in output, different possible values 
              for depending on axis. To reduce memory usage, only include the data
              types you will actually use
    drug_feature_fp: path to .npy file where rows are cell lines and columns are
                     features
    drug_feature_cell_fp: cell ids in the order of rows of drug_feature_fp
    """
    self.axis = axis
    self.datasets = datasets
    if axis not in DATASETS.keys():
      raise ValueError(f"axis must be one of {DATASETS.keys()}")
    if any([ds not in DATASETS[axis].keys() for ds in datasets]):
      raise ValueError(f"datasets param must be in {DATASETS[axis].keys()}")
    are_drug_features = drug_feature_fp is not None and drug_feature_cell_fp is not None
    self.are_drug_features = are_drug_features
    if axis != "drug" and are_drug_features:
      raise ValueError("Can't use drug features unless we are iterating over drugs")

    # Maps from dataset to a tensor where each row represents a data point
    self.ds2table = {}
    
    if axis == "cell_line":
      for ds in datasets:
        numpy_fp = path.join(data_fp, DATASETS[axis][ds])
        self.ds2table[ds] = torch.squeeze(to_tensor(np.load(numpy_fp)))
      if "crispr" in datasets and "gene_expr" in datasets:
        crispr = self.ds2table["crispr"]
        expr = self.ds2table["gene_expr"]
        combined = np.stack([crispr, expr]).transpose((1,0,2))
        self.datasets.append("crispr_gene_expr")
        self.ds2table["crispr_gene_expr"] = combined
      # Read in metadata
      self.genes = np.load(path.join(data_fp, GENE_LIST_FN), allow_pickle=True)
      self.cells = np.load(path.join(data_fp, CELL_LINES_FN), allow_pickle=True)
    elif axis == "drug":
      # Drug response table
      self.drug_response = torch.squeeze(to_tensor(np.load(
        path.join(data_fp, "drug/drug_only_overlap/ov_dose_resp.npy"))))
      self.resp_broad_ids = np.load(
        path.join(data_fp, "drug/drug_only_overlap/ov_resp_drug.npy"))

      # Drug info table
      self.drug2moa_inds = pickle.load(
        open(path.join(data_fp, "drug/drug_only_overlap/drug2moa_inds.pi"), "rb")
      )
      self.moa_broad_ids = np.load(
        path.join(data_fp, "drug/drug_only_overlap/moa_broad_ids.npy"),
        allow_pickle=True
      )
      self.moas = np.load(
        path.join(data_fp, "drug/drug_only_overlap/moas.npy"),
        allow_pickle=True
      )
      self.cells = np.load(
        path.join(data_fp, "drug/drug_only_overlap/ov_resp_cell.npy"),
        allow_pickle=True
      )
      self.ds2table["response"] = self.drug_response

      if self.are_drug_features:
        # Read in drug feature vector table
        self.drug_feature = np.load(drug_feature_fp)
        self.drug_feature_cells = np.load(drug_feature_cell_fp, allow_pickle=True)

        # Get overlapping cells, only keep drug responses, features for those
        ov_cells = np.array(list(set(self.cells) & set(self.drug_feature_cells)))
        response_keep_inds = np.where(np.isin(self.cells, ov_cells))[0]
        feature_keep_inds = np.where(np.isin(self.drug_feature_cells, ov_cells))[0]

        # drug response holds the dosage which doesn't map to a cell line
        # saving it for later
        dosage = self.drug_response[:, 0]
        self.drug_response = self.drug_response[:, response_keep_inds]
        self.drug_feature = self.drug_feature[feature_keep_inds, :]

        self.cells = self.cells[response_keep_inds]
        self.drug_feature_cells = self.drug_feature_cells[feature_keep_inds]

        # Make order of responses, features match
        feature_sort_inds = np.argsort(self.drug_feature_cells)
        response_sort_inds = np.argsort(self.cells)

        self.cells = self.cells[response_sort_inds]
        self.drug_response = self.drug_response[:,response_sort_inds]

        self.drug_feature_cells = self.drug_feature_cells[feature_sort_inds]
        self.drug_feature = self.drug_feature[feature_sort_inds, :]

        # concatenate dosage back into drug response
        self.drug_response = np.concatenate([dosage[:,np.newaxis], self.drug_response], axis=1)

        # Stack the features onto our drug response matrix
        # For now, replicating the dosage value accross all features
        n_drug = self.drug_response.shape[0]
        n_feat = self.drug_feature.shape[1]
        stack_features = np.tile(self.drug_feature, (n_drug, 1, 1))
        dosage = np.array(dosage)
        stack_dosage = np.tile(dosage[:,np.newaxis], (1,1,n_feat)).transpose((1,0,2))
        to_stack = np.concatenate((stack_dosage, stack_features), axis=1)
        self.drug_response = np.concatenate(
          (self.drug_response[:,:,np.newaxis], to_stack), axis=2)
        #dosage_stack = np.tile(dosage[:,np.newaxis])
        #stack_features = np.concate([np.zeros])
    

  def __len__(self):
    if self.axis == "cell_line":
      return self.ds2table[self.datasets[0]].shape[0]
    else:
      return self.drug_response.shape[0]

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    out_dict = {}
    if self.axis == "cell_line":
      for ds in self.datasets:
        tbl = self.ds2table[ds]
        if len(tbl.shape) == 2:
          out_dict[ds] = tbl[idx, :]
        elif len(tbl.shape) == 3:
          out_dict[ds] = tbl[idx, :, :]
    else:
      drug = self.resp_broad_ids[idx]
      moa_inds = self.drug2moa_inds[drug]
      if self.are_drug_features:
        response = self.drug_response[idx,:,:]
      else:
        response = self.drug_response[idx,:]
      out_dict = {
        "response": response,
        "drug": drug,
        "moa_ind": torch.Tensor(moa_inds),
      }

    return out_dict

if __name__ == "__main__":
  def drug_collate_fn(batch):
    moas = []
    for s in batch:
      moas.append(s["moa_ind"].int().tolist())
      del s["moa_ind"]
    out_dict = default_collate(batch)
    out_dict["moa_ind"] = moas
    return out_dict

  print("----------Iterating over cell lines-----------")
  DD = "/Users/massoudmaher/Documents/Code/mr-ridc/data"
  #DD = "/content/drive/My Drive/Mr RIDC/Data"
  cell_achilles = AchillesDataset(
    data_fp=DD,
    axis="cell_line",
    datasets=["crispr","gene_expr"]
   )

  cell_achilles_loader = DataLoader(cell_achilles, batch_size=4, shuffle=False)
  i = 0
  for batch in cell_achilles_loader:
    print("batch[\"crispr\"].shape")
    print(batch["crispr"].shape)
    print("batch[\"gene_expr\"].shape")
    print(batch["gene_expr"].shape)
    print("batch[\"crispr_gene_expr\"].shape")
    print(batch["crispr_gene_expr"].shape)

    if i == 2:
      break
    i += 1

  print("cell_achilles.genes holds gene names in correct order")
  print(cell_achilles.genes[:5])
  print("cell_achilles.cells holds cell line names in correct order")
  print(cell_achilles.cells[:5])

  print("\n\n----------Iterating over drugs----------")
  drug_achilles = AchillesDataset(
    data_fp=DD,
    axis="drug",
    datasets=["response"]
  )

  drug_loader = DataLoader(drug_achilles, batch_size=4, shuffle=True, 
                           collate_fn=drug_collate_fn)
  i = 0
  for batch in drug_loader:
    print("batch[\"response\"].shape")
    print(batch["response"].shape)
    print("First column is dosage!!!!")

    print("drug_achilles.cells holds cells in same order as batch[response] cols")
    print(drug_achilles.cells[:5])
    print("note that drug_achilles.cells.shape is 1 less since the first column is dosage")
    print(drug_achilles.cells.shape)

    print("batch[\"drug\"]")
    print(batch["drug"])

    print("batch[\"moa_ind\"]")
    moa_inds = batch["moa_ind"]
    print(moa_inds)

    print("corresponding moas")
    print([drug_achilles.moas[list(i)] for i in moa_inds])


    if i == 2:
      break
    i += 1
  
  print("\n\n----Iterating over drugs with feature vectors for each cell line CRISPR ONLY-----")
  print("The data was already subsetted by only taking cell lines and genes that overlap between crispr and RNA-seq")
  feat_drug_achilles = AchillesDataset(
    data_fp=DD,
    axis="drug",
    datasets=["response"],
    drug_feature_fp=path.join(DD, "pca50_imputed_crispr.npy"),
    drug_feature_cell_fp=path.join(DD, "clean_cell_lines.npy")
  )
  feat_drug_loader = DataLoader(feat_drug_achilles, batch_size=4, shuffle=True, 
                                collate_fn=drug_collate_fn)
  i = 0
  for batch in feat_drug_loader:
    print("batch[\"response\"].shape")
    print(batch["response"].shape)
    print("First column is dosage!!!!")

    if i == 2:
      break
    i += 1

  # Garbage collect so we don't overload memory
  feat_drug_achilles, feat_drug_loader = None, None
  gc.collect()
  print("\n\n----Iterating over drugs with feature vectors for each cell line EXPRESSION ONLY-----")
  print("The data was already subsetted by only taking cell lines and genes that overlap between crispr and RNA-seq")
  feat_drug_achilles = AchillesDataset(
    data_fp=DD,
    axis="drug",
    datasets=["response"],
    drug_feature_fp=path.join(DD, "pca50_imp_expr.npy"),
    drug_feature_cell_fp=path.join(DD, "clean_cell_lines.npy")
  )
  feat_drug_loader = DataLoader(feat_drug_achilles, batch_size=4, shuffle=True, 
                                collate_fn=drug_collate_fn)
  i = 0
  for batch in feat_drug_loader:
    print("batch[\"response\"].shape")
    print(batch["response"].shape)
    print("First column is dosage!!!!")

    if i == 2:
      break
    i += 1

  # Garbage collect so we don't overload memory
  feat_drug_achilles, feat_drug_loader = None, None
  gc.collect()
  print("\n\n----Iterating over drugs with feature vectors for each cell line CRISPR AND EXPRESSION-----")
  print("The data was already subsetted by only taking cell lines and genes that overlap between crispr and RNA-seq")
  feat_drug_achilles = AchillesDataset(
    data_fp=DD,
    axis="drug",
    datasets=["response"],
    drug_feature_fp=path.join(DD, "pca50_imp_crispr_expr.npy"),
    drug_feature_cell_fp=path.join(DD, "clean_cell_lines.npy")
  )
  feat_drug_loader = DataLoader(feat_drug_achilles, batch_size=4, shuffle=True, 
                                collate_fn=drug_collate_fn)
  i = 0
  for batch in feat_drug_loader:
    print("batch[\"response\"].shape")
    print(batch["response"].shape)
    print("First column is dosage!!!!")

    if i == 2:
      break
    i += 1



