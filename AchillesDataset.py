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
    "response": "secondary-screen-cell-line-info.csv",
    "info": "secondary-screen-replicate-treatment-info.csv",
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
    datasets=["crispr"]
    # TODO train test split?
  ):
    """
    data_fp: path to directory that holds all of the data_files.
             Default is for collab. If running locally, must change
    axis: what one item is, must be "cell_line" or "drug"
          eg. if "cell_line", __getitem__ returns the crispr, rna for 1 cell line
    datasets: list of data types to include in output, different possible values 
              for depending on axis. To reduce memory usage, only include the data
              types you will actually use
    """
    self.axis = axis
    self.datasets = datasets
    if axis not in DATASETS.keys():
      raise ValueError(f"axis must be one of {DATASETS.keys()}")
    if any([ds not in DATASETS[axis].keys() for ds in datasets]):
      raise ValueError(f"datasets param must be in {DATASETS[axis].keys()}")

    # Maps from dataset to a tensor where each row represents a data point
    self.ds2table = {}
    for ds in datasets:
      numpy_fp = path.join(data_fp, DATASETS[axis][ds])
      self.ds2table[ds] = torch.squeeze(to_tensor(np.load(numpy_fp)))
    
    if axis == "cell_line" and "crispr" in datasets and "gene_expr" in datasets:
      crispr = self.ds2table["crispr"]
      expr = self.ds2table["gene_expr"]
      combined = np.stack([crispr, expr]).transpose((1,0,2))
      self.datasets.append("crispr_gene_expr")
      self.ds2table["crispr_gene_expr"] = combined
    
    # Read in metadata
    self.genes = np.load(path.join(data_fp, GENE_LIST_FN), allow_pickle=True)
    self.cells = np.load(path.join(data_fp, CELL_LINES_FN), allow_pickle=True)

  def __len__(self):
    return self.ds2table[self.datasets[0]].shape[0]

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    out_dict = {}
    for ds in self.datasets:
      tbl = self.ds2table[ds]
      if len(tbl.shape) == 2:
        out_dict[ds] = tbl[idx, :]
      elif len(tbl.shape) == 3:
        out_dict[ds] = tbl[idx, :, :]

    return out_dict
      

if __name__ == "__main__":
  print("Iterating over cell lines")
  cell_achilles = AchillesDataset(
    data_fp="/Users/massoudmaher/Documents/Code/mr-ridc/data",
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
