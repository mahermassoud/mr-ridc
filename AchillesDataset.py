import torch
from os import path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Change this to path to data directory 
DD = "/Users/massoudmaher/Documents/Code/mr-ridc/data"

class AchillesDataset(Dataset):
  
  def __init__(self):

  def __len__(self):

    pass


  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.annot.iloc[idx, 0]

    class_id = self.annot.iloc[idx, 1]
    species = self.annot.iloc[idx, 2]
    breed_id = self.annot.iloc[idx, 3]

    dict_out = {
      "img_name": img_name,
      "class_id": class_id,
      #"class_vec": class_vec,
      "species": species,
      "breed_id": breed_id,
    }
    if self.load_feature:
      feat_fp = path.join(FEAT_FP, img_name + ".pt")
      feat = torch.load(feat_fp)
      if self.norm_feature:
        feat = NORM_TRANSFORM(feat)
      dict_out["feat"] = feat
    if self.load_image:
      img_fp = path.join(ALL_IMG_FP, img_name + ".jpg")
      img = Image.open(img_fp).convert("RGB")
      if self.transform is not None:
        img = self.transform(img)
      dict_out["t_img"] = img

    return dict_out

  def __one_hot_class(self, cid):
    one_hot = np.zeros(self.n_label)
    one_hot[cid - 1] = 1
    return torch.Tensor(one_hot)

if __name__ == "__main__":
  op_train = OxfordPetsDataset()
  op_test = OxfordPetsDataset(False)

  train_loader = DataLoader(op_train, batch_size=4,
                            shuffle=False, num_workers=0)
  test_loader = DataLoader(op_test, batch_size=4,
                            shuffle=False, num_workers=0)

  i = 0
  for bi, d in enumerate(train_loader):
    print(d["img_name"])
    print(d["t_img"].shape)

    if i >= 2:
      break
    i += 1
 
  print("test-------")
  for bi, d in enumerate(test_loader):
    print(d["img_name"])

    if i >= 2:
      break
    i += 1

