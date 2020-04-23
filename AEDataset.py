# Dataset Implementation
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class AEDataset(Dataset):
  #'Characterizes a dataset for PyTorch'
  def __init__(self, x, y):
    #'Initialization'
    self.x = x
    self.y = y
    self.transform = transforms.Compose([transforms.ToTensor()])
  def __len__(self):
    #'Denotes the total number of samples'
    return len(self.x)

  def __getitem__(self, index):
    # Select sample and get label
    x = self.x[index]
    y = self.y[index]
    return self.transform(x), self.transform(y)
