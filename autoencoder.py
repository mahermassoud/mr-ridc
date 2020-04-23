from torch import nn
import torch.nn.functional as F

# Define networks
class Encoder(nn.Module):
  def __init__(self, n_gene=18293, encode_dim=1000):
    super(Encoder, self).__init__()
    # Kernel size is key. We want to take crispr/expression pairs
    # We do NOT want to assume dependency between different genes
    self.conv1 = nn.Conv1d(2, 10, 1)
    self.conv2 = nn.Conv1d(10,1,1)
    self.linear1 = nn.Linear(n_gene, encode_dim)

  def forward(self,x):
    # I prefer doing it this way over sequential because you can
    # add print statements to see the shape at each stage
    #print("x0.shape")
    #print(x.shape)
    x = self.conv1(x)
    x = F.relu(x)

    x = self.conv2(x)
    x = F.relu(x)

    x = self.linear1(x)

    return x

class Decoder(nn.Module):
  def __init__(self, n_gene=18293, encode_dim=1000):
    super(Decoder, self).__init__()
    self.linear1 = nn.Linear(encode_dim, n_gene)
    self.upconv1 = nn.ConvTranspose1d(1, 10, 1)
    self.upconv2 = nn.ConvTranspose1d(10, 2, 1)

  def forward(self,x):
    x = self.linear1(x)

    x = self.upconv1(x)
    x = F.relu(x)

    x = self.upconv2(x)
    x = F.relu(x)

    return x
