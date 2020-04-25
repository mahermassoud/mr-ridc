#%% Imports -------------------------------
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
from os import path
import numpy as np
from AEDataset import *
from autoencoder import *
from time import time

#%% Params -----------------------------------
# Path to repo directory
RD = "/Users/massoudmaher/Documents/Code/mr-ridc/"
DD = path.join(RD, "data")

# Subsets data so its easier to debug, set to None to use whole dataset
MAX_ITEM = None
MAX_EPOCH = 10
LEARN_RATE = 0.0001
MOMENTUM = 0.3
BATCH_SZ = 10
N_WORKER = 0 # Set higher if on collab? this never improves my runtime --M


#%% Load in data -------------------------------
# cell-line x gene x 2 last dim is order: expression, crispr
cell_gene_channel = np.load(path.join(DD, "combined.npy"))
print("cell_gene_channel.shape")
print(cell_gene_channel.shape)

# Subset our data loader if needed
if MAX_ITEM is not None:
  sampler = data.SubsetRandomSampler(np.arange(MAX_ITEM))
  shuffle = False # Shuffle false or else we get different data each epoch..
else:
  sampler = None
  shuffle = True
dataset = data.TensorDataset(torch.Tensor(np.transpose(
  cell_gene_channel, (0,2,1) # So dataset is cell-line x 2 x gene
)))
params = {
  'batch_size': BATCH_SZ,
  'shuffle': shuffle,
  'num_workers': N_WORKER,
  'sampler': sampler 
}
train_loader = data.DataLoader(dataset, **params)

#%% Create nets, optimizers, losses
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
decoder = Decoder().to(device)

loss_fn = nn.MSELoss()
params = list(encoder.parameters()) + list(decoder.parameters())
print("num params")
print(len(params))
optimizer = torch.optim.SGD(params, lr=LEARN_RATE, momentum=MOMENTUM)

encoder.train()
decoder.train()
#%% Training loop ---------------------------------
losses = np.zeros(MAX_EPOCH)
for epoch in range(MAX_EPOCH):
  start_time = time()
  for batch in train_loader:
    optimizer.zero_grad()
    batch = batch[0].to(device)

    encoded = encoder(batch)
    decoded = decoder(encoded)
    loss = loss_fn(decoded, batch)
    loss.backward()
    optimizer.step()

  losses[epoch] = loss.item()
  print(f"epoch {epoch}, {time()-start_time}s loss {loss.item()}")
