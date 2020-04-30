#%% Imports
from os import path
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

#%% Load in data
data_fp = "/Users/massoudmaher/Documents/Code/mr-ridc/data"
data_fn = "norm_imputed_expression.npy"

data = np.load(path.join(data_fp, data_fn))

#%% PCA
pca = PCA()
pca_data = pca.fit_transform(data)
print("pca_data.shape")
print(pca_data.shape)

# %%
def get_cum_var(pca_obj):
  nc = pca.explained_variance_ratio_.shape[0]
  return [np.sum(pca.explained_variance_ratio_[:i]) 
          for i in range(0, nc)]

cum_var = get_cum_var(pca)
# %%
def plot_variance_explained(cumm_variance_explained, pt_title, 
                            out_fp=None):
  plt.figure(figsize=(8,8))
  plt.plot(cumm_variance_explained)
  plt.title("Cummulative Variance: " + pt_title + " Explained By Number of PCs")
  plt.xlabel("Number of Principal Components")
  plt.ylabel("Fraction of Total Variance Explained")
  plt.xticks(np.arange(0,len(cumm_variance_explained),50))
  plt.grid()
  if out_fp is not None:
    plt.savefig(out_fp)

#plot_variance_explained(cum_var, "CRISPR andGene expression", 
#                        out_fp=path.join(data_fp, "plots/imputed_crispr_and_expr_pca.jpg"))
# %%
reduced_data = pca_data[:,:50]
out_fp = path.join(data_fp, "pca50_imp_expr.npy")
np.save(out_fp, reduced_data)

# %%
