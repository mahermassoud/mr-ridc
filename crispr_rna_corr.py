# Script to check correlation between CRISPR and RNA-seq
#%% Imports
import pandas as pd
import seaborn as sns
from os import path

#%% Constants
DD = "/Users/massoudmaher/Documents/Code/mr-ridc/data"
M_CRISPR_FP = path.join(DD, "melt_crispr.csv")
RNA_FP = path.join(DD, "CCLE_RNAseq_reads.csv")

#%% Read in CRISPR
m_crispr = pd.read_csv(M_CRISPR_FP)
m_crispr.head()

# %% Read in gene expression
rna = pd.read_csv(RNA_FP)
rna = rna.rename(columns={"Unnamed: 0": "broad_id"})

# %% Check cell_id overlaps
crispr_lines = set(m_crispr["broad_id"])
rna_lines = set(rna["broad_id"])
print("len(crispr_lines)")
print(len(crispr_lines))
print("len(rna_lines)")
print(len(rna_lines))
print("len(rna_lines.intersection(crispr_lines))")
print(len(rna_lines.intersection(crispr_lines)))
# There are 19 / 739 cell lines we have crispr for but not RNA-seq

# %% Process rna seq
m_rna = rna.melt(id_vars="broad_id", var_name="gene", value_name="expression")
m_rna["gene_hgnc"] = m_rna["gene"].str.split(" ", expand=True).iloc[:,0]
m_rna["gene_ensembl"] = (
    m_rna["gene"].str.split("(", expand=True).iloc[:,1].str.replace(")", ""))
m_rna.head()

# %%
