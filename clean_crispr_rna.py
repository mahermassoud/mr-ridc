#%% Imports
import re
import pandas as pd
from os import path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#%% Read in data
data_fp =  "/Users/massoudmaher/Documents/Code/mr-ridc/data"

gene_expression = pd.read_csv(path.join(data_fp, "CCLE_expression_full.csv"))
gene_effect = pd.read_csv(path.join(data_fp, "Achilles_gene_effect.csv"))

#%% Rename Columns
gene_expression_s = gene_expression.rename(columns={"Unnamed: 0": "broad_id"})
gene_effect_s = gene_effect.rename(columns = {"Unnamed: 0": "broad_id"})

#%% Sort columns (genes) alphabetically
gene_effect_s = gene_effect_s.reindex(sorted(gene_effect_s.columns), axis=1)
gene_expression_s = gene_expression_s.reindex(sorted(gene_expression_s.columns), axis=1)

#%% Delete stuff in parentheses from gene names to match to other data
gene_effect_rn = gene_effect_s.rename(columns=lambda x: re.sub('\W+\(.*','',x))
gene_expression_rn = gene_expression_s.rename(columns=lambda x: re.sub('\W+\(.*','',x))

#%% Get Duplicate Columns - gets 2nd+ instance of these
dup_cols = gene_expression_rn.loc[:,gene_expression_rn.columns.duplicated()].columns

#%% Now get them out of the set
gene_expression_unique_cols = [gene 
                               for gene in gene_expression_rn.columns.array
                               if re.sub('\W+\(.*','',gene) not in dup_cols]
gene_effect_unique_cols = [gene for gene in gene_effect_rn.columns.array 
                           if re.sub('\W+\(.*','',gene) not in dup_cols]

#%% Get intersection of genes
shared_genes = list(set(gene_expression_unique_cols) & 
                    set(gene_effect_unique_cols))
expression_final_g = gene_expression_rn[shared_genes]
crispr_final_g = gene_effect_rn[shared_genes]

#%% Now match the cell lines
cl_exp = expression_final_g['broad_id']
cl_crispr = crispr_final_g['broad_id']
shared_cl = list(set(cl_exp) & set(cl_crispr))

#%% Subset each dataset so that they have the same cell lines
expression_final = expression_final_g.loc[expression_final_g['broad_id'].isin(shared_cl)]
expression_final = expression_final.sort_values(by=['broad_id'])
crispr_final = crispr_final_g.loc[crispr_final_g['broad_id'].isin(shared_cl)]
crispr_final = crispr_final.sort_values(by=['broad_id'])

cell_lines = crispr_final["broad_id"]

expression_final = expression_final.drop('broad_id', 1)
crispr_final = crispr_final.drop('broad_id', 1)

#%% Convert to Numpy
expression_np = expression_final.values
crispr_np = crispr_final.values

#%% Replace Nans with mean of that column (column=gene)
exp_impute = SimpleImputer(strategy="mean").fit(expression_np)
crispr_impute = SimpleImputer(strategy="mean").fit(crispr_np)
imp_expression_np = exp_impute.transform(expression_np)
imp_crispr_np = exp_impute.transform(crispr_np)
#np.nan_to_num(expression_np, copy=False)
#np.nan_to_num(crispr_np, copy=False)

#%% Z normalization
#norm_expression = z_normalize(expression_np)
exp_normalizer = StandardScaler().fit(imp_expression_np)
z_imp_expression_mp = exp_normalizer.transform(imp_expression_np)
# We do not need to normalize CRISPR since it has been cleaned
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5709193/
#norm_crispr = z_normalize(crispr_np)

#%% Combine the sets and set type to float
stack_expression = np.expand_dims(z_imp_expression_mp, axis=2).astype(float)
stack_crispr = np.expand_dims(imp_crispr_np, axis=2).astype(float)

#%% Save files for later
#np.save(path.join(data_fp, "norm_imputed_expression.npy"), z_imp_expression_mp)
#np.save(path.join(data_fp, "imputed_crispr.npy"), crispr_np)

#%% Prepare metadata
cell_lines_np = cell_lines.to_numpy()
gene_list_np = expression_final.columns.to_numpy()

#%% Save metadata
np.save(path.join(data_fp, "clean_gene_list.npy"), gene_list_np)
np.save(path.join(data_fp, "clean_cell_lines.npy"), cell_lines_np)
#%%
#combined = np.concatenate( (norm_expression,norm_crispr),axis=2) 
#combined = combined.astype(float)

# save files for easy loading later
#np.save("/content/drive/My Drive/Mr RIDC/Data/norm_expression", norm_expression)
#np.save("/content/drive/My Drive/Mr RIDC/Data/norm_crispr", norm_crispr)
#np.save("/content/drive/My Drive/Mr RIDC/Data/combined", combined)
