library(tidyverse)
library(data.table)

#%% Constants
DD <- "/Users/massoudmaher/Documents/Code/mr-ridc/data"
RNA_FP <- file.path(DD, "CCLE_expression_full.csv")

rna <- fread(RNA_FP)
names(rna)[1] <- "broad_id"
names(rna)[2:ncol(rna)] <- str_split(names(rna)[2:ncol(rna)], " \\(") %>% 
  map(1) %>% unlist()
m.rna <- melt(rna, id.vars="broad_id", variable.name="gene_hugo", 
              value.name="expression")

m_crispr <- fread(file.path(DD, "melt_crispr.csv"))
m_crispr <- m_crispr %>% dplyr::select(broad_id, ceres, gene_hugo)

crispr_rna <- merge(m_crispr, m.rna)

(ggplot(crispr_rna) + geom_bin2d(aes(x=ceres, y=expression)) + 
  scale_fill_continuous(type="viridis"))