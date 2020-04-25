require(DESeq)
library(tidyverse)
library(data.table)

data.fp <- "/Users/massoudmaher/Documents/Code/mr-ridc/data"
rna.fp <- file.path(data.fp, "CCLE_expression_full.csv")
rna <- fread(rna.fp)

cds <- newCountDataSet(rnaseq.matrix.raw, rnaseq.meta.raw)
cds <- estimateSizeFactors(cds)
cds <- estimateDispersions(cds, method="blind")
vsd <- varianceStabilizingTransformation(cds)
dat <- exprs(vsd)