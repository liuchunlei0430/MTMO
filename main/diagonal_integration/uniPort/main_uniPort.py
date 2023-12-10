import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import csv
import gzip
import scipy.io
import argparse
import time

import uniport as up
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix
###
parser = argparse.ArgumentParser("Portal")
parser.add_argument('--data_path1', metavar='DIR', default='NULL', help='path to train data1')
parser.add_argument('--data_path2', metavar='DIR', default='NULL', help='path to train data2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()
begin_time = time.time()

def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.mat(np.array(f['matrix/data']).transpose())
    adata = AnnData(X=csr_matrix(X))
    return adata
    
adata_rna = data_loader(args.data_path1)
adata_atac = data_loader(args.data_path2)

adata_rna.obs['domain_id'] = 1
adata_rna.obs['domain_id'] = adata_rna.obs['domain_id'].astype('category')
adata_rna.obs['source'] = 'RNA'

adata_atac.obs['domain_id'] = 0
adata_atac.obs['domain_id'] = adata_atac.obs['domain_id'].astype('category')
adata_atac.obs['source'] = 'ATAC'

adata_cm = adata_atac.concatenate(adata_rna, join='inner', batch_key='domain_id')

sc.pp.normalize_total(adata_cm)
sc.pp.log1p(adata_cm)
sc.pp.highly_variable_genes(adata_cm, n_top_genes=2000, inplace=False, subset=True)
up.batch_scale(adata_cm)

sc.pp.normalize_total(adata_rna)
sc.pp.log1p(adata_rna)
sc.pp.highly_variable_genes(adata_rna, n_top_genes=2000, inplace=False, subset=True)
up.batch_scale(adata_rna)

sc.pp.normalize_total(adata_atac)
sc.pp.log1p(adata_atac)
sc.pp.highly_variable_genes(adata_atac, n_top_genes=2000, inplace=False, subset=True)
up.batch_scale(adata_atac)

adata = up.Run(adatas=[adata_atac,adata_rna], adata_cm=adata_cm, lambda_s=1.0)

embedding = adata.obsm['latent']

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")

print(embedding.shape)

file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=(embedding))
file.close()







