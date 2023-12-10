import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import muon
import scanpy as sc
import anndata
import pandas as pd
import h5py
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
import argparse
import time
import os

from torch.utils.data import Dataset, DataLoader
from snapccess.model import snapshotVAE, Encoder
from snapccess.train import train_model
from snapccess.util import setup_seed


parser = argparse.ArgumentParser("SnapCCESS")
parser.add_argument('--path1', metavar='DIR',  default=None, help='path to train data1')
parser.add_argument('--path2', metavar='DIR',  default=None, help='path to train data2')
parser.add_argument('--path3', metavar='DIR',  default=None, help='path to train data3')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
args = parser.parse_args()

begin_time = time.time()
    
def data_loader(path):
    print(path)
    with h5py.File(path, "r") as f:
        X = np.mat(np.array(f['matrix/data']).transpose())
    return X
    
    
def run_snapccess(file_paths):
    rna_path=file_paths['rna_path']
    adt_path=file_paths['adt_path']
    atac_path=file_paths['atac_path']
    
    # read rna
    if rna_path is not None:
        rna = data_loader(rna_path)
        nfeatures_rna = rna.shape[1]
        print(pd.DataFrame(rna).std().min())
        rna_sample_scaled=(pd.DataFrame(rna)-pd.DataFrame(rna).mean())/(pd.DataFrame(rna).std()+0.00001)
    if adt_path is not None:
        pro = data_loader(adt_path)
        nfeatures_pro = pro.shape[1]
        pro_sample_scaled=(pd.DataFrame(pro)-pd.DataFrame(pro).mean())/pd.DataFrame(pro).std()
    if atac_path is not None:
        atac = data_loader(atac_path)
        nfeatures_atac = atac.shape[1]
        atac_sample_scaled=(pd.DataFrame(atac)-pd.DataFrame(atac).mean())/pd.DataFrame(atac).std()
        
    
    ## parameters
    batch_size = args.batch_size
    epochs_per_cycle = 2
    epochs = epochs_per_cycle*10
    lr = 0.02
    z_dim = 100
    hidden_rna2 = 185
    hidden_pro2 = 30
    hidden_atac2 = 185
    
    if (rna_path is not None) and (adt_path is not None) and (atac_path is None):
        feature_num = nfeatures_rna + nfeatures_pro
        citeseq = pd.concat([rna_sample_scaled, pro_sample_scaled], axis=1)
        train_data=citeseq.to_numpy(dtype=np.float32)
        train_transformed_dataset = (train_data)
        train_dl = DataLoader(train_transformed_dataset, batch_size=batch_size,shuffle=False, num_workers=0,drop_last=False)
        test_transformed_dataset = (train_data)
        valid_dl = DataLoader(test_transformed_dataset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last=False)
        model = snapshotVAE(num_features=[nfeatures_rna,nfeatures_pro], num_hidden_features=[hidden_rna2,hidden_pro2], z_dim=z_dim).cuda()
        model,histroy,embedding = train_model(model, train_dl, valid_dl, lr=lr, epochs=epochs,epochs_per_cycle=epochs_per_cycle, save_path="",snapshot=True,embedding_number=1)
    
    if (rna_path is not None) and (adt_path is None) and (atac_path is not None):
        feature_num = nfeatures_rna + nfeatures_atac
        shareseq = pd.concat([rna_sample_scaled, atac_sample_scaled], axis=1)
        train_data=shareseq.to_numpy(dtype=np.float32)
        train_transformed_dataset = (train_data)
        train_dl = DataLoader(train_transformed_dataset, batch_size=batch_size,shuffle=False, num_workers=0,drop_last=False)
        test_transformed_dataset = (train_data)
        valid_dl = DataLoader(test_transformed_dataset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last=False)
        model = snapshotVAE(num_features=[nfeatures_rna,nfeatures_atac], num_hidden_features=[hidden_rna2,hidden_atac2], z_dim=z_dim).cuda()
        model,histroy,embedding = train_model(model, train_dl, valid_dl, lr=lr, epochs=epochs,epochs_per_cycle=epochs_per_cycle, save_path="",snapshot=True,embedding_number=1)
        
    if (rna_path is not None) and (adt_path is not None) and (atac_path is not None):
        feature_num = nfeatures_rna + nfeatures_pro + nfeatures_atac
        teaseq = pd.concat([rna_sample_scaled, pro_sample_scaled, atac_sample_scaled], axis=1)
        train_data=teaseq.to_numpy(dtype=np.float32)
        train_transformed_dataset = (train_data)
        train_dl = DataLoader(train_transformed_dataset, batch_size=batch_size,shuffle=False, num_workers=0,drop_last=False)
        test_transformed_dataset = (train_data)
        valid_dl = DataLoader(test_transformed_dataset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last=False)
        model = snapshotVAE(num_features=[nfeatures_rna,nfeatures_pro,nfeatures_atac], num_hidden_features=[hidden_rna2,hidden_pro2,hidden_atac2], z_dim=z_dim).cuda()
        model,histroy,embedding = train_model(model, train_dl, valid_dl, lr=lr, epochs=epochs,epochs_per_cycle=epochs_per_cycle, save_path="",snapshot=True,embedding_number=1)

    return embedding



file_paths = {
    "rna_path": args.path1,
    "adt_path": args.path2,
    "atac_path": args.path3
}

result = run_snapccess(file_paths)

end_time = time.time()
all_time = end_time - begin_time
print(all_time)
print(result[0].shape)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")


file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result[0])
file.close()

np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")
