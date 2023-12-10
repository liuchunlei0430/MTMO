import argparse
import sys, os

import numpy as np
from umap import UMAP
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd  
import scipy.sparse as sp
import h5py

import scmomat 
import scipy



parser = argparse.ArgumentParser("scMoMaT")
parser.add_argument('--path1', metavar='DIR', nargs='+', default=[], help='path to train data1')
parser.add_argument('--path2', metavar='DIR', nargs='+', default=[], help='path to train data2')
parser.add_argument('--path3', metavar='DIR', nargs='+', default=[], help='path to train data3')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

begin_time = time.time()


def h5_to_matrix(path):
    data = h5py.File(path,"r")
    h5_data = data['matrix/data']
    sparse_data = scipy.sparse.csr_matrix(np.array(h5_data).transpose())
    data = np.array(sparse_data.todense())
    return data
    
    
def read_h5_data(rna_path=None, adt_path=None, atac_path=None, list_len=3):
    rna_list = []
    adt_list = []
    atac_list = []
    rna_feature_num = "None"
    adt_feature_num = "None"
    atac_feature_num = "None"
    
    # read rna
    if rna_path != "None":
        for i in range(len(rna_path)):
            if rna_path[i] == "None":
                rna_list.append(None)
            else:
                rna_feature_num = h5_to_matrix(rna_path[i]).shape[1]
                rna_list.append(scmomat.preprocess((h5_to_matrix(rna_path[i])), modality = "RNA"))
    
    # read adt
    if adt_path != "None":
        for i in range(len(adt_path)):
            if adt_path[i] == "None":
                adt_list.append(None)
            else:
                adt_feature_num = h5_to_matrix(adt_path[i]).shape[1]
                adt_list.append(scmomat.preprocess((h5_to_matrix(adt_path[i])), modality = "ADT"))
    
    # read atac
    if atac_path != "None":
        for i in range(len(atac_path)):
            if atac_path[i] == "None":
                atac_list.append(None)
            else:
                atac_feature_num = h5_to_matrix(atac_path[i]).shape[1]
                atac_list.append(scmomat.preprocess((h5_to_matrix(atac_path[i])), modality = "ATAC"))
    
    result = {"rna": rna_list, "adt": adt_list, "atac": atac_list}
    feature_num = [rna_feature_num, adt_feature_num, atac_feature_num]
    return result, feature_num

def run_scMoMaT(file_paths):
    
    processed_data,feature_num = read_h5_data(file_paths["rna_path"], file_paths["adt_path"], file_paths["atac_path"], len(file_paths["rna_path"]))
    counts_rnas = processed_data['rna']
    counts_adts = processed_data['adt']
    counts_atacs = processed_data['atac']

    rna_none = all(element is None for element in counts_rnas)
    adt_none = all(element is None for element in counts_adts)
    atac_none = all(element is None for element in counts_atacs)
    
    print(rna_none)
    print(adt_none)
    print(atac_none)

    if feature_num[0]!="None":
        genes = np.array(["rna" + str(i) for i in range(1, feature_num[0] + 1)])
    if feature_num[1]!="None":
        adt = np.array(["adt" + str(i) for i in range(1, feature_num[1] + 1)])
    if feature_num[2]!="None":
        atac = np.array(["atac" + str(i) for i in range(1, feature_num[2] + 1)])
    
    if not rna_none and not adt_none and atac_none:
        print(1)
        feats_name = {"rna": genes, "adt": adt}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[0]]), "rna":counts_rnas, "adt": counts_adts}
    elif not rna_none and adt_none and not atac_none:
        print(2)
        feats_name = {"rna": genes, "atac": atac}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[0]]), "rna":counts_rnas, "atac": counts_atacs}
    elif rna_none and not adt_none and not atac_none:
        print(3)
        print(len(file_paths[list(file_paths.keys())[0]]))
        
        feats_name = {"adt": adt, "atac": atac}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[2]]), "adt":counts_adts, "atac": counts_atacs}
    elif not rna_none and not adt_none and not atac_none:
        print(4)
        feats_name = {"rna": genes, "adt": adt, "atac": atac}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[0]]), "rna":counts_rnas,"adt":counts_adts, "atac": counts_atacs}
    elif not rna_none and adt_none and atac_none:
        print(5)
        feats_name = {"rna": genes}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[0]]), "rna":counts_rnas}
    elif rna_none and not adt_none and atac_none:
        print(6)
        feats_name = {"adt": adt}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[1]]), "adt":counts_adts}
    elif rna_none and adt_none and not atac_none:
        print(7)
        feats_name = {"atac": atac}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[2]]), "atac":counts_atacs}
        
    ######### training ############
    K = 30
    lamb = 0.001 
    T = 4000
    interval = 1000
    batch_size = 0.1
    lr = 1e-2
    seed = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = scmomat.scmomat_model(counts = counts, K = K, batch_size = batch_size, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
    losses = model.train_func(T = T)
    embedding = model.extract_cell_factors()

    return embedding


file_paths = {
    "rna_path": args.path1,
    "adt_path": args.path2,
    "atac_path": args.path3
}



result,marker_score = run_scMoMaT(file_paths)

end_time = time.time()
all_time = end_time - begin_time
print(all_time)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")

result_list = []
for i in range(len(result)):
    result_list.append(np.transpose(result[i]))

result = np.concatenate(result_list, 1)
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")
