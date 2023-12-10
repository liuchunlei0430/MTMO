
import argparse
import time
import os

from iPoLNG import iPoLNG
import torch
import scanpy as sc
import matplotlib.pyplot as plt
import h5py
import numpy as np
torch.set_default_tensor_type("torch.cuda.FloatTensor" if torch.cuda.is_available() else "torch.FloatTensor")


parser = argparse.ArgumentParser("iPOLNG")
parser.add_argument('--path1', metavar='DIR',  default="", help='path to train data1')
parser.add_argument('--path2', metavar='DIR',  default="", help='path to train data2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

begin_time = time.time()

    
def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.mat(np.array(f['matrix/data']).transpose())
        X = torch.from_numpy(X)
    return X
    
    
def run_iPOLNG(data1_path, data2_path):
    data1 = data_loader(data1_path)
    data2 = data_loader(data2_path)
    W = {"W1":data1.type("torch.cuda.FloatTensor"), "W2":data2.type("torch.cuda.FloatTensor")}

    print(W.keys())
    print(W['W1'].shape)
    print(W['W2'].shape)

    model = iPoLNG.iPoLNG(W, num_topics=20, integrated_epochs=300, warmup_epochs=500, seed=42, verbose=True)
    result = model.Run()
    
    embedding = result['L_est']
    embedding_rna = result['Ls_est']['W1']
    embedding_atac = result['Ls_est']['W2']

    return embedding, embedding_rna, embedding_atac


embedding, embedding_rna, embedding_atac = run_iPOLNG(args.path1, args.path2)

end_time = time.time()
all_time = end_time - begin_time
print(all_time)
print(embedding.shape)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")


file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=embedding)
file.close()
