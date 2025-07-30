import torch
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset, DataLoader
import jax_dataloader as jdl
from modules.clean_data_qp import clean
import os
import jax


RANDOM_SEED = 42
rng = jax.random.PRNGKey(RANDOM_SEED)



class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    



def get_and_clean_data(ezfio_path,prune, n_mo, batch_size=64, quantum=False,qlstm=False):
    
    '''
        Clean the qp files:
            - convert from decimal to binary with the same number of digits

        parameters:
            - ezfio_path: path to the ezfio folder
        
        return:
            - numpy array with the training dataset(determinants in  binary format) 
            - create the file with the deleted determinants to avoid repeating them in the next iteration
    '''

    
    qp_folder=os.path.join(ezfio_path,'determinants')
    psi_det_path=os.path.join(qp_folder,'psi_det')
    psi_coef_path=os.path.join(qp_folder,'psi_coef')

    x_train=clean(psi_det_path, psi_coef_path,prune)
    x_train=np.array(x_train,dtype=np.float32)  #no entiendo porque se convierte a float32, pero sino da error si le pasas int64
    if quantum:
         if x_train.ndim == 2:
            train_data=jdl.ArrayDataset(x_train, x_train) #uno es el input y el otro es el target, pero como son iguales no importa
            train_loader=jdl.DataLoader(train_data,backend='jax',batch_size=4,shuffle=True,drop_last=True, rng=rng)
            return train_loader, x_train

    if qlstm:
         if x_train.ndim == 2:
            x_train2 = np.expand_dims(x_train, axis=-1)
            train_data=jdl.ArrayDataset(x_train2, x_train2) #uno es el input y el otro es el target, pero como son iguales no importa
            train_loader=jdl.DataLoader(train_data,backend='jax',batch_size=batch_size,shuffle=True,drop_last=True, rng=rng)
            return train_loader, x_train

    #convert the training dataset to a pytorch tensor
    so_vectors=torch.tensor(x_train)
    seq_len=x_train.shape[1] #the secuence length is the number of molecular orbitals
    features=1
    num_samples = len(so_vectors) 
    tensor_data = so_vectors[:num_samples * seq_len]
    tensor_data = tensor_data.reshape((num_samples, seq_len, features))
    indices = torch.randperm(num_samples)
    tensor_data = tensor_data[indices]
    train_dataset = TimeSeriesDataset(tensor_data, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, x_train




def repited_determinants(determinantes, train):
    '''
        Identify repeated determinants:

        Function to remove the repeated determinants in the same set and the determinants that are in the training set:
            - convert vectors of 1 and 0 to decimal, is more efficient to compare
            - convert training vectors of 1 and 0 to decimal
            - first lets remove the repeated determinants in the same set, because during generation of determinants, we can have repeated determinants
            - find the new determinants that are not in the training set
        
        parameters:
            - determinantes: list of determinants
            - train: training dataset
        
        return:
            - final_dets: list of determinants that are not repeated in the same set and are not in the training set
    
    '''
    
    #convert vectors of 1 and 0 to decimal, is more efficient to compare
    determinantes_dec=[]
    for i in range(len(determinantes)):
        determinantes_dec.append(int("".join(map(str, determinantes[i][:][:][::-1])), 2))
    
    #convert training vectors of 1 and 0 to decimal
    train_dec=[]
    for i in range(len(train)):
        train_dec.append(int("".join(map(str, train[i][::-1])), 2))


    #first lets remove the repeated determinants in the same set, because during generation of determinants, we can have repeated determinants
    determinantes_dec, unique_indices=np.unique(determinantes_dec,return_index=True)
    determinantes_unique = determinantes[unique_indices]
    print('Number of determinants removed because they are repeated:',len(determinantes)-len(determinantes_unique))

    #find the new determinants that are not in the training set
    mask = np.isin(determinantes_dec, train_dec).astype(int)
    print('Number of determinants removed because they are in the training set:',np.count_nonzero(mask==1))

    #---------------------------------------------------------------------------------------------------
    #validate if the generated dets are not in the previous removed dets--------------------------------
    #---------------------------------------------------------------------------------------------------

    #read the deleted determinants file
    f = open('deleted_dets.txt', 'r')
    lines = f.readlines()
    f.close()
    lines=[int(line.strip()) for line in lines] #convert to int

    #find the new determinants that are not in the deleted determinants file
    determinantes_flip=np.fliplr(determinantes_unique.astype(int))
    #binary to decimal for a faster comparison
    determinantes_flip_dec=[int("".join(map(str, row)), 2) for row in determinantes_flip]
    mask2 = np.isin(determinantes_flip_dec, lines).astype(int)

    final_dets=determinantes_unique[np.logical_and(mask==0,mask2==0)]   #determinantes que no estan en el training set ni en el deleted_dets file
    print('Final number of determinants to be added to the training set:',len(final_dets))
    return final_dets



    