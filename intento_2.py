# ==============================================================================================
# Full Configuration Interaction optimization with LSTM as Autoencoder
# Authors: Jorge I. Hernandez-Martinez
# Affiliations:
# CINVESTAV Guadalajara, Department of Electrical Engineering and Computer Science, Jalisco, 45017, Mexico
# 
# This code is part of the research conducted for the paper titled:
# "Configuration Interaction Guided Sampling with Interpretable LSTM Autoencoders for Quantum Chemistry" (posible title)
# 
#python version: 3.10.12, recordar usar python3.10 -m pip install ***** para instalar paquetes
# ==============================================================================================
import sys


import numpy as np
import clean_data_qp
import format_to_qp
import matplotlib.pyplot as plt
import bash_commands
import time
import os
import multiprocessing
import clean

from joblib import Parallel, delayed


import random
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch
print('python version:',sys.version)
print('torch version:',torch.__version__)



#----------------------------------- Set the random seed for reproducibility-----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


#----------------------------------- Define LSTM Class-----------------------------------
class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return x
  
class Decoder(nn.Module):
  def __init__(self, seq_len,input_dim=64, n_features=8):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((-1,self.seq_len, self.hidden_dim))  
    x = F.normalize(torch.abs(self.output_layer(x)), p=1, dim=1)

    return x 
  

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features,embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

        
    def forward(self, x):    
        x = self.encoder(x)
        x = self.decoder(x)
        #x=self.electron_constriction(x)
        return x
    

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    



class SGLD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SGLD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']

            for param in group['params']:
                if param.grad is None:
                    continue
                
                # Aplicar weight decay si se especifica
                if weight_decay != 0:
                    param.grad = param.grad.add(param.data, alpha=weight_decay)
                
                # Añadir ruido gaussiano al gradiente
                noise = torch.randn_like(param.grad) * (lr ** 0.5)
                
                # Actualizar el parámetro añadiendo el gradiente y el ruido
                param.data.add_(param.grad, alpha=-lr).add_(noise)


    

def train_model(model, train_loader,  n_epochs,lr):
    #lambda_e = 1.0    # Ponderación de la penalización de la constricción electrónica
    optimizer = torch.optim.Adam(model.parameters(), lr)#optimizer Adam

    #optimizer ASGD
    #optimizer = torch.optim.ASGD(model.parameters(), lr=lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    #optimizer SGD
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    #nota despues hacer prueba con SGLD
    #optimizer = SGLD(model.parameters(), lr=lr, weight_decay=0.0001)

    criterion = nn.L1Loss(reduction='mean').to(device) #analizies MAE between prediction and target
    #criterion = nn.MSELoss(reduction='mean').to(device) #analizies MSE between prediction and target
    #criterion = nn.BCELoss()    # binary cross entropy loss, ya que las salidas de decoder estan entre 0 y 1 (usamos sigmoid)
    #criterion = nn.BCEWithLogitsLoss().to(device) # binary cross entropy loss, ya que las salidas de decoder estan entre 0 y 1 (usamos sigmoid)

    losses=[]

    for epoch in range(1, n_epochs + 1):
        model = model.train()
        print(f'\rÉpoca: {epoch}', end='', flush=True)
        train_losses = []
        lista_seq = []
        for seq_true in train_loader:
            #print('sequnce shape:',seq_true.shape)
            
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            #print(seq_true[0])
            #print('shape de seq_true:',seq_true[0].shape)
            seq_pred = model(seq_true)
            for i in range(len(seq_pred)):
                #print('shape de seq_pred:',seq_pred[i].shape)
                print('seq_pred:',seq_pred[i])
                lista_seq.append(seq_pred[i].cpu().detach().numpy())

            #loss_recon = criterion(seq_pred, seq_true)
            #loss_electrons = electron_penalty_strict(seq_pred, ne)
            #loss = loss_recon + lambda_e * loss_electrons
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print('len de la lista de secuencias',len(lista_seq))
        print('elementos unicos en la secuencia',len(np.unique(lista_seq)))

        
        losses.append(np.mean(train_losses))

        if epoch % 10 == 0:
                #print(f"Recon loss: {loss_recon.item():.4f}, Electron penalty: {loss_electrons.item():.4f}")
                #print(f"Total loss: {loss.item():.4f}")
                print(f"Loss:{losses[-1]:.4f}")

        #losses.append(train_losses)
    loss_plot(losses)
        

    return model.eval(),losses




def lstm_initialization(n_mo,ne,embedding_dim):
    '''
        Initialize the rbm class

        parameters:
            - ezfio_path: path to the ezfio folder
            - prune: threshold to prune the coefficients of the determinants
        
        return:
            - rbm: rbm class initialized
    '''
    #bash_commands.unzip_dets_coefs(ezfio_path)
    #x_train=get_and_clean_data(ezfio_path,prune)
    #num_visible = x_train.shape[1] 
    #num_hidden = x_train.shape[1] 
    sequence_length=n_mo



    #model = RecurrentAutoencoder(sequence_length, 1,embedding_dim)   #seq_len,ne, n_features,embedding_dim
    model = RecurrentAutoencoder(26, 1,64)
    lstm = model.to(device)

    #bash_commands.zip_dets_coefs(ezfio_path)

    return lstm




def get_and_clean_data_2():
    #working_directory=os.getcwd()
    working_directory='/home/ivan/Descargas/Python_Codes_DFT/paper_code_implementation/lstm_fci'
    #name='psi_det_2'
    name=working_directory+'/psi_det_2'
    so_vectors=clean.clean(name)
    so_vectors=np.array(so_vectors,dtype=np.float32)
    #display(so_vectors[:3])
    so_vectors=torch.tensor(so_vectors)
    #
    #print(so_vectors.shape)

    seq_len=26
    features=1
    num_samples = len(so_vectors) 
    tensor_data = so_vectors[:num_samples * seq_len]
    print('tensor_data:',tensor_data.shape)
    #que tipo de dato es?
    print('tipo de dato:',tensor_data.dtype)
    tensor_data = tensor_data.reshape((num_samples, seq_len, features))
    indices = torch.randperm(num_samples)
    tensor_data = tensor_data[indices]
    train_dataset = TimeSeriesDataset(tensor_data, seq_len)
    batch_size=64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, so_vectors





            


def get_and_clean_data(ezfio_path,prune, n_mo, batch_size=64):
    
    '''
        Clean the qp files:
            - convert from decimal to binary with the same number of digits

        parameters:
            - ezfio_path: path to the ezfio folder
        
        return:
            - numpy array with the training dataset(determinants in  binary format) 
            - create the file with the deleted determinants to avoid repeating them in the next iteration
    '''

    qp_folder=ezfio_path+'/determinants/'
    x_train=clean_data_qp.clean(qp_folder+'psi_det', qp_folder+'psi_coef',prune)
    x_train=np.array(x_train,dtype=np.float32)  #no entiendo porque se convierte a float32, pero sino da error si le pasas int64

    #convert the training dataset to a pytorch tensor
    so_vectors=torch.tensor(x_train)
    seq_len=n_mo #the secuence length is the number of molecular orbitals
    features=1
    num_samples = len(so_vectors) 
    tensor_data = so_vectors[:num_samples * seq_len]
    print('tensor_data shape:',tensor_data.shape)
    print('tipo de dato:',tensor_data.dtype)
    tensor_data = tensor_data.reshape((num_samples, seq_len, features))
    indices = torch.randperm(num_samples)
    tensor_data = tensor_data[indices]
    train_dataset = TimeSeriesDataset(tensor_data, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, x_train








    
if __name__=='__main__':
    Initial_time = time.time()

    ##-----------------------------------examples to test the code-----------------------------------
    ##------------------------------------------------------------------------------------------------

    #set woking directory
    working_directory='/home/ivan/Descargas/Python_Codes_DFT/paper_code_implementation/lstm_fci'
    




    #path to the ezfio folder for the molecule------------------------
    #ezfio_path='/home/ivan/Descargas/solving_fci/to_diagonalize.ezfio' #c2 ccpvdz
    ezfio_path='/home/ivan/Descargas/QP_examples/h2o/h2o_631g.ezfio'   #h2o 6-31g
    #ezfio_path='/home/ivan/Descargas/QP_examples/h2o/h2o_ccpvdz.ezfio'   #h2o ccpvdz
    #ezfio_path='/home/ivan/Descargas/QP_examples/c2/c2_631g.ezfio' #c2 6-31g


    #primeras pruebas con times det num 20, max iter 10, aprox davidson 1e-10,1e-6, y 1e-8, prune 1e-8
    max_iterations=4; num_epochs=10; learning_rate=0.001;times_num_dets_gen=15;prune=1e-10;tol=1e-5; times_max_diag_time=10
    batch_size=64; embedding_dim=64


    #exanple of FCI energy for the molecule to compare the convergence
    FCI_energy=-76.12237    #h2o 6-31g
    #FCI_energy=-75.72984    #c2 ccpvdz
    #FCI_energy=-76.24195    #h2o ccpvdz
    #FCI_energy = -75.64418 #c2 6-31g

    n_mo=26
    DataLoader_1,x_train =get_and_clean_data_2()
    lstm=lstm_initialization(ezfio_path,n_mo,embedding_dim)
            #rbm.train(x_train, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, k=k)  #train the rbm
    model_trained,_=train_model(lstm, DataLoader_1, num_epochs,learning_rate)     #train the lstm



    