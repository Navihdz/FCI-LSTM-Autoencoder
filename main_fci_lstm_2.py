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
#se puede seleccionar version de python en vs usando Ctrl+Shift+P y escribiendo python: select interpreter
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
import lstm_classic as lstm
import jax.numpy as jnp
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # no preasignes toda la memoria de GPU
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'
#XLA_PYTHON_CLIENT_PREALLOCATE=False

from joblib import Parallel, delayed


import random
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
import multiprocessing as mp
from multiprocessing import Process,Queue
import torch
print('python version:',sys.version)
print('torch version:',torch.__version__)
import quantum_autoencoder as qautoencoder
from qlstm_ae import train_model, qlstm_initialization
from multiprocessing.pool import ThreadPool
import jax_dataloader as jdl
import gc
import jax





#----------------------------------- Set the random seed for reproducibility-----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


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

    qp_folder=ezfio_path+'/determinants/'
    x_train=clean_data_qp.clean(qp_folder+'psi_det', qp_folder+'psi_coef',prune)
    x_train=np.array(x_train,dtype=np.float32)  #no entiendo porque se convierte a float32, pero sino da error si le pasas int64
    if quantum:
         if x_train.ndim == 2:
            train_data=jdl.ArrayDataset(x_train, x_train) #uno es el input y el otro es el target, pero como son iguales no importa
            train_loader=jdl.DataLoader(train_data,backend='jax',batch_size=4,shuffle=True,drop_last=True)
            return train_loader, x_train

    if qlstm:
         if x_train.ndim == 2:
            x_train2 = np.expand_dims(x_train, axis=-1)
            print("Shape after expand_dims:", x_train.shape) 
            train_data=jdl.ArrayDataset(x_train2, x_train2) #uno es el input y el otro es el target, pero como son iguales no importa
            train_loader=jdl.DataLoader(train_data,backend='jax',batch_size=batch_size,shuffle=True,drop_last=True)
            return train_loader, x_train

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





def mutate_det_with_prob(visible_probs, dets_train):
    import random
    import numpy as np
    import jax.numpy as jnp

    excitations = 2 if np.random.rand() <= 0.5 else 1

    random_number_det = np.random.randint(len(dets_train))
    det = dets_train[random_number_det]
    occupied_orbitals = np.where(det == 1)[0]
    unoccupied_orbitals = np.where(det == 0)[0]

    # Probabilidades de salto
    occupied_probs = visible_probs[occupied_orbitals]
    unoccupied_probs = visible_probs[unoccupied_orbitals]
    jumps = (1 - occupied_probs[:, None]) * unoccupied_probs[None, :]
    jumps = jumps.reshape(len(occupied_orbitals), len(unoccupied_orbitals))

    # Determinar si estamos usando JAX
    use_jax = isinstance(jumps, jnp.ndarray)

    new_det = det.copy()
    saved_jumps = []

    for _ in range(excitations):
        p = random.random() * (jnp.sum(jumps) if use_jax else np.sum(jumps))
        c = 0.0
        found_change = False

        for j in range(len(occupied_orbitals)):
            for k in range(len(unoccupied_orbitals)):
                c += jumps[j, k]
                if p <= c and jumps[j, k] != 0:
                    if (occupied_orbitals[j] % 2) == (unoccupied_orbitals[k] % 2):
                        new_det[occupied_orbitals[j]] = 0
                        new_det[unoccupied_orbitals[k]] = 1

                        if use_jax:
                            jumps = jumps.at[j].set(0)
                            jumps = jumps.at[:, k].set(0)
                        else:
                            jumps[j] = 0
                            jumps[:, k] = 0

                        saved_jumps.append([occupied_orbitals[j], unoccupied_orbitals[k]])
                        found_change = True
                        break
            if found_change:
                break

        if not found_change:
            for j in range(len(occupied_orbitals) - 1, -1, -1):
                for k in range(len(unoccupied_orbitals) - 1, -1, -1):
                    if jumps[j, k] != 0 and (occupied_orbitals[j] % 2) == (unoccupied_orbitals[k] % 2):
                        new_det[occupied_orbitals[j]] = 0
                        new_det[unoccupied_orbitals[k]] = 1

                        if use_jax:
                            jumps = jumps.at[j].set(0)
                            jumps = jumps.at[:, k].set(0)
                        else:
                            jumps[j] = 0
                            jumps[:, k] = 0

                        found_change = True
                        break
                if found_change:
                    break

    return new_det




@torch.no_grad()
def generate_batch_probs(autoencoder,params, dets_train, batch_size, n_mo,classical=False, quantum=False,qlstm=True):
    if classical==True:
        print('predicting with classical lstm')
        device = next(autoencoder.parameters()).device
        indices = np.random.choice(len(dets_train), size=batch_size, replace=True)
        input_dets = torch.tensor(dets_train[indices], dtype=torch.float32, device=device).unsqueeze(-1)
        output_probs = autoencoder(input_dets).squeeze(-1).cpu().numpy()  # (batch_size, n_mo)
        new_dets = Parallel(n_jobs=-1)(
        delayed(mutate_det_with_prob)(output_probs[i], dets_train) for i in range(batch_size))
    elif qlstm==True:
        print('predicting with qlstm')
        #jax.clear_backends() 
        #model_name = qlstm_autoencoder
        indices = np.random.choice(len(dets_train), size=batch_size, replace=True)
        input_dets = jnp.array(dets_train[indices], dtype=np.float32)
        input_dets=jnp.expand_dims(input_dets, axis=-1)
        #input_shape = input_dets.shape
        #model, params = load_model(model_name, input_shape=input_shape)
        output_probs = autoencoder.apply(params, input_dets)
        output_probs = np.array(output_probs)
        new_dets = Parallel(n_jobs=-1)(
        delayed(mutate_det_with_prob)(output_probs[i], dets_train) for i in range(batch_size))
        #gc.collect()
        #jax.clear_caches()

    else:
        print('predicting with dummy version of quantum')
        indices = np.random.choice(len(dets_train), size=batch_size, replace=True)
        input_dets = np.array(dets_train[indices], dtype=np.float32)
        model, params = qautoencoder.load_model(n_mo)

        output_probs = model.apply(params,input_dets)
        #output_probs = jax.device_get(output_probs)
        output_probs = np.array(output_probs)
        new_dets = Parallel(n_jobs=-1)(
        delayed(mutate_det_with_prob)(output_probs[i], dets_train) for i in range(batch_size))

    #print('output_probs shape:',output_probs[0].shape)
    

    # Parallel execution (usa todos los núcleos disponibles)
        new_dets = Parallel(n_jobs=-1)(
        delayed(mutate_det_with_prob)(output_probs[i], dets_train) for i in range(batch_size)
    )
    #from joblib import Parallel, delayed
    #import threading
    #print("JAX activo:", threading.active_count())

        

    #print('new_dets shape:',np.array(new_dets).shape)
    #print('new_dets:',np.array(new_dets)[0])
    #print('new_dets:',np.array(new_dets)[-1])

    return new_dets, output_probs


def det_generation(autoencoder,params,dets_train,dets_list,num_dets, n_mo, quantum=False,qlstm=True):
    m=0 #Counter of determinants generated
    train_dec_set = set(int("".join(map(str, x[::-1])), 2) for x in dets_train) #usar set para hacer la busqueda mas rapida (de O(n) a O(1))

    #set the seed for each multiprocess
    random.seed(os.getpid())
    set_dets_list=set()

    #run until we have the number of determinants that fulfill the conditions of single and double substitutions
    batch_size=num_dets//3
    while len(dets_list) < num_dets:
        new_det_batch,output_probs = generate_batch_probs(autoencoder,params, dets_train, batch_size=batch_size, n_mo=n_mo, quantum=quantum,qlstm=qlstm)
        for det in new_det_batch:
            det_dec = int("".join(map(str, det[::-1])), 2)
            if det_dec not in train_dec_set and det_dec not in set_dets_list:
                dets_list.append(det)
                set_dets_list.add(det_dec)
                #if len(dets_list) % 1000 == 0:
                print('number of determinants generated : ', len(dets_list),'  of ',num_dets, end='\r')
            if len(dets_list) >= num_dets:
                break

    return dets_list


    


def plot_ground_energy(ground_energy_list, ezfio_name,FCI_energy=None):
    '''
        scatter Plot of ground energy-
        this shows the ground energy of each iteration

        parameters:
            - ground_energy_list: list of ground energies

        return:
            - plot of ground energy
    '''
    #figure with 300 dpi
    plt.figure(dpi=300)
    plt.plot(ground_energy_list,'-o')

    if FCI_energy!=None:
        plt.axhline(y=FCI_energy, color='r', linestyle='-')

    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Computed Ground Energy in each iteration for "+ezfio_name)
    plt.legend(['Computed Ground Energy','FCI Energy'])
    plt.savefig('graphs/energy_plot/'+ezfio_name+'_ground_energy.png', bbox_inches='tight') #esto es para que no las recorte al guardar
    plt.close()


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


def get_energy(ezfio_path,calculation='cisd'):
    '''
        Get the energy from the qp output file

        parameters:
            - ezfio_path: path to the ezfio folder
            - calculation: type of calculation (cisd, scf, diagonalization)
        
        return:
            - energy: ground energy
    '''
    output_file=ezfio_path+'/determinants/qp.out'
    print('output file:',output_file)
    with open(output_file) as f:
        lines = f.readlines()

        if calculation=='cisd':
            for i in range(len(lines)):
                if 'CISD Energies' in lines[i]:
                    energy=lines[i+1].split()[-1]
                    break
        elif calculation=='scf':
            for i in range(len(lines)):
                if 'SCF energy' in lines[i]:
                    energy=lines[i].split()[-1]
                    break
        elif calculation=='diagonalization':
            for i in range(len(lines)):
                if 'Energy of state' in lines[i]:
                    energy=lines[i].split()[-1]
                    break

    energy=float(energy)
    print('Computed Ground Energy:',energy)

    return energy


def plot_DetDistribution(determinantes, x_train,ezfio_name,iteration):
    dets_to_plot=[]
    for i in range(len(determinantes)):
        dets_to_plot.append(np.reshape(determinantes[i],determinantes.shape[1]))

    # Compute the frequency of ones (occuped MO) in each position for all the determinants
    frecuencias_1 = [0] * len(dets_to_plot[0])  # Initialize with zeros for the distribution of the generated determinants
    frecuencias_2 = [0] * len(x_train[0])  # Initialize with zeros for the distribution of the training determinants

    # Sum the bits in each position for all the determinants
    for vector in dets_to_plot:
        for i, bit in enumerate(vector):
            frecuencias_1[i] += bit    

    for vector in x_train:
        for i, bit in enumerate(vector):
            frecuencias_2[i] += bit

    
    #figure with 300 dpi
    plt.figure(dpi=300)
    
    plt.figure(figsize=(18, 6))
    posiciones = np.arange(len(frecuencias_1))

    plt.bar(posiciones, frecuencias_1, width=0.6, label='Generated Determinants', color='b', alpha=0.7)
    plt.bar(posiciones + 0.6, frecuencias_2, width=0.6, label='Training Determinants', color='g', alpha=0.7)

    plt.xlabel("Molecular Orbitals")
    plt.ylabel("Occupancy frequency")
    plt.title('occupancy frequency in each MO in iteration '+str(iteration)+' of '+ezfio_name)
    plt.xticks(posiciones + 0.3, range(len(frecuencias_1)))
    plt.legend()
    #plt.show()
    plt.savefig('graphs/det_distribution/det_distribution_'+ezfio_name+'_iteration_'+str(iteration)+'.png')
    plt.close()

def plot_DetDistribution2(determinantes, x_train,ezfio_name,iteration,ezfio_path):
    '''
        Plot the distribution of the determinants but weighted using the coefficient of the determinants to weight the distribution
        -note: this was used to compare the distribution weighted and not weighted of the determinants
    '''

    #unzip the files psi_coef and psi_det in the ezfio folder
    bash_commands.unzip_dets_coefs(ezfio_path)

    #open file to read the coefficients (omit the first two lines)
    f = open(ezfio_path+'/determinants/psi_coef', 'r')
    lines = f.readlines()
    f.close()
    lines=lines[2:] #omit the first two lines

    #convert to float
    coeficientes=[float(line.strip()) for line in lines]

    #zip again the files psi_coef and psi_det in the ezfio folder
    bash_commands.zip_dets_coefs(ezfio_path)

    dets_to_plot=[]
    for i in range(len(determinantes)):
        dets_to_plot.append(np.reshape(determinantes[i],determinantes.shape[1]))
    
    
    # Compute the frequency of ones (occuped MO) in each position for all the determinants
    frecuencias_1 = [0] * len(dets_to_plot[0])  # Initialize with zeros for the distribution of the generated determinants
    frecuencias_2 = [0] * len(x_train[0])  # Initialize with zeros for the distribution of the training determinants

    # Sum the bits in each position for all the determinants
    for i in range(len(dets_to_plot)):
        for j in range(len(dets_to_plot[i])):
            if dets_to_plot[i][j]==1:
                frecuencias_1[j] += coeficientes[i]
    
    for i in range(len(x_train)):
        for j in range(len(x_train[i])):
            if x_train[i][j]==1:
                frecuencias_2[j] += 1
    

    posiciones = np.arange(len(frecuencias_1))
    plt.figure(figsize=(18, 6))
    posiciones = np.arange(len(frecuencias_1))

    #figure with 300 dpi
    plt.figure(dpi=300)

    #plot each bar as a subplot
    plt.bar(posiciones, frecuencias_1, width=0.6, label='Generated Determinants', color='b', alpha=0.7)
    plt.bar(posiciones + 0.6, frecuencias_2, width=0.6, label='Training Determinants', color='g', alpha=0.7)

    plt.xlabel("Molecular Orbitals")
    plt.ylabel("Occupancy frequency")
    plt.title('occupancy frequency in each MO in iteration '+str(iteration)+' of '+ezfio_name)
    plt.xticks(posiciones + 0.3, range(len(frecuencias_1)))
    plt.legend()
    #plt.show()

    plt.savefig('graphs/weighted_det_distribution/w_det_distribution'+ezfio_name+'_iteration_'+str(iteration)+'.png')
    plt.close()

    

def plot_time_per_dets(number_of_det_list,ezfio_name,times_per_iteration_list):
    '''
        Plot of time per number of determinants-
        this shows the time per iteration and the number of determinants generated in each iteration

        parameters:
            - number_of_det_list: list of number of determinants
            - ezfio_name: name of the ezfio folder (ex: h2o_631g.ezfio)
            - times_per_iteration_list: list of time per iteration

        return:
            - plot of time per iteration
    '''
    #figure with 300 dpi
    plt.figure(dpi=300)
    plt.plot(number_of_det_list,times_per_iteration_list,'-o',color='b')
    plt.xlabel("Number of determinants")
    plt.ylabel("Time (s)")
    plt.title("Computational Time per determinants number for "+ezfio_name)
    plt.savefig('graphs/time_per_dets/'+ezfio_name+'_time_per_dets.png')
    plt.close()



def loss_plot(losses):
    '''
        Plot the loss function

        parameters:
            - losses: list of losses

        return:
            - plot of loss function
    '''
    plt.figure(dpi=300)
    #tamaño de letra
    plt.rcParams.update({'font.size': 13})
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Function")
    plt.savefig('graphs/loss_plot/loss_function.png', bbox_inches='tight') #esto es para que no las recorte al guardar
    plt.close()


def hamming_dist(determinantes, x_train):
    '''
        Compute the hamming distance between the determinants generated and the training set

        parameters:
            - determinantes: list of determinants
            - x_train: training dataset
        
        return:
            - hamming distance
    '''
    print('shape determinantes:',determinantes.shape)
    print('shape x_train:',x_train.shape)
    print('len determinantes:',len(determinantes))
    
    '''
    min_hamming_dist=[]
    for i in range(len(determinantes)):
        hamming_dist=np.count_nonzero(determinantes[i] != x_train, axis=1)
        # Encuentra las distancias mínimas
        min_distance=np.min(hamming_dist)
        min_hamming_dist.append(min_distance)
    '''

    #vetorizada
    hamming_dist = np.sum(determinantes[:, None, :] != x_train[None, :, :], axis=2)

    #si estamos comparando entre el mismo set de determinantes, entonces la distancia minima es la segunda minima ya que la primera es cero (el mismo determinante)
    if determinantes.shape==x_train.shape and np.array_equal(determinantes,x_train):
        # Encuentra la distancia mínima para cada determinante en la matriz de comparación
        min_hamming_dist = np.partition(hamming_dist, 1, axis=1)[:, 1]
        #print('min_hamming_dist:',min_hamming_dist)

    else:    
        # Encuentra la distancia mínima para cada determinante en la matriz de comparación
        min_hamming_dist = np.min(hamming_dist, axis=1)
    
    return min_hamming_dist

def save_outputs(ground_energy_list, number_of_det_list, times_per_iteration_list, ezfio_name):
        #save ground_energy_list, number_of_det_list and times_per_iteration_list to a file
    with open('ouput_files/'+ezfio_name+'_ground_energy_list.txt', 'w') as f:
        for item in ground_energy_list:
            f.write("%s\n" % item)
    with open('ouput_files/'+ezfio_name+'_number_of_det_list.txt', 'w') as f:
        for item in number_of_det_list:
            f.write("%s\n" % item)
    with open('ouput_files/'+ezfio_name+'_times_per_iteration_list.txt', 'w') as f:
        for item in times_per_iteration_list:
            f.write("%s\n" % item)
def run_training(DataLoader, n_mo, q):
    #DataLoader, x_train = get_and_clean_data(ezfio_path, prune, n_mo, batch_size, qlstm=True)
    trained_model, losses = train_model(n_mo, DataLoader, epochs=10)
    #q.put("done")  # puedes enviar confirmación (no el modelo directamente si pesa mucho)


def main(working_directory,ezfio_path,qpsh_path,iterations=2,num_epochs=1, learning_rate=0.01,times_num_dets_gen=2, 
         prune=1e-12,tol=1e-5,FCI_energy=None, times_max_diag_time=2, batch_size=64, embedding_dim=16, n_qubits=2):
    '''
        Main function to run the script

        parameters:
            - working_directory: path to the working directory
            - ezfio_path: path to the ezfio folder
            - qpsh_path: path to the qpsh folder
            - iterations: number of iterations to run
            - num_epochs: number of epochs to train the rbm
            - learning_rate: learning rate for the rbm
            - times_num_dets_gen: number of determinants to generate
            - prune: threshold to prune the coefficients of the determinants
            - tol: tolerance for the convergence of the diagonalization
            - FCI_energy: FCI energy for the system (optional)
    '''
    n_mo=bash_commands.get_num_mo(ezfio_path)
    ne=bash_commands.get_num_electrons(ezfio_path)
    print('Number of electrons:',ne )

    print('Number of molecular orbitals(alphas + betas):',n_mo)

    os.chdir(working_directory)
    print('Current working directory:',os.getcwd())
    
    ground_energy_list=[]
    number_of_det_list=[]
    times_per_iteration_list=[]
    print('The selected ezfio_path is :',ezfio_path)
    ezfio_name=ezfio_path.split('/')[-1]

    print("current qpsh path:",qpsh_path)

    #delete the deleted_dets.txt file if exists, this file is used to store the determinants that are removed from the training set
    if os.path.exists('deleted_dets.txt'):
        os.remove('deleted_dets.txt')

    #num_nucleos = multiprocessing.cpu_count()
    #print("Number of cores available:", num_nucleos)

    for iteration in range (iterations):
        print('**********************************************************')
        print('************ iteration:',iteration,'***************************')
        print('**********************************************************')

        #compute scf and cisd in the first iteration
        if iteration==0:
            bash_commands.reset_ezfio(qpsh_path,ezfio_path)
            bash_commands.scf(qpsh_path,ezfio_path)
            ground_energy_list.append(get_energy(ezfio_path,calculation='scf'))
            bash_commands.cisd(qpsh_path,ezfio_path)
            ground_energy_list.append(get_energy(ezfio_path,calculation='cisd'))

            #initialize the rbm here to not reset the weights in each iteration
            #model=lstm.lstm_initialization(n_mo,embedding_dim)

            #vamos a inicializar el modelo de lstm cuantico
            jax.clear_caches()
            gc.collect()
            bash_commands.unzip_dets_coefs(ezfio_path)
            DataLoader, x_train = get_and_clean_data(ezfio_path, prune, n_mo, batch_size, qlstm=True)
            autoencoder,params,train_step=qlstm_initialization(n_mo, n_qubits,embedding_dim)
            print('model initialized')
    
        
        #if is not the first iteration, use the rbm to generate new determinants and then diagonalize 
        else:
            init_time=time.time()
            bash_commands.unzip_dets_coefs(ezfio_path) #unzip the files psi_coef and psi_det in the determinants of the ezfio folder

            quantum=False
            qlstm=True

            if quantum:
                print('predicting with quantum autoencoder')
                jax.clear_caches()
                DataLoader,x_train =get_and_clean_data(ezfio_path,prune, n_mo, batch_size, quantum=True)
                model,params,dropoutkey=qautoencoder.qae_initialization(n_mo)  #quantum encoder decoder
                model_trained,params, losses=qautoencoder.train_model(model,params, dropoutkey, DataLoader, num_epochs,learning_rate)  #quantum lstm
                loss_plot(losses) #plot the loss function
                qautoencoder.save_model(model_trained,params)
                model,params=qautoencoder.load_model(n_mo)  #quantum lstm
                #autoencoder = model.apply(params, x_train, train=False, rngs={'dropout': jax.random.PRNGKey(0)})
                autoencoder = model.apply(params, x_train)
            elif qlstm:
                print('predicting with qlstm')
                DataLoader, x_train = get_and_clean_data(ezfio_path, prune, n_mo, batch_size, qlstm=True)
                #mp.set_start_method("spawn", force=True)
                #jax.clear_caches()
                #gc.collect()
                '''
                q = Queue()
                print('creando el proceso')
                p = Process(target=run_training, args=(DataLoader, n_mo, q))
                p.start()
                result = q.get()  # espera el "done"
                p.join()
                '''
                
                autoencoder, params = train_model(autoencoder, params,train_step, DataLoader, epochs=num_epochs, lr=learning_rate)


                print('proceso terminado')
                # luego cargas el modelo ya entrenado
                
                del DataLoader
                #del losses
                #gc.collect()
                #model_class = qlstm_autoencoder
                #autoencoder, params = load_model(model_class, input_shape=n_mo)
                

            
            else:

                #get the training dataset and the number of electrons in the molecule (in binary format) and 
                #convert the training dataset to a pytorch tensor
                DataLoader,x_train =get_and_clean_data(ezfio_path,prune, n_mo, batch_size) #classical lstm
                # the file with the deleted determinants to avoid repeating them in the next iteration
                #initialize the rbm here to reset the weights in each iteration (you need to comment the previous initialization)----------
                model=lstm.lstm_initialization(n_mo,embedding_dim)     #classical lstm
                model_trained,losses=lstm.train_model(model, DataLoader, num_epochs,learning_rate)     #classical lstm
                lstm.save_model(model_trained)
                autoencoder=lstm.load_model(n_mo,ne)
            
            
            

            num_dets_gen=times_num_dets_gen*len(x_train)  #number of determinants to generate
            dets_list=[]
            x_train=x_train.astype(int)
            determinantes=det_generation(autoencoder,params,x_train,dets_list,num_dets_gen, n_mo, quantum)
            print('number of determinants generated:',len(determinantes))


            #Validate if the new determinants, are not in the training set
            determinantes_2 = np.array(determinantes).astype(int).reshape(len(determinantes),n_mo) # reshape uusing the number of molecular orbitals
            
            #este ya no parece ser necesario dado que en la generacion de determinantes ya se eliminan los repetidos
            determinantes_3 = repited_determinants(determinantes_2, x_train.astype(int))
            #determinantes_3=determinantes_2


            #hamming_distance=hamming_dist(determinantes_3, x_train)
            #print('Number of dets with hamming distance >3:',np.count_nonzero(np.array(hamming_distance)>3))

            #hamming_distance=hamming_dist(x_train,x_train)
            #print('Number of dets with hamming distance after concatenate>3:',np.count_nonzero(np.array(hamming_distance)>3))
            

            #join the new determinants with the training set
            new_dets=np.concatenate((x_train,determinantes_3),axis=0)

            #convert the new determinants to qp format and add them to the qp files
            determinantes_3_toqp=np.fliplr(determinantes_3.astype(int))  #format_to_qp function needs the determinants in reverse order ex from 110000 to 000011 to a correct conversion to decimal               
            format_to_qp.formatting2qp(determinantes_3_toqp,ezfio_path)

            number_of_det_list.append(len(new_dets))

            #run the diagonalization
            bash_commands.zip_dets_coefs(ezfio_path)
            bash_commands.write_det_num(ezfio_path)

            #we can modify the threshold of the davidson algorithm depending on the number of determinants

            if len(new_dets)<20000:
                bash_commands.modify_threshold_davidson(ezfio_path,'1e-10')
            elif len(new_dets)>20000 and len(new_dets)<50000:
                bash_commands.modify_threshold_davidson(ezfio_path,'1e-10')
            elif len(new_dets)>50000:
                bash_commands.modify_threshold_davidson(ezfio_path,'1e-10')
            
            #2 times greater than previous time
            max_diag_time= times_max_diag_time*times_per_iteration_list[-1] if iteration>1 else 1000
            print('max_diag_time:',max_diag_time)
            process_completed=bash_commands.diagonalization(qpsh_path,ezfio_path,max_diag_time)

            #if the diagonalization process is completed successfully continue with the next steps
            if process_completed or iteration==1:
                
                #get the energy from the qp output file
                ground_energy_list.append(get_energy(ezfio_path,calculation='diagonalization'))

                end_time=time.time()

                plot_DetDistribution(determinantes_3, x_train,ezfio_name,iteration)
                plot_DetDistribution2(determinantes_3, x_train,ezfio_name,iteration,ezfio_path)

                times_per_iteration_list.append(end_time-init_time)


                #check if the energy is converged
                if iteration>1:
                    if abs(ground_energy_list[-1]-ground_energy_list[-2])<tol:
                        print('Energy converged at iteration',iteration,'to a value of',ground_energy_list[-1])
                        break

                    # If the last energy is > 2% or 1% higher than the previous energy, stop the iterations. This observation comes from experiments,
                    # since when the energy change is too large, it indicates that the diagonalization did not converge properly and the algorithm
                    # has a high probability of not converging in the next iteration.
                    if abs(ground_energy_list[-1] - ground_energy_list[-2]) > 1.5e-1:
                        print('the Algorith has to stop because the diagonalization did not converge properly')
                        print('Energy converged at iteration',iteration,'to a value of',ground_energy_list[iteration])
                        #remove the last calculations
                        ground_energy_list.pop()
                        number_of_det_list.pop()
                        times_per_iteration_list.pop()

                        break
                    
                    
            else:
                print('Diagonalization process not completed or not converged')
                #remove the last number of determinants because not exist time for the diagonalization process
                number_of_det_list.pop()
                break

        #plot the ground energy, time per iteration and time per number of determinants in each iteration

        plot_ground_energy(ground_energy_list,ezfio_name, FCI_energy)
        plot_time_per_dets(number_of_det_list,ezfio_name,times_per_iteration_list)

        #save the outputs to a file
        save_outputs(ground_energy_list, number_of_det_list, times_per_iteration_list, ezfio_name)

        #hall time 
        current_time=time.time()
        print(f'Hall time: {current_time-Initial_time} seconds or {round((current_time-Initial_time)/60,2)} minutes')

            
    print('number of determinants in each iteration:',number_of_det_list)
    print('time per iteration:',times_per_iteration_list)
    return ground_energy_list,number_of_det_list,times_per_iteration_list

    


    
if __name__=='__main__':
    Initial_time = time.time()
    ##-----------------------------------examples to test the code-----------------------------------
    ##------------------------------------------------------------------------------------------------
    #set woking directory
    working_directory='/home/ivan/Descargas/Python_Codes_DFT/paper_code_implementation/lstm_fci'
    #working_directory='/home/sandra-juarez/Documentos/Doctorado/fci-autoencoder/FCI-LSTM-Autoencoder'

    os.chdir(working_directory)
    



    #path to the ezfio folder for the molecule------------------------
    #ezfio_path='/home/ivan/Descargas/solving_fci/to_diagonalize.ezfio' #c2 ccpvdz
    ezfio_path='/home/ivan/Descargas/QP_examples/h2o/h2o_631g.ezfio' #h2o 6-31g
    #ezfio_path='/home/sandra-juarez/Documentos/Doctorado/fci-autoencoder/FCI-LSTM-Autoencoder/QP_examples/h2o/h2o_631g.ezfio' #/home/ivan/Descargas/QP_examples/h2o/h2o_631g.ezfio'   #h2o 6-31g
    #ezfio_path='/home/ivan/Descargas/QP_examples/h2o/h2o_ccpvdz.ezfio'   #h2o ccpvdz
    #ezfio_path='/home/ivan/Descargas/QP_examples/c2/c2_631g.ezfio' #c2 6-31g

    #path to the Quantum Package qpsh---------------------------------
    #qpsh_path='/home/sandra-juarez/qp2/bin/qpsh'
    qpsh_path='/home/ivan/Descargas/qp2/bin/qpsh' #/home/sandra-juarez/qp2/bin/qpsh
    ezfio_name=ezfio_path.split('/')[-1]

    #primeras pruebas con times det num 20, max iter 10, aprox davidson 1e-10,1e-6, y 1e-8, prune 1e-8
    max_iterations=4; num_epochs=2; learning_rate=0.001;times_num_dets_gen=15;prune=1e-12;tol=1e-5; times_max_diag_time=10
    batch_size=128; embedding_dim=16; n_qubits=4

    #parametros que funcionaron bien en primera prueba qlstm
    #max_iterations=4; num_epochs=10; learning_rate=0.01;times_num_dets_gen=15;prune=1e-12;tol=1e-5; times_max_diag_time=10
    #batch_size=64; embedding_dim=16; n_qubits=2




    #exanple of FCI energy for the molecule to compare the convergence
    FCI_energy=-76.12237    #h2o 6-31g
    #FCI_energy=-75.72984    #c2 ccpvdz
    #FCI_energy=-76.24195    #h2o ccpvdz
    #FCI_energy = -75.64418 #c2 6-31g

    #n_mo=26
    #DataLoader_1,x_train =get_and_clean_data(ezfio_path,prune, n_mo, batch_size)
    #lstm=lstm_initialization(ezfio_path,n_mo,embedding_dim)
            #rbm.train(x_train, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, k=k)  #train the rbm
    #model_trained,_=train_model(lstm, DataLoader_1, num_epochs,learning_rate)     #train the lstm

    

    ground_energy_list,number_of_det_list,times_per_iteration_list=main(working_directory,ezfio_path,qpsh_path,max_iterations,num_epochs, learning_rate, 
                                                                        times_num_dets_gen,prune,tol,FCI_energy,times_max_diag_time, batch_size, embedding_dim)
    print('Final Ground Energy List:',ground_energy_list)

    Final_time = time.time()
    print('Total execution time:',Final_time-Initial_time)


    