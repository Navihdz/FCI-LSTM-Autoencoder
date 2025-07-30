import sys


import numpy as np
from modules.clean_data_qp import clean 
#import clean_data_qp
#import format_to_qp
from modules.format_to_qp import formatting2qp
import matplotlib.pyplot as plt
import modules.bash_commands as bash_commands
import time
import os
import multiprocessing
#import clean
import modules.lstm_classic as lstm
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
import modules.quantum_autoencoder_amplitude as qautoencoder
from modules.qlstm_ae import train_model, qlstm_initialization
from multiprocessing.pool import ThreadPool
import jax_dataloader as jdl
import gc
import jax



from modules.data_utils import get_and_clean_data, TimeSeriesDataset, repited_determinants
from modules.sampling import generate_batch_probs, det_generation
from modules.visualization import (
    plot_ground_energy, plot_DetDistribution, plot_DetDistribution2, 
    plot_time_per_dets, loss_plot
)

from modules.metrics import hamming_dist





#----------------------------------- Set the random seed for reproducibility-----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

rng = jax.random.PRNGKey(RANDOM_SEED)






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







def FCIWorkflow(working_directory,ezfio_path,qpsh_path,iterations=2,num_epochs=1, learning_rate=0.01,times_num_dets_gen=2, 
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
    Initial_time = time.time()
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
            #ground_energy_list.append(get_energy(ezfio_path,calculation='scf'))
            bash_commands.cisd(qpsh_path,ezfio_path)
            #ground_energy_list.append(get_energy(ezfio_path,calculation='cisd'))

            #initialize the rbm here to not reset the weights in each iteration
            #model=lstm.lstm_initialization(n_mo,embedding_dim)

            #vamos a inicializar el modelo de lstm cuantico
            jax.clear_caches()
            gc.collect()
            bash_commands.unzip_dets_coefs(ezfio_path)
            DataLoader, x_train = get_and_clean_data(ezfio_path, prune, n_mo, batch_size, qlstm=True)
            n_mo=x_train.shape[1]
            autoencoder,params,train_step=qlstm_initialization(n_mo, n_qubits,embedding_dim,learning_rate)
            print('model initialized')
            bash_commands.zip_dets_coefs(ezfio_path)  #zip again the files psi_coef and psi_det in the ezfio folder
    
        
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
            #determinantes_3 =determinantes_2


            #hamming_distance=hamming_dist(determinantes_3, x_train)
            #print('Number of dets with hamming distance >3:',np.count_nonzero(np.array(hamming_distance)>3))

            #hamming_distance=hamming_dist(x_train,x_train)
            #print('Number of dets with hamming distance after concatenate>3:',np.count_nonzero(np.array(hamming_distance)>3))
            

            #join the new determinants with the training set
            new_dets=np.concatenate((x_train,determinantes_3),axis=0)

            #convert the new determinants to qp format and add them to the qp files
            determinantes_3_toqp=np.fliplr(determinantes_3.astype(int))  #format_to_qp function needs the determinants in reverse order ex from 110000 to 000011 to a correct conversion to decimal               
            formatting2qp(determinantes_3_toqp,ezfio_path)

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

            
    
    Final_time = time.time()
    print('Total execution time:',Final_time-Initial_time)
    return ground_energy_list,number_of_det_list,times_per_iteration_list