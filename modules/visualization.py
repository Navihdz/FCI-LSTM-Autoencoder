import matplotlib.pyplot as plt
import numpy as np
import os
from modules import bash_commands


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
    
    posiciones = np.arange(len(frecuencias_1))

    #figure with 300 dpi
    plt.figure(dpi=300)
    plt.figure(figsize=(18, 6))

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
