# ==============================================================================================
# Full Configuration Interaction optimization with LSTM / QLSTM Autoencoders
# Author: Jorge I. Hernandez-Martinez, Sandra Leticia Juarez-Osorio
# CINVESTAV Guadalajara
# Paper: 
# ==============================================================================================


import os
from modules.workflow import FCIWorkflow
from modules.config import Config


def main():
    '''
        Main function to run the script
    '''
    
    #set working directory
    os.chdir(Config.working_directory)

    #run the FCI workflow
    ground_energies,number_of_det_list,times_per_iteration_list=FCIWorkflow(Config.working_directory,Config.ezfio_path,Config.qpsh_path,Config.max_iterations,Config.num_epochs, Config.learning_rate, 
                                                                        Config.times_num_dets_gen,Config.prune,Config.tol,Config.FCI_energy,Config.times_max_diag_time, Config.batch_size, Config.embedding_dim,Config.n_qubits)


    # Imprimir resumen
    print("\n[SUMMARY]")
    print(f"Final Ground Energies: {ground_energies}")
    print(f"Determinants per Iteration: {number_of_det_list}")
    print(f"Times per Iteration: {times_per_iteration_list}")


    
if __name__=='__main__':
    main()



    