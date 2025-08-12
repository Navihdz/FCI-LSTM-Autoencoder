class Config:
    # Paths
    working_directory = '/home/ivan/Descargas/Python_Codes_DFT/paper_code_implementation/lstm_fci'
    #Working directory for DGX
    #working_directory = '/raid/home/jorgehdztec/scripts/FCI-LSTM-Autoencoder/'


    ezfio_path = 'QP_examples/h2o/h2o_631g.ezfio'

    qpsh_path = '/home/ivan/Descargas/qp2/bin/qpsh'
    #qpsh path for DGX
    #qpsh_path = '/home/qpackage/bin/qpsh'

    # Parameters
    max_iterations = 2
    num_epochs = 1
    learning_rate = 0.0005
    times_num_dets_gen = 15
    prune = 1e-12
    tol = 1e-5
    batch_size = 64
    embedding_dim = 8
    n_qubits = 2
    times_max_diag_time = 10

    # Reference energy
    FCI_energy = -76.12237  # h2o 6-31g



    #optional parameters
    # Uncomment to use different parameters

    ###########################optional molecules for testing###########################
    #ezfio_path='QP_examples/h2o/h2o_631g.ezfio' #h2o 6-31g
    #ezfio_path='QP_examples/h2o/h2o_ccpvdz.ezfio' #h2o ccpvdz
    #ezfio_path='QP_examples/c2/c2_631g.ezfio' #c2 6-31g
    #ezfio_path='QP_examples/c2/c2_ccpvdz.ezfio' #c2 ccpvdz
    #ezfio_path='QP_examples/n2/n2_631g.ezfio' #n2 6-31g
    #ezfio_path='QP_examples/n2/n2_ccpvdz.ezfio' #n2 ccpvdz


    ########################### Optional FCI energy for testing###########################
    #FCI_energy=-76.12237    #h2o 6-31g
    #FCI_energy=-75.72984    #c2 ccpvdz
    #FCI_energy=-76.24195    #h2o ccpvdz
    #FCI_energy = -75.64418  #c2 6-31g
    #FCI_energy = -109.10842 #n2 6-31g
    #FCI_energy = -109.27834 #n2 ccpvdz
