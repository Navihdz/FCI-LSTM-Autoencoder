import numpy as np

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