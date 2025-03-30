import pandas as pd
import numpy as np
import os
def clean(path_determinants):
    #read data, without 3 lines of header (this is specific to quantum package output)
    print (path_determinants)

    data = pd.read_csv(path_determinants, sep='\t', header=None, skiprows=2)
   #coefs = pd.read_csv(path_coefficients, sep='\t', header=None, skiprows=2)

    #if path exist
    deleted_path=False
    if os.path.exists('deleted_dets.txt'):
        deleted_path=True
    
    #convert heach row from decimal to binary
    data = data.applymap(lambda x: bin(int(x))[2:])

    #get the length of the longest binary string
    max_len = data.applymap(lambda x: len(x)).max().max()

    #reverse the strings to make them easier to read, this is 63 -> [1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    data = data.applymap(lambda x: x[::-1])
    
    #pad at the end of each string with 0s to make them all the same length
    data = data.applymap(lambda x: x + '0' * (max_len - len(x)))

    #convert each string to a list of integers
    data = data.applymap(lambda x: [int(i) for i in x])

    #convert each list of integers to a numpy array
    data = data.applymap(lambda x: np.array(x))
    #datos pares e impares
    data_alphas=data.iloc[::2]
    data_betas=data.iloc[1::2]

    #unir en una sola lista
    SO_vectors=[]
    for i in range(0,data_alphas.shape[0]):
        vector_so=[]
        for elemento_alpha, elemento_beta in zip(data_alphas.iloc[i,0][:],data_betas.iloc[i,0][:]):
            vector_so.append(elemento_alpha)
            vector_so.append(elemento_beta)
        SO_vectors.append(vector_so)
            
    #convet to numpy array
    SO_vectors=np.array(SO_vectors)
    SO_vectors=SO_vectors.astype(int)
    return SO_vectors

if __name__ == '__main__':
    clean(path_determinants)