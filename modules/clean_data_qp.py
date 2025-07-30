import numpy as np
import pandas as pd
import os

#path='../quantum_package/cisd_vectors_and_coeff/'

def clean(path_determinants, path_coefficients,prune=1e-10):
    #document_name = data_determinants
    #data = path + document_name
    #read data, without 3 lines of header (this is specific to quantum package output)
    data = pd.read_csv(path_determinants, sep='\t', header=None, skiprows=2)
    coefs = pd.read_csv(path_coefficients, sep='\t', header=None, skiprows=2)

    #if path exist 
    deleted_path=False
    if os.path.exists('deleted_dets.txt'):
        deleted_path=True
    
    #convert heach row from decimal to binary
    data = data.map(lambda x: bin(int(x))[2:])

    #get the length of the longest binary string
    max_len = data.map(lambda x: len(x)).max().max()

    #reverse the strings to make them easier to read, this is 63 -> [1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    data = data.map(lambda x: x[::-1])
    
    #pad at the end of each string with 0s to make them all the same length
    data = data.map(lambda x: x + '0' * (max_len - len(x)))

    #convert each string to a list of integers
    data = data.map(lambda x: [int(i) for i in x])

    #convert each list of integers to a numpy array
    data = data.map(lambda x: np.array(x))
    #datos pares e impares
    data_alphas=data.iloc[::2]
    data_betas=data.iloc[1::2]

    #add the alphas and betas to the same vector
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



    until_pruned_count  = 0
    print('number of coefficients before pruning',len(coefs))
    for i in range(len(coefs)):
        if abs(coefs.iloc[i,0]**2) > prune:
            until_pruned_count += 1
        else:
            break
    print('number of coefficients after pruning',until_pruned_count)

    deleted_dets=SO_vectors[until_pruned_count:,:]

    #prune the vectors
    SO_vectors = SO_vectors[:until_pruned_count,:]





    #write the deleted determinants to a file, this is to avoid repeating the same determinants during the next iteration in the generation 
    #and avoid diagonalizing the same pruned determinants

    #if exist and is not empty the file 
    if deleted_path and os.stat("deleted_dets.txt").st_size != 0:
        deleted_dets=np.fliplr(deleted_dets.astype(int))    #flip from 111000 to 000111 and convert to decimal
        deleted_dets_dec=[int("".join(map(str, row)), 2) for row in deleted_dets]

        #read the deleted determinants file
        f = open('deleted_dets.txt', 'r')
        lines = f.readlines()
        f.close()
        lines=[int(line.strip()) for line in lines]#convert to int

        #create a mask to delete the determinants that are already in the file and not repeat them
        mask=np.isin(deleted_dets_dec,lines).astype(int)

        deleted_dets_dec=np.array(deleted_dets_dec)
        deleted_dets_dec=deleted_dets_dec[np.where(mask==0)]


        #write the deleted determinants dec to the file
        f = open('deleted_dets.txt', 'a')
        for i in range(len(deleted_dets_dec)):
            f.write(str(deleted_dets_dec[i])+'\n')
        f.close()


    else:
        #create the file with the deleted determinants in decimal
        deleted_dets=np.fliplr(deleted_dets.astype(int))
        deleted_dets_dec=[int("".join(map(str, row)), 2) for row in deleted_dets]

        f = open('deleted_dets.txt', 'w')
        for i in range(len(deleted_dets)):
            f.write(str(deleted_dets_dec[i])+'\n')
        f.close()
    

    #prune the dets file and the coef file, without modifying the header
    f = open(path_determinants, 'r')
    lines = f.readlines()
    f.close()
    prune_after_line = until_pruned_count*2+2   #*2 because there are two lines per determinant and +2 because of the header. 
    lines = lines[:prune_after_line]
    f = open(path_determinants, 'w')
    f.writelines(lines)
    f.close()

    f = open(path_coefficients, 'r')
    lines = f.readlines()
    f.close()
    prune_after_line = until_pruned_count+2   #+2 because of the header.
    lines = lines[:prune_after_line]
    f = open(path_coefficients, 'w')
    f.writelines(lines)
    f.close()

    #create a txt file with the deleted determinants if doenst exist
    if not os.path.exists('deleted_dets.txt'):
        f = open('deleted_dets.txt', 'w')
        f.close()
    

    

    return SO_vectors


if __name__ == "__main__":
    #qp_folder='/folder/to_diagonalize.ezfio/determinants/'
    #path_determinants=qp_folder+'psi_det'
    #path_coefficients=qp_folder+'psi_coef'
    clean(path_determinants, path_coefficients,prune)

