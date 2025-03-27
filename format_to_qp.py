"""
    Convert Binary Determinants to Decimal and Write to Files

    This function takes a numpy array of binary determinants and converts them to decimal. 
    The resulting decimal determinants are then written to the 'psi_det' and 'psi_coef' files 
    within the specified quantum package (qp) folder.

    Parameters:
    - dets (numpy.ndarray): Numpy array containing binary determinants.

    Output:
    - Writes the converted determinants to 'psi_det' and 'psi_coef' files in the qp_folder.


    Note: Make sure to replace 'qp_folder' with the actual path to your quantum package folder.
    """


def formatting2qp(dets,ezfio_path):
    print('converting determinants to qp format and writing to files...')
    # get vectorized dets_alpha and dets_beta
    #dets_alpha = dets[:, 0::2].astype(int)
    #dets_beta = dets[:, 1::2].astype(int)


    dets_alpha = dets[:, 1::2].astype(int)  #antes tenia 0::2 el correcto es 1::2
    dets_beta = dets[:, 0::2].astype(int)   #antes tenia 1::2 el correcto es 0::2 y lo que pasaba era que al escribir los determinantes en el archivo psi_det 
                                            #los escribia en el orden incorrecto, es decir primero los alphas y luego los betas, lo que generaba 
                                            # un error al diagonalizar con energias mas bajas, ya que se podian repetir determinantes ya que yo los
                                            #habia comparado para eliminar los repetidos en el otro orden



    # convert binary determinants to decimal
    decimal_alpha = [int("".join(map(str, row)), 2) for row in dets_alpha]
    decimal_beta = [int("".join(map(str, row)), 2) for row in dets_beta]

    # format decimal determinants
    dets_decimal = [f"{val: >20}    " for pair in zip(decimal_alpha, decimal_beta) for val in pair]

    
    #open file and append new coefficients == 0
    f = open(ezfio_path+'/determinants/'+'psi_coef', 'a')
    for i in range(0,len(dets_decimal)//2):
        f.write(str('%24.15E'%0)+'\n')
    f.close()


    #Modify the second line of the coefficients file
    f = open(ezfio_path+'/determinants/'+'psi_coef', 'r')
    lines = f.readlines()
    f.close()
    numero_coefs=(len(lines)-2)
    spaces=20- len(str(numero_coefs))
    lines[1]=(" "*(spaces)+str(numero_coefs)+' '*20+'1'+ '\n')
    f = open(ezfio_path+'/determinants/'+'psi_coef', 'w')
    f.writelines(lines)
    f.close()

    #open file and append new decimal determinants
    f = open(ezfio_path+'/determinants/'+'psi_det', 'a')
    for i in range(0,len(dets_decimal)):
        f.write(dets_decimal[i]+'\n')
    f.close()

    #Modify the second line of the determinants file
    f = open(ezfio_path+'/determinants/'+'psi_det', 'r')
    lines = f.readlines()
    f.close()
    numero_dets=(len(lines)-2)//2
    numbers_spaces_dets_num=21-len(str(numero_dets))
    lines[1]=(" "*(19)+'1'+' '*(20)+'2'+' '*(numbers_spaces_dets_num)+str(numero_dets)+'\n')
    f = open(ezfio_path+'/determinants/'+'psi_det', 'w')
    f.writelines(lines)
    f.close()

if __name__ == "__main__":
    formatting2qp(dets,ezfio_path)
