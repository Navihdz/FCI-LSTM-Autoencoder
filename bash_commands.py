import subprocess
import pandas as pd



def zip_dets_coefs(ezfio_path):
    '''
        zips the files psi_coef and psi_det in the ezfio folder determinants 
    '''
    dets_ezfio=ezfio_path+'/determinants/'
    #print('Zipping files ....')
    command = "gzip -n " + dets_ezfio + 'psi_coef'
    subprocess.run(command, shell=True)

    command = "gzip -n " + dets_ezfio + 'psi_det'
    subprocess.run(command, shell=True)

def unzip_dets_coefs(ezfio_path):
    '''
        unzips the files psi_coef and psi_det in the determinants folder  in ezfio folder
    '''
    dets_ezfio=ezfio_path+'/determinants/'
    #print('Unzipping files ....')
    command = "gzip -df " + dets_ezfio +'psi_coef.gz'
    subprocess.run(command, shell=True)

    command = "gzip -df " + dets_ezfio +'psi_det.gz'
    subprocess.run(command, shell=True)


def write_det_num(ezfio_path):
    '''
        write the number of determinants in the file n_det in the determinants folder in ezfio folder
    '''
    dets_ezfio=ezfio_path+'/determinants/'
    readed_det_qp = pd.read_csv(dets_ezfio + 'psi_det.gz', sep="\t", header=None,skiprows=2)
    n_det=str(len(readed_det_qp)//2)
    print('Number of dets in QP to diagonalize:',len(readed_det_qp)//2)

    with open(dets_ezfio + 'n_det', 'w') as file:
        file.write(n_det)  # Escribe 'F' como nueva línea

    #Modify the file n_det_qp_edit  with the new number of determinants, apparently if the number of determinants previously diagonalized
    #is greater than the new number of determinants (after pruning and adding new determinants), then the qpsh will give an error because
    #it will try to diagonalize using the old number of determinants
    with open(dets_ezfio + 'n_det_qp_edit', 'w') as file:
        file.write(n_det)




def modify_threshold_davidson(ezfio_path,threshold):
    threshold_file=ezfio_path+'/davidson_keywords/threshold_davidson'

    #convert for ex 1e-6 to '   1.000000000000000e-6'
    threshold_str='   '+ '1.000000000000000e'+str(threshold).split('e')[1]+'\n '

    with open(threshold_file, 'w') as file:
        file.write(threshold_str)

    print('Threshold used for davidson diagonalization:',threshold)




def reset_ezfio(qpsh_path,ezfio_path):
    '''
        resets the ezfio file to avoid errors in the qpsh shell
    '''
    print('Resetting ezfio file....', end='\r')

    path_to_ezfio=ezfio_path+'/../' #eis necessary to go up one level so that qpsh can find the to_diagonalize.ezfio file
    ezfio_name=ezfio_path.split('/')[-1]

    dets_ezfio=ezfio_path+'/determinants/'
    commands = "qp set_file "+path_to_ezfio+ezfio_name+"\nqp reset --all |tee "+dets_ezfio+"qp.out\nqp unset_file "+ezfio_name+"\nexit\n"

    # start the subprocess
    process = subprocess.Popen([qpsh_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # send the commands and close the standard input so the process completes
    stdout, stderr = process.communicate(input=commands.encode())

    '''
    #----------------------show the output in real time---------------------------------------------------------

    # Muestra la salida y errores si los hay
    print("Salida:")
    print(stdout.decode())
    print("Salidas:")
    print(stderr.decode())
    '''

    print('Resetting ezfio file finished')


def scf(qpsh_path,ezfio_path):
    '''
        runs the scf in the qpsh shell
    '''
    print('SCF started....')

    # commands to execute in the interactive shell
    path_to_ezfio=ezfio_path+'/../' #is necessary to go up one level so that qpsh can find the to_diagonalize.ezfio file
    dets_ezfio=ezfio_path+'/determinants/'
    ezfio_name=ezfio_path.split('/')[-1]
    commands = "qp set_file "+path_to_ezfio+ezfio_name+"\nqp run scf |tee "+dets_ezfio+"qp.out\nqp unset_file "+ezfio_name+"\nexit\n"

    # start the subprocess
    process = subprocess.Popen([qpsh_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    
    # send the commands and close the standard input so the process completes
    stdout, stderr = process.communicate(input=commands.encode())

    '''
    #--------------------------------show the output in real time---------------------------------------------------------

    # Muestra la salida y errores si los hay
    print("Salida:")
    print(stdout.decode())
    print("Salidas:")
    print(stderr.decode())

    '''

    print('SCF finished')

def cisd(qpsh_path,ezfio_path):
    '''
        runs the cisd in the qpsh shell
    '''
    print('CISD started....')

    # commands to execute in the interactive shell
    path_to_ezfio=ezfio_path+'/../' 
    dets_ezfio=ezfio_path+'/determinants/'
    ezfio_name=ezfio_path.split('/')[-1]

    commands = "qp set_file "+path_to_ezfio+ezfio_name+"\nqp run cisd |tee "+dets_ezfio+"qp.out\nqp unset_file "+ezfio_name+"\nexit\n"
    process = subprocess.Popen([qpsh_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=commands.encode())

    '''
    ##--------------------------------show the output in real time---------------------------------------------------------

    # Muestra la salida y errores si los hay
    print("Salida:")
    print(stdout.decode())
    print("Salidas:")
    print(stderr.decode())
    '''

    print('CISD finished')




import subprocess
import psutil
import time
import os
import signal


def diagonalization(qpsh_path,ezfio_path,timeout=5):

    '''
    
        runs the diagonalization of the determinants in the qpsh shell to find the new coefficients
    
    print('Diagonalization started....')

    path_to_ezfio=ezfio_path+'/../' 
    dets_ezfio=ezfio_path+'/determinants/'
    ezfio_name=ezfio_path.split('/')[-1]
    commands = "qp set_file "+path_to_ezfio+ezfio_name+"\nqp run diagonalize_h |tee "+dets_ezfio+"qp.out\nqp unset_file "+ezfio_name+"\nexit\n"

    process = subprocess.Popen([qpsh_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=commands.encode())


    '''





    '''
    Runs the diagonalization of the determinants in the qpsh shell to find the new coefficients.
    If the process takes longer than `timeout` seconds, it will be terminated along with its subprocesses.
    '''
    print('Diagonalization started....')
    process_completed = True

    path_to_ezfio = ezfio_path + '/../'
    dets_ezfio = ezfio_path + '/determinants/'
    ezfio_name = ezfio_path.split('/')[-1]
    commands = "qp set_file " + path_to_ezfio + ezfio_name + "\nqp run diagonalize_h |tee " + dets_ezfio + "qp.out\nqp unset_file " + ezfio_name + "\nexit\n"

    process = subprocess.Popen([qpsh_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    try:
        stdout, stderr = process.communicate(input=commands.encode(), timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f'Process exceeded {timeout} seconds. Terminating process and subprocesses...')
        #Send the signal to kill the process and all its subprocesses
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        #stdout, stderr = process.communicate()  # get the output of the process
        process_completed = False
        return process_completed


    finally:
        if process_completed:
            print('Diagonalization process completed successfully.')
        else:
            print('Diagonalization process was terminated due to timeout.')

        return process_completed  # Return True if the process completed successfully, False otherwise

    '''
    ##--------------------------------show the output in real time---------------------------------------------------------
    print("Salida:")
    print(stdout.decode())
    print("Salidas:")
    print(stderr.decode())
    '''


def get_num_mo(ezfio_path):
    '''
        returns the number of molecular orbitals in the file n_mo in the determinants folder in ezfio folder
    '''
    dets_ezfio=ezfio_path+'/ao_basis/'
    with open(dets_ezfio + 'ao_num', 'r') as file:
        n_mo = file.read()

    #acceseible molecular orbital= 2*n_mo (considering alpha and beta)
    return 2*int(n_mo)

if __name__ == 'main':
    zip_dets_coefs(ezfio_path)
    unzip_dets_coefs(ezfio_path)
    write_det_num(ezfio_path)
    process_completed=diagonalization(qpsh_path)
    
