import numpy as np
import random
import jax.numpy as jnp
from joblib import Parallel, delayed
import torch
import modules.quantum_autoencoder_amplitude as qautoencoder





def mutate_det_with_prob(visible_probs, dets_train):
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
        output_probs = np.array(output_probs)
        new_dets = Parallel(n_jobs=-1)(
        delayed(mutate_det_with_prob)(output_probs[i], dets_train) for i in range(batch_size))
    

    # Parallel execution (usa todos los núcleos disponibles)
        new_dets = Parallel(n_jobs=-1)(
        delayed(mutate_det_with_prob)(output_probs[i], dets_train) for i in range(batch_size)
    )
    #from joblib import Parallel, delayed
    #import threading
    #print("JAX activo:", threading.active_count())
    return new_dets, output_probs


def det_generation(autoencoder,params,dets_train,dets_list,num_dets, n_mo, quantum=False,qlstm=True):
    m=0 #Counter of determinants generated
    train_dec_set = set(int("".join(map(str, x[::-1])), 2) for x in dets_train) #usar set para hacer la busqueda mas rapida (de O(n) a O(1))

    #set the seed for each multiprocess
    #random.seed(os.getpid())
    set_dets_list=set()

    #run until we have the number of determinants that fulfill the conditions of single and double substitutions
    if num_dets > 2000000:
        batch_size=500000
    else:
        batch_size=num_dets//3

    while len(dets_list) < num_dets:
        new_det_batch,output_probs = generate_batch_probs(autoencoder,params, dets_train, batch_size=batch_size, n_mo=n_mo, quantum=quantum,qlstm=qlstm)
        for det in new_det_batch:
            det_dec = int("".join(map(str, det[::-1])), 2)
            if det_dec not in train_dec_set and det_dec not in set_dets_list:
                dets_list.append(det)
                set_dets_list.add(det_dec)
                print('number of determinants generated : ', len(dets_list),'  of ',num_dets, end='\r')
            if len(dets_list) >= num_dets:
                break

    return dets_list


    
