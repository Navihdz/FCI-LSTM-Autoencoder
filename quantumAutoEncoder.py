import jax
import jax.numpy as jnp
import pennylane as qml
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial
from typing import Callable
import pickle
import os
import numpy as np
import math

class QuantumAutoEncoder(nn.Module):
    input_size: int

    def setup(self):
        # Calcular embedding_dim como la menor potencia de 2 >= input_size
        self.embedding_dim =2* 2 ** math.ceil(math.log2(self.input_size))
        self.n_qubits = int(math.log2(self.embedding_dim))
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.n_qlayers = 2
        
        @qml.qnode(self.dev, interface='jax')
        def circuit(x, w):
            qml.AmplitudeEmbedding(x, wires=range(self.n_qubits), normalize=True)
            qml.templates.BasicEntanglerLayers(w, wires=range(self.n_qubits))
            result=qml.probs(wires=range(self.n_qubits))#[qml.expval(qml.PauliZ(i)) for i in range(nq)]
            return result
        
        
         

        self.qnode = circuit
        self.batched_qnode = jax.vmap(self.qnode, in_axes=(0, None))

        self.weightsEnc = self.param(
            "weightsEnc", nn.initializers.normal(stddev=0.1),
            (self.n_qlayers, self.n_qubits)
        )
        self.weightsDec = self.param(
            "weightsDec", nn.initializers.normal(stddev=0.1),
            (self.n_qlayers, self.n_qubits)
        )

    @nn.compact
    def __call__(self, inputs,train: bool = True):
        # inputs: (batch, input_size) = (16, 8)
        x = nn.Dense(self.embedding_dim)(inputs)         # (16, 32)
        x = nn.relu(x)
        x = self.batched_qnode(x, self.weightsEnc)          # (16, 5)
        #x = jnp.array(x).T
        x = nn.Dense(2 * self.embedding_dim)(x)          # (16, 64)
        x = nn.Dense(self.embedding_dim)(x)              # (16, 32)
        x = nn.relu(x)
        x = self.batched_qnode(x, self.weightsDec)          # (16, 5)
        #x = jnp.array(x).T
        x = nn.Dense(self.input_size)(x)                 # (16, 8)
        x=nn.sigmoid(x)
        return x

    
def qAE_initialization(input_size:int):
    """Initialize the Quantum Autoencoder model and optimizer."""
    model = QuantumAutoEncoder(input_size=input_size)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, input_size))
    params = model.init(rng, dummy_input)
    return model, params
    

def train_model(train_data,train_loader,test_loader,batch,epochs):
    """Train the Quantum Autoencoder model."""
    
    net,params=qAE_initialization(input_size=train_data.shape[1])
    optimizer=optax.adam(0.01)
    opt_state=optimizer.init(params)
    
    @jax.jit
    def train_step(params,opt_state,inputs,targets):
        def loss_fn(params,inputs,targets):
            preds=net.apply(params,inputs)
            loss = -jnp.mean(targets * jnp.log(preds + 1e-7) + (1 - targets) * jnp.log(1 - preds + 1e-7))
            #jax.debug.print(">>> preds mean: {}", jnp.mean(preds))
            return loss
        loss,grad=jax.value_and_grad(loss_fn)(params,inputs,targets)
        updates, opt_state=optimizer.update(grad,opt_state)
        new_params=optax.apply_updates(params,updates)
        return loss, new_params, opt_state
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in train_loader:
            inputs, targets = data[0], data[1]
            loss, params, opt_state = train_step(params, opt_state, inputs, targets)
            epoch_loss += loss
        epoch_loss /= len(train_loader)

        print(f"Epoch {epoch}, Loss: {epoch_loss}")
    


    return net,params

def save_model(model, params, save_path="model_params.pkl"):
    """Save the model parameters to a file."""
    with open(save_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Model parameters saved to {save_path}")

def load_model(model_class, input_shape, param_path="model_params.pkl"):


    net = model_class(input_size=input_shape[1])
    with open(param_path, "rb") as f:
        params = pickle.load(f)
    print(f"Pesos cargados desde {param_path}")
    return net, params
