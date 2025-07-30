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
        self.embedding_dim = 2 ** math.ceil(math.log2(self.input_size))
        self.n_qubits = int(math.log2(self.embedding_dim))
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.n_qlayers = 3
        
        @qml.qnode(self.dev, interface='jax')
        def circuit(x, w):
            qml.AmplitudeEmbedding(x, wires=range(self.n_qubits), normalize=True)
            qml.templates.StronglyEntanglingLayers(w, wires=range(self.n_qubits))
            result=qml.probs(wires=range(self.n_qubits))#[qml.expval(qml.PauliZ(i)) for i in range(nq)]
            return result
        
        
         

        self.qnode = circuit
        self.batched_qnode = jax.vmap(self.qnode, in_axes=(0, None))

        self.weightsEnc = self.param(
            "weightsEnc", nn.initializers.normal(stddev=0.1),
            (self.n_qlayers, self.n_qubits,3)
        )
        self.weightsDec = self.param(
            "weightsDec", nn.initializers.normal(stddev=0.1),
            (self.n_qlayers, self.n_qubits,3)
        )

    @nn.compact
    def __call__(self, inputs,train: bool = False):
        # inputs: (batch, input_size) = (16, 8)
        x = nn.Dense(self.embedding_dim)(inputs)   
        #if train:
            #x = nn.Dropout(rate=0.1)(x, deterministic=not train)
        #x = nn.relu(x)
        x = self.batched_qnode(x, self.weightsEnc)          # (16, 5)
        #x = jnp.array(x).T
        x = nn.Dense(2 * self.embedding_dim)(x)          # (16, 64)
        x = nn.Dense(self.embedding_dim)(x)              # (16, 32)
        #x = nn.relu(x)
        x = self.batched_qnode(x, self.weightsDec)          # (16, 5)
        #x = jnp.array(x).T
        x = nn.Dense(self.input_size)(x)                 # (batch, input_size(n_mo))
        #absolute and a normalizaation
        x = jnp.abs(x)
        x = x / jnp.sum(x, axis=1, keepdims=True)
        #x=nn.sigmoid(x)
        #softmax to get probabilities normalized
        #x = nn.softmax(x, axis=1)
        return x

    
def qae_initialization(input_size:int):
    """Initialize the Quantum Autoencoder model and optimizer."""
    model = QuantumAutoEncoder(input_size=input_size)
    rng = jax.random.PRNGKey(0)
    dropout_rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.ones((1, input_size))
    params = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_input)
    return model, params,dropout_rng
    

def train_model(model, params,dropout_rng,train_loader,epochs,lr):
    """Train the Quantum Autoencoder model."""
    
    optimizer=optax.adam(lr)
    opt_state=optimizer.init(params)
    
    @jax.jit
    def train_step(params, dropout_rng, opt_state, inputs, targets):
        # Split the dropout key so that each call gets a fresh one.
        dropout_rng, subkey = jax.random.split(dropout_rng)
        
        def loss_fn(params):
            preds = model.apply(params, inputs, train=True, rngs={'dropout': subkey})
            recon_loss = jnp.mean(jnp.abs(preds - targets))
            #l2_reg = 1e-4
            #l2_loss = sum([jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)])
            return recon_loss
        
        loss, grad = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, opt_state, dropout_rng

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in train_loader:
            inputs, targets = data[0], data[1]
            loss, params, opt_state, dropout_rng = train_step(params, dropout_rng, opt_state, inputs, targets)
            epoch_loss += loss
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch}, Loss: {epoch_loss}")
    


    return model,params,losses

def save_model(model, params, save_path="model_params.pkl"):
    """Save the model parameters to a file."""
    with open(save_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Model parameters saved to {save_path}")

def load_model(input_shape, param_path="model_params.pkl"):
    net = QuantumAutoEncoder(input_size=input_shape)
    with open(param_path, "rb") as f:
        params = pickle.load(f)
    #print(f"Pesos cargados desde {param_path}")
    
    return net, params
