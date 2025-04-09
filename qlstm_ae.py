import pennylane as qml
import jax
import jax.numpy as jnp
import flax.linen as nn
from pennylane.templates import RandomLayers
from pennylane.templates import StronglyEntanglingLayers
from pennylane.templates import BasicEntanglerLayers
import jax_dataloader as jdl
import numpy as np
import optax
import pickle

class qlstm_autoencoder(nn.Module):
    seq_lenght:int
    n_qlayers:int
    n_qubits:int
    hidden_size:int
    target_size:int
    
    
    def setup(self):
        self.device=qml.device('default.qubit', wires=self.n_qubits)
        self.weightsf=self.param('weightsf',nn.initializers.xavier_uniform(),(self.n_qlayers, self.n_qubits))
        self.weightsi=self.param('weightsi',nn.initializers.xavier_uniform(),(self.n_qlayers, self.n_qubits))
        self.weightsu=self.param('weightsu',nn.initializers.xavier_uniform(),(self.n_qlayers, self.n_qubits))
        
        self.weightso=self.param('weightso',nn.initializers.xavier_uniform(),(self.n_qlayers, self.n_qubits))
        #self.weightsf=self.param('weightsf',nn.initializers.normal(),(self.n_qlayers, self.n_qubits))
        #self.weightsi=self.param('weightsi',nn.initializers.normal(),(self.n_qlayers, self.n_qubits))
        #self.weightsu=self.param('weightsu',nn.initializers.normal(),(self.n_qlayers, self.n_qubits))
        #self.weightso=self.param('weightso',nn.initializers.normal(),(self.n_qlayers, self.n_qubits))
        self.qnode = self.make_qnode()
    def make_qnode(self):
        @qml.qnode(self.device, interface="jax", diff_method="backprop")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
            pesos = weights[0, :]
            qml.templates.BasicEntanglerLayers(pesos.reshape(1, -1), wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return jax.jit(circuit)
    
    def qlstm_layer(self,x,ae_size, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        ae_size is the embedding dimension in the encoder part and in the decoder part we pass the target size
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
       
        hidden_seq = []
        batch_size=16
        
        h_t = jnp.zeros((batch_size, self.hidden_size))  # hidden state (output)
        c_t = jnp.zeros((batch_size, self.hidden_size)) # cell state
        
        for t in range(self.seq_lenght):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :] #x has shape (batch,seq_len,features)
            # Concatenate input and hidden state
            v_t = jnp.concatenate((h_t, x_t), axis=1)
            #print('el shape de vt',v_t.shape)
            # match qubit dimension
            y_t = nn.Dense(self.n_qubits)(v_t) #Dense gives an output of n_qubits
            #print('el shape de yt',y_t.shape)
            f_t=self.qnode(y_t,self.weightsf)
            f_t=jnp.asarray(f_t)
            f_t = nn.sigmoid(f_t)  # forget block
            #print('el shape de ft antes del dense',f_t.shape)
            f_t=jnp.transpose(f_t)
            f_t=nn.Dense(self.hidden_size)(f_t)
            #print('el shape de f_t',f_t.shape)
            i_t = self.qnode(y_t,self.weightsi)  # input block
            i_t=jnp.asarray(i_t)
            i_t=nn.sigmoid(i_t)
            i_t=jnp.transpose(i_t)
            i_t=nn.Dense(self.hidden_size)(i_t)
            #print('el shape de i_t',i_t.shape)
            g_t = self.qnode(y_t,self.weightsu) # update block
            g_t=jnp.asarray(g_t)
            g_t=jnp.tanh(g_t)
            g_t=jnp.transpose(g_t)
            g_t=nn.Dense(self.hidden_size)(g_t)
            #print('el shape de g_t',g_t.shape)
            o_t = self.qnode(y_t,self.weightso)# output block
            
            o_t=jnp.asarray(o_t)
            o_t=nn.sigmoid(o_t)
            o_t=jnp.transpose(o_t)
            o_t=nn.Dense(self.hidden_size)(o_t)
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * nn.tanh(c_t) #it has size (batch_size, hidden)
            hidden_seq.append(jnp.expand_dims(h_t, axis=0))#we will end with a number of sequences of the size of the window of time 
                                 
        hidden_seq = jnp.concatenate(hidden_seq, axis=0) #(window, batch_size,hidden)
        hidden_seq = hidden_seq.transpose(1, 0, 2)  #(batch_size,window,hidden)
        final_hidden_seq=hidden_seq[:, -1, :]
        #target=nn.Dense(ae_size)(final_hidden_seq)
        
        return final_hidden_seq

    
    @nn.compact
    def __call__(self, x):
        out=self.qlstm_layer(x,self.hidden_size)
        #expand_dims to match the input shape
        out=jnp.expand_dims(out, axis=-1)
        out=self.qlstm_layer(out,self.hidden_size)
        target=nn.Dense(self.seq_lenght)(out)
        target=jnp.expand_dims(target, axis=-1)
        return target
    
def qlstm_initialization(input_size:int):
    """Initialize the QLSTM Autoencoder model and optimizer."""
    n_qlayers=1
    n_qubits=6
    hidden_size=64
    target_size=1
    
    model = qlstm_autoencoder(input_size,n_qlayers, n_qubits, hidden_size, target_size)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((16, input_size,1)) #input must be (batch,seq_len,features)
    params = model.init(rng, dummy_input)
    return model, params
    

def train_model(input_size,train_loader,test_loader,batch,epochs):
    """Train the Quantum Autoencoder model."""
    
    net,params=qlstm_initialization(input_size)
    optimizer=optax.adam(0.01)
    opt_state=optimizer.init(params)
    
    @jax.jit
    def train_step(params,opt_state,inputs,targets):
        def loss_fn(params,inputs,targets):
            preds=net.apply(params,inputs)
            loss = loss = jnp.mean((preds - targets) ** 2)
            #-jnp.mean(targets * jnp.log(preds + 1e-7) + (1 - targets) * jnp.log(1 - preds + 1e-7))
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

        