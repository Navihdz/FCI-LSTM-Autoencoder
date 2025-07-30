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
        self.linear_f = nn.Dense(self.hidden_size)
        self.linear_i = nn.Dense(self.hidden_size)
        self.linear_u = nn.Dense(self.hidden_size)
        self.linear_o = nn.Dense(self.hidden_size)
        self.to_qubits = nn.Dense(self.n_qubits)
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
        batch_size=x.shape[0]
        
        h_t = jnp.zeros((batch_size, self.hidden_size))  # hidden state (output)
        c_t = jnp.zeros((batch_size, self.hidden_size)) # cell state
        
        for t in range(self.seq_lenght):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :] #x has shape (batch,seq_len,features)
            # Concatenate input and hidden state
            v_t = jnp.concatenate((h_t, x_t), axis=1)
            # match qubit dimension
            y_t = self.to_qubits(v_t)
            #print('el shape de yt',y_t.shape
            f_t = self.linear_f(nn.sigmoid(jnp.array(self.qnode(y_t, self.weightsf)).T))
            i_t = self.linear_i(nn.sigmoid(jnp.array(self.qnode(y_t, self.weightsi)).T))#self.qnode(y_t,self.weightsi)  # input block
            g_t = self.linear_u(jnp.tanh(jnp.array(self.qnode(y_t, self.weightsu)).T))#self.qnode(y_t,self.weightsg) # update block
            o_t =self.linear_o(nn.sigmoid(jnp.array(self.qnode(y_t, self.weightso)).T)) #self.qnode(y_t,self.weightso)# output block
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
    
def qlstm_initialization(input_size:int,n_qubits,hidden_size,lr):
    """Initialize the QLSTM Autoencoder model and optimizer."""
    n_qlayers=1
    
    target_size=1
    
    model = qlstm_autoencoder(input_size,n_qlayers, n_qubits, hidden_size, target_size)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, input_size,1)) #input must be (batch,seq_len,features)
    params = model.init(rng, dummy_input)

    optimizer=optax.adam(lr)
    opt_state = optimizer.init(params)
    train_step = create_train_step(model, optimizer)
    return model, params, train_step


def create_train_step(net, optimizer):
    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        def loss_fn(params, inputs, targets):
            preds = net.apply(params, inputs)
            loss = jnp.mean((preds - targets) ** 2)
            return loss
        loss, grad = jax.value_and_grad(loss_fn)(params, inputs, targets)
        updates, opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, opt_state
    return train_step



    

def train_model(net, params,train_step,train_loader,epochs,lr=0.01):
    """Train the Quantum Autoencoder model."""
    print("Training the QLSTM Autoencoder...")
    print("Model initialized")
    optimizer=optax.adam(lr)
    opt_state=optimizer.init(params)
    print("Optimizer initialized")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in train_loader:
            
            inputs, targets = data[0], data[1]
            loss, params, opt_state = train_step(params, opt_state, inputs, targets)
            epoch_loss += loss
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch}, Loss: {epoch_loss}")
    save_model(net, params, save_path="weights/model_params.pkl")
    return net,params

def save_model(model, params, save_path="weights/model_params.pkl"):
    """Save both model configuration and parameters."""
    to_save = {
        "params": params,
        "config": {
            "seq_lenght": model.seq_lenght,
            "n_qlayers": model.n_qlayers,
            "n_qubits": model.n_qubits,
            "hidden_size": model.hidden_size,
            "target_size": model.target_size
        }
    }
    with open(save_path, "wb") as f:
        pickle.dump(to_save, f)
    print(f"Model parameters saved to {save_path}")
def load_model(model_class, input_shape, param_path="weights/model_params.pkl"):
    with open(param_path, "rb") as f:
        loaded = pickle.load(f)
    params = loaded["params"]
    config = loaded["config"]

    net = model_class(
        seq_lenght=config["seq_lenght"],
        n_qlayers=config["n_qlayers"],
        n_qubits=config["n_qubits"],
        hidden_size=config["hidden_size"],
        target_size=config["target_size"]
    )
    print(f"Pesos cargados desde {param_path}")
    return net, params