#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#MODULE  for classic lstm autoencoder
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os




#----------------------------------- Set the random seed for reproducibility-----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


#----------------------------------- Define LSTM Class-----------------------------------
class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return x
  
class Decoder(nn.Module):
  def __init__(self, seq_len,input_dim=64, n_features=8):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((-1,self.seq_len, self.hidden_dim))  
    x = F.normalize(torch.abs(self.output_layer(x)), p=1, dim=1)

    return x 
  

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features,embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

        
    def forward(self, x):    
        x = self.encoder(x)
        x = self.decoder(x)
        #x=self.electron_constriction(x)
        return x
    


def lstm_initialization(n_mo,embedding_dim):
    '''
        Initialize the rbm class

        parameters:
            - n_mo: number of molecular orbitals
            - embedding_dim: number of hidden units in the lstm
    '''
    sequence_length=n_mo
    model = RecurrentAutoencoder(sequence_length, 1,embedding_dim)   #seq_len,ne, n_features,embedding_dim
    model = model.to(device)

    return model


def train_model(model, train_loader,  n_epochs,lr):
    #lambda_e = 1.0    # Ponderación de la penalización de la constricción electrónica
    optimizer = torch.optim.Adam(model.parameters(), lr)#optimizer Adam

    #optimizer ASGD
    #optimizer = torch.optim.ASGD(model.parameters(), lr=lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    #optimizer SGD
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    #nota despues hacer prueba con SGLD
    #optimizer = SGLD(model.parameters(), lr=lr, weight_decay=0.0001)

    criterion = nn.L1Loss(reduction='mean').to(device) #analizies MAE between prediction and target
    #criterion = nn.MSELoss(reduction='mean').to(device) #analizies MSE between prediction and target
    #criterion = nn.BCELoss()    # binary cross entropy loss, ya que las salidas de decoder estan entre 0 y 1 (usamos sigmoid)
    #criterion = nn.BCEWithLogitsLoss().to(device) # binary cross entropy loss, ya que las salidas de decoder estan entre 0 y 1 (usamos sigmoid)

    losses=[]

    for epoch in range(1, n_epochs + 1):
        model = model.train()
        print(f'\rÉpoca: {epoch}', end='', flush=True)
        train_losses = []

        for seq_true in train_loader:
            #print('sequnce shape:',seq_true.shape)
            
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            #print(seq_true[0])
            #print('shape de seq_true:',seq_true[0].shape)
            seq_pred = model(seq_true)

            #loss_recon = criterion(seq_pred, seq_true)
            #loss_electrons = electron_penalty_strict(seq_pred, ne)
            #loss = loss_recon + lambda_e * loss_electrons
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
      
        losses.append(np.mean(train_losses))

        if epoch % 10 == 0:
                #print(f"Recon loss: {loss_recon.item():.4f}, Electron penalty: {loss_electrons.item():.4f}")
                #print(f"Total loss: {loss.item():.4f}")
                print(f"Loss:{losses[-1]:.4f}")

        #losses.append(train_losses)
    loss_plot(losses)
        

    return model.eval(),losses


def loss_plot(losses):
    '''
        Plot the loss function

        parameters:
            - losses: list of losses

        return:
            - plot of loss function
    '''
    plt.figure(dpi=300)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Function")
    print('current directory:',os.getcwd())
    plt.savefig('graphs/loss_plot/loss_function.png', bbox_inches='tight') #esto es para que no las recorte al guardar
    plt.close()


def electron_penalty_strict(output, ne):
    """
    Penaliza si:
    - La suma total de electrones != ne
    - La mitad (pares o impares) no tiene ne/2
    Parámetros:
        output: tensor (batch_size, n_mo, 1)
        ne: número total de electrones
    Devuelve:
        penalización escalar
    """
    batch_size = output.shape[0]

    # Elimina la última dimensión para que output sea (batch_size, n_mo)
    output = output.squeeze(-1)

    # Electrón total
    total_e = output.sum(dim=1)  # shape: (batch_size,)

    # Electrones alfa (índices pares) y beta (impares)
    alpha_e = output[:, ::2].sum(dim=1)  # shape: (batch_size,)
    beta_e  = output[:, 1::2].sum(dim=1)

    # Targets ideales
    ne_tensor = torch.full_like(total_e, fill_value=ne)
    ne_half   = torch.full_like(alpha_e, fill_value=ne // 2)

    # Penalizaciones
    loss_total = F.mse_loss(total_e, ne_tensor)
    loss_alpha = F.mse_loss(alpha_e, ne_half)
    loss_beta  = F.mse_loss(beta_e, ne_half)

    total_penalty = loss_total + loss_alpha + loss_beta
    return total_penalty


def saving_weights(model):
    torch.save(model.state_dict(), 'lstm_autoencoder_weights.pth')
    print('weights saved')

def loading_weights(n_mo, ne):  #esta opcion es para cargar los pesos del encoder y decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentAutoencoder(seq_len=n_mo, n_features=1, embedding_dim=64).to(device)
    model.load_state_dict(torch.load('lstm_autoencoder_weights.pth', map_location=device, weights_only=False))
    #model.eval()
    return model

if __name__ == "__main__":

    lstm_initialization(n_mo,embedding_dim)
    train_model(model, train_loader, n_epochs, lr)
    saving_weights(model)
    loading_weights(n_mo, ne)


    
