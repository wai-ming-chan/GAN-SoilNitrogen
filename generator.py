import torch.nn as nn
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dataloader import GAN_dataloader
import torch.utils.data as data
from discriminator import Discriminator, Discriminator_CNN, Discriminator_GRU
from predictor import Prediction_GRU
from torchinfo import summary


class LSTMModel_demo(nn.Module):
  def __init__(self,hidden_dim=80, input_dim=8,  n_layers=1, outout_dim=4, batch_size=32, seq_len=5):
    super(LSTMModel_demo, self).__init__()
    self.input_dim = input_dim
    self.n_layers = n_layers
    self.hidden_dim = hidden_dim
    self.bidirectional = True 
    self.num_states = 2 if self.bidirectional else 1
    self.batch_size = batch_size
    self.seq_len = seq_len


    self.rnn1 = nn.GRU(self.input_dim,self.hidden_dim, self.n_layers, batch_first=True, bidirectional=self.bidirectional)
    self.fc1 = nn.Linear(self.num_states * hidden_dim, self.rnn1.hidden_size * 2)
    self.relu1  = nn.ReLU()

    self.rnn2 = nn.GRU(self.fc1.out_features, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=self.bidirectional)
    self.fc2 = nn.Linear(self.num_states * self.rnn2.hidden_size, outout_dim)
    self.relu2  = nn.ReLU()

    if False:
        summary(self, input_size=(self.batch_size, self.seq_len,self.input_dim), device="cpu")       

  def forward(self, input):
    batch_size = input.size(0)
    seq_len = input.size(1)

    h0 = torch.zeros(self.num_states * self.n_layers, batch_size, self.hidden_dim)
    c0 = torch.zeros(self.num_states * self.n_layers, batch_size, self.hidden_dim)

    h1 = torch.zeros(self.num_states * self.n_layers, batch_size, self.rnn2.hidden_size)
    c1 = torch.zeros(self.num_states * self.n_layers, batch_size, self.rnn2.hidden_size)

    out,( _,_) = self.rnn1(input,(h0))
    # out,( _,_) = self.rnn1(input,(h0, c0))

    forward_output, backward_output = out[:, :, :self.hidden_dim] ,  out[:, :, self.hidden_dim:]
    # forward_output, backward_output = torch.split(pred, split_size_or_sections=2, dim=2)
    
    out = self.fc1(out)
    out = self.relu1(out)

    out, (_,_) = self.rnn2(out,(h1))
    # out, (_,_) = self.rnn2(out,(h1,c1))
    out = self.fc2(out)
    out = self.relu2(out)

    # output = nn.Softmax( dim=1)(pred)
    return out

######################################################################################################################
class Generator_GRU(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,dropout_prob, bidirectional, output_size, batch_size, seq_len):
        super(Generator_GRU, self).__init__()
        self.input_dim = input_size
        self.output_dim = output_size
        self.n_layers = num_layers
        self.hidden_dim = hidden_size
        self.bidirectional = bidirectional 
        self.num_states = 2 if self.bidirectional else 1
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.rnn1 = nn.GRU(self.input_dim,self.hidden_dim, self.n_layers, batch_first=True, bidirectional=self.bidirectional)
        self.fc1 = nn.Linear(self.num_states * self.hidden_dim, self.rnn1.hidden_size * 2)
        self.bn1 = nn.BatchNorm1d(self.rnn1.hidden_size * 2)

        self.rnn2 = nn.GRU(self.fc1.out_features, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=self.bidirectional)
        self.fc2 = nn.Linear(self.num_states * self.rnn2.hidden_size, self.output_dim)
        
        self.bn2 = nn.BatchNorm1d(self.output_dim)

        self.tanh = nn.Tanh()
        self.leakReLu = nn.LeakyReLU(0.2)
        self.relu  = nn.ReLU()

        if False:
            summary(self, input_size=(self.batch_size, self.seq_len,self.input_dim), device="cpu")       
    
    def forward(self, input):
        batch_size = input.size(0)
        seq_len = input.size(1)

        h0 = torch.zeros(self.num_states * self.n_layers, batch_size, self.hidden_dim).to(input.device)
        c0 = torch.zeros(self.num_states * self.n_layers, batch_size, self.hidden_dim).to(input.device)

        h1 = torch.zeros(self.num_states * self.n_layers, batch_size, self.rnn2.hidden_size).to(input.device)
        c1 = torch.zeros(self.num_states * self.n_layers, batch_size, self.rnn2.hidden_size).to(input.device)

        out,( _,_) = self.rnn1(input,(h0))
        # out,( _,_) = self.rnn1(input,(h0, c0))

        # forward_output, backward_output = out[:, :, :self.hidden_dim] ,  out[:, :, self.hidden_dim:]
        
        out = self.fc1(out)
        # out = self.bn1(out)
        out = self.tanh(out)
        # out = self.leakReLu(out)


        out, (_,_) = self.rnn2(out,(h1))
        # out, (_,_) = self.rnn2(out,(h1,c1))
        out = self.fc2(out)
        # out = self.bn2(out)
        out = self.tanh(out)        
        # out = self.leakReLu(out)

        # output = nn.Softmax( dim=1)(pred)
        return out

class Generator_GRU_COND(nn.Module):
    def __init__(self,input_size, cond_input_size, hidden_size, num_layers, dropout_prob, bidirectional, output_size, batch_size, seq_len):
        super(Generator_GRU_COND, self).__init__()
        self.input_dim = input_size + cond_input_size
        self.output_dim = output_size - cond_input_size
        self.n_layers = num_layers
        self.hidden_dim = hidden_size
        self.bidirectional = bidirectional 
        self.num_states = 2 if self.bidirectional else 1
        self.batch_size = batch_size
        self.seq_len = seq_len


        self.rnn1 = nn.GRU(self.input_dim,self.hidden_dim, self.n_layers, batch_first=True, bidirectional=self.bidirectional)
        self.fc1 = nn.Linear(self.num_states * self.hidden_dim, self.rnn1.hidden_size * 2)
        self.bn1 = nn.BatchNorm1d(self.rnn1.hidden_size * 2)

        self.rnn2 = nn.GRU(self.fc1.out_features, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=self.bidirectional)
        self.fc2 = nn.Linear(self.num_states * self.rnn2.hidden_size, self.output_dim)
        
        self.bn2 = nn.BatchNorm1d(self.output_dim)

        self.tanh = nn.Tanh()
        self.leakReLu = nn.LeakyReLU(0.2)
        self.relu  = nn.ReLU()

        if False:
            summary(self, input_size=(self.batch_size, self.seq_len,self.input_dim), device="cpu")       
    
    def forward(self, input, embedding):
        batch_size = input.size(0)
        seq_len = input.size(1)
        input_e = torch.cat( (input, embedding), dim=2 )

        h0 = torch.zeros(self.num_states * self.n_layers, batch_size, self.hidden_dim).to(input.device)
        c0 = torch.zeros(self.num_states * self.n_layers, batch_size, self.hidden_dim).to(input.device)

        h1 = torch.zeros(self.num_states * self.n_layers, batch_size, self.rnn2.hidden_size).to(input.device)
        c1 = torch.zeros(self.num_states * self.n_layers, batch_size, self.rnn2.hidden_size).to(input.device)

        out,( _,_) = self.rnn1(input_e,(h0))
        # out,( _,_) = self.rnn1(input,(h0, c0))

        # forward_output, backward_output = out[:, :, :self.hidden_dim] ,  out[:, :, self.hidden_dim:]
        
        out = self.fc1(out)
        # out = self.bn1(out)
        out = self.tanh(out)
        out = self.leakReLu(out)


        out, (_,_) = self.rnn2(out,(h1))
        # out, (_,_) = self.rnn2(out,(h1,c1))
        out = self.fc2(out)
        # out = self.bn2(out)
        out = self.tanh(out)        
        out = self.leakReLu(out)
        out = torch.cat( (embedding, out), dim=2 )
        # output = nn.Softmax( dim=1)(pred)
        return out


# class Generator_GRU(nn.Module):
#     def __init__(self,input_size,hidden_size,num_layers,dropout_prob, bidirectional, output_size):
#         super(Generator_GRU, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.num_layers = num_layers # Number of GRU layers
#         self.dropout_p = dropout_prob
#         self.bidirectional = bidirectional 
#         self.num_states = 2 if self.bidirectional else 1

#         self.dropout = nn.Dropout(p=self.dropout_p)  # Adjust the dropout probability as needed
#         self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

#         self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True) 
#         # , dropout=self.dropout_p, bidirectional=self.bidirectional)
#         self.fc2 = nn.Linear(in_features=self.num_states*self.hidden_size, out_features=4)
#         self.tanh = nn.Tanh()

#     def forward(self, x):
#         batch_size = x.size(0)
#         seq_len = x.size(1)
        
#         # Initialize the hidden and cell states to zero
#         # h0 = torch.zeros(self.num_states * self.num_layers, batch_size, self.hidden_size).to(x.device)
#         # c0 = torch.zeros(self.num_states * self.num_layers, batch_size, self.hidden_size).to(x.device)

#         h0 = torch.zeros(self.num_layers,  self.hidden_size)
#         c0 = torch.zeros(self.num_layers,  self.hidden_size)

#         # out, _ = self.lstm(x, (h0, c0)) # out, _ = self.lstm(x)
#         out, _ = self.gru(x, h0)
#         # out = self.tanh(self.fc2(out))
        
#         return out

######################################################################################################################
class Generator_CNN(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu=1):
        super(Generator_CNN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( in_channels=nz, out_channels=ngf * 8, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d( in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( in_channels=ngf * 2, out_channels=nc, kernel_size=4, stride=2, padding= 1, bias=False),
            nn.Tanh() 
        )

    def forward(self, input):
        return self.main(input)

######################################################################################################################

class Generator_simple(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob, bidirectional, output_size):
        super(Generator_simple, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers # Number of LSTM layers

        self.bidirectional = bidirectional 
        self.num_states = 2 if self.bidirectional else 1
        
        self.dropout_p = dropout_prob
        self.dropout = nn.Dropout(p=self.dropout_p)  # Adjust the dropout probability as needed
        
        self.lstm = nn.LSTM(input_size, self.hidden_size, \
                            num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional,\
                           dropout=self.dropout_p)
        
        self.fc2 = nn.Linear(self.num_states*self.hidden_size, input_size)
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Initialize the hidden and cell states to zero
        h0 = torch.zeros(self.num_states * self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_states * self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0)) # out, _ = self.lstm(x)
        out = self.tanh(self.fc2(out))
        
        return out

######################################################################################################################

class Generator_st(nn.Module):
    def __init__(self, input_size_1, input_size_2, input_size_3, lstm_layers, hidden_size1, hidden_size, output_size):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size1 = hidden_size1
        
        # self.output_size = output_size -
        self.input_size_1 = input_size_1
        self.input_size_2 = input_size_2
        
        self.output_size1 = input_size_1 # init_N, other_N
        self.output_size2 = input_size_2 # day_cos, day_sin
        self.output_size3 = input_size_3 # remaining 5 features + 1 label
        
        self.bidirectional = False
        self.num_lstm_layers = lstm_layers  # Number of LSTM layers
        self.dropout_p = 0.1
        self.dropout = nn.Dropout(p=self.dropout_p)  # Adjust the dropout probability as needed
        
        self.lstm = nn.LSTM(input_size_1 + input_size_2 + input_size_3, hidden_size, \
                            num_lstm_layers=self.num_lstm_layers, batch_first=True, bidirectional=self.bidirectional,\
                           dropout=self.dropout_p)
        
        self.num_states = 2 if self.bidirectional else 1
        self.fc_lstm = nn.Linear(self.num_states*self.hidden_size, self.output_size3)

        self.tanh = nn.Tanh()
        self.fc1a =nn.Linear(self.input_size_1, self.hidden_size1)
        self.fc1b =nn.Linear(self.hidden_size1, self.hidden_size*2)
        self.fc1c =nn.Linear(self.hidden_size*2, self.hidden_size1)
        self.fc1d =nn.Linear(self.hidden_size1, self.output_size1)
        
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Initialize the hidden and cell states to zero
        h0 = torch.zeros(self.num_states * self.num_lstm_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_states * self.num_lstm_layers, batch_size, self.hidden_size).to(x.device)
        out3, _ = self.lstm(x, (h0, c0)) # out3, _ = self.lstm(x)
        out3 = self.tanh(self.fc_lstm(out3))
        
        out1 = self.tanh(self.fc1a(x[:,0,2:2+self.input_size_2])) # init_N, other_N
        out1 = self.tanh(self.fc1b(out1))
        out1 = self.tanh(self.fc1c(out1))
        out1 = self.tanh(self.fc1d(out1))
        out1 = out1.view(x.shape[0], 1, self.output_size1)
        out1 = out1.repeat(1, x.shape[1], 1)

        out2 = x[:,:,0:2] # day_cos, day_sin
        
        out = torch.cat( (out2, out1, out3), dim=2)
        return out

######################################################################################################################

def genNoise(batch_len,lookback, seq_len, input_size_2, input_size_3, device):
        days = torch.arange(0,seq_len).unsqueeze(1).repeat(batch_len,1).reshape(batch_len,-1,1)
        # print(f'[genNoise-line12]:days={days}')
        noise_1a = torch.cos(2 * np.pi * days / 61).to(device)
        noise_1b = torch.sin(2 * np.pi * days / 61).to(device)

        noise_2 = torch.randn([batch_len, 1, input_size_2], device=device).repeat(1,seq_len,1)  # init_N & other_N
        noise_3 = torch.randn([batch_len, seq_len, input_size_3], device=device)  # other 7 features
        noise_combine = torch.cat((noise_1a, noise_1b, noise_2, noise_3), dim=2)

        return noise_combine

######################################################################################################################

def genNoise_reduce(batch_len, seq_len, input_size_2, input_size_3, device):
    days = torch.arange(0,seq_len).unsqueeze(1).repeat(batch_len,1).reshape(batch_len,-1,1)
    # print(f'[genNoise-line12]:days={days}')
    if False:
        noise_1a = torch.cos(2 * np.pi * days / 61).to(device)
        noise_1b = torch.sin(2 * np.pi * days / 61).to(device)

        noise_2 = torch.randn([batch_len, 1, input_size_2], device=device).repeat(1,seq_len,1)  # init_N & other_N
    
        noise_3 = torch.randn([batch_len, seq_len, input_size_3], device=device)  # other 7 features
        noise_combine = torch.cat((noise_1a, noise_1b, noise_2, noise_3), dim=2)
    else:
        noise_3 = torch.randn([batch_len, seq_len, input_size_3], device=device)
        noise_combine = noise_3

    return noise_combine

def genNoise_CNN(batch, z_dim, device):
    noise_z = torch.randn(batch, z_dim, 1, 1).to(device)  # Generate random noise with shape (batch_size, nz, 1, 1)

    return noise_z
######################################################################################################################

def plot_GAN_training(error_output, script_name):
    epochs = [row[0] for row in error_output]
    g_losses = [row[1] for row in error_output]
    d_losses = [row[2] for row in error_output]
    accuracies = [row[3] for row in error_output]
    fps = [row[4] for row in error_output]
    
    # Plot Epoch vs G-loss
    plt.rcParams['font.size'] = 12  # Set the desired font size
    
    plt.figure()
    plt.plot(epochs, d_losses, marker='x', label='D-loss')
    plt.plot(epochs, g_losses, marker='o', label='G-loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('G-loss and D-loss')
    plt.legend()
    plt.grid(True)  # Add grid
    
    # Set line width and marker size for the first plot
    lines = plt.gca().lines
    plt.setp(lines, linewidth=2, markersize=6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.ylim(0, 4)  
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'figs/{script_name}_plot_GAN_loss_{current_time}.png')
    # plt.savefig(f'plot_GAN_loss_{current_time}.fig', format='fig')
    
    # plt.show()
    plt.close()
    ##########################################################################
    
    # Plot Epoch vs acc
    plt.figure()
    plt.plot(epochs, accuracies, marker='o', label='Accuracy')
    plt.plot(epochs, fps, marker='o', label='False Positive rate')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Accuracy and False Positive')
    plt.legend()
    plt.grid(True)  # Add grid
    
    # Set line width and marker size for the first plot
    lines = plt.gca().lines
    plt.setp(lines, linewidth=2, markersize=6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)  
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'figs/{script_name}_plot_GAN_accu_{current_time}.png')
    # plt.savefig(f'plot_GAN_accu_{current_time}.fig', format='fig')
    # plt.show()
    plt.close()

######################################################################################################################
def seq_to_image(X):
    ########### zero-padding to make it square images ###############
    Xy_images = X
    side_length = max(Xy_images.size(1), Xy_images.size(2)) 
    side_length = 2 ** (int(side_length - 1).bit_length())  # Least power of 2 larger than max(b, c)

    # print(f'side length: {side_length} ')
    padded_images = torch.zeros(Xy_images.size(0), side_length, side_length)  # Initialize an empty tensor to hold the padded images
    padded_images[:, :Xy_images.size(1), :Xy_images.size(2)] = Xy_images # Copy the Xy_images into the padded tensor

    padded_images = padded_images.unsqueeze(1)
    # print(f'[predictor-line317] new size:{padded_images.size()} ')
     
    return padded_images

def image_to_seq(image_data, seq_length, seq_numFeatures):
    image_data = image_data.squeeze(1)  # Remove the channel dimension
    # seq_length = image_data.size(0)
    
    # Extract the sequence data from the image
    sequence = image_data[:, :seq_length, :seq_numFeatures]
    
    return sequence

def trainGAN(batch_size, lookback, num_epochs, num_GAN_data, GAN_train_rate, \
        GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_lr, DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_lr, k_step):
    print(f'batch_size:{batch_size}, lookback:{lookback}, num_epochs:{num_epochs}, num_GAN_data:{num_GAN_data}, GAN_train_rate:{GAN_train_rate}')
    gan_dl = GAN_dataloader()    

    # X_train, y_train, X_test, y_test = gan_dl.create_dataset_GAN_slidingWindow_all(lookback,GAN_train_rate)
    X_train, y_train, _,_ = gan_dl.create_dataset_GAN_slidingWindow_all(lookback,GAN_train_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if False:
        print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_train_pair = torch.cat((X_train, y_train),dim=2) # combine feature and label as one entity for GAN training

    ########### zero-padding to make it square images ###############
    padded_images = seq_to_image(X_train_pair)
    print(f'seq-to-image: {X_train_pair.shape} -> {padded_images.shape}')
    #################################################################
    seq_data = image_to_seq(padded_images, seq_length=lookback+1, seq_numFeatures=X_train.size(2)+1).to(device)
    # print(f'lookback:{lookback} | X.shape:{X_train.size(1), X_train.size(2)}')
    print(f'image-to-seq: {padded_images.shape} -> {seq_data.shape}')
    # Compare the matrices element-wise
    are_equal = torch.equal(X_train_pair, seq_data)

    if are_equal:
        print("The matrices have the same values.")
    else:
        print("The matrices have different values.")
        
    if False:
        dataloader = data.DataLoader(data.TensorDataset(X_train_pair, y_train), shuffle=True, batch_size=batch_size)
    else: # true for CNN 
        dataloader = data.DataLoader(data.TensorDataset(padded_images, y_train), shuffle=True, batch_size=batch_size)

    # Set the parameters for the GAN
    cols_feature = gan_dl.get_cols_feature()
    input_size = len(cols_feature) + 1 + 0  # Dimensionality of the input noise
    
    input_size_1 = 2 # day_cos, day_sin
    input_size_2 = 2 # input_size_1 = 2
    input_size_3 = input_size - input_size_1 - input_size_2 # exclude (1) init_N, (2) other_N, (3) day_cos, (4) day_sin
    hidden_size1 = 16  # Number of hidden units in fc1
    
    GEN_bidirectional = False
    DIS_bidirectional = True

    output_size = len(cols_feature)+1  # Dimensionality of the output sequence
    
    # Instantiate the generator and discriminator models
    # generator = Generator_st(input_size_1, input_size_2, input_size_3, hidden_size1, hidden_size, output_size).to(device)
    # Define model dimensions
    nz = 100  # Size of input noise vector
    ngf = 64 #64  # Number of generator filters
    nc = 1    # Number of channels in the generated images
    ndf = 64 # 64  # Number of discriminator filters 
    if False:
        generator = Generator_simple(input_size, GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_bidirectional, output_size).to(device)
        discriminator = Discriminator(output_size, DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_bidirectional).to(device)
    else:
        generator = Generator_CNN(nz, ngf, nc).to(device)
        discriminator = Discriminator_CNN(nz, ngf, nc, ndf).to(device)
        # (output_size, DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_bidirectional).to(device)
   
    # Define the loss functions and optimizers
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=GEN_lr, weight_decay=1e-5, betas = (0.5, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=DIS_lr, weight_decay=1e-5, betas = (0.5, 0.9))

    error_output = []
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, real_data in enumerate(dataloader): # (real_data, _) to discard the target labels
            batch_len,ch_len,seq_len,feature_dim = real_data[0].shape
            # real_data_RGB = (real_data[0]).unsqueeze(1)
            # print(f'real_data_RGB:{real_data_RGB.size()}')
            if False:
                print(f'batch_len,seq_len,feature_dim = {real_data[0].shape}' )
            
            real_labels     = torch.ones(batch_len, 1, device=device)
            real_labels_gen = torch.ones(batch_len, 1, device=device)
            fake_labels     = -torch.zeros(batch_len, 1, device=device)

            #----- Train the discriminator ----------------------------------------------------------------------
            discriminator.zero_grad()
            optimizer_D.zero_grad()
            
            # # Apply label smoothing
            if False:
                delta_smooth = 0.1
                real_labels = torch.FloatTensor(real_labels.size()).uniform_(1.0-delta_smooth, 1.0+delta_smooth).to(device)
                fake_labels = torch.FloatTensor(fake_labels.size()).uniform_(0.0, 0.0+delta_smooth).to(device)

            # train the Discriminator for every epoch
            if epoch % 1 == 0:
                # Generate fake data using the generator
                # noise_z = genNoise_reduce(batch_len, seq_len, input_size_2, input_size, device)
                # noise_z = torch.randn([batch_len, 100,1,1], device=device)
                noise_z = genNoise_CNN(batch_len, nz, device) # torch.randn(batch_size, nz, 1, 1).to(device)  # Generate random noise with shape (batch_size, nz, 1, 1)

                if False:
                    print(f'noise_z:{noise_z.size()}')
                fake_data = generator(noise_z)
                if False:
                    print(f'fake data: {fake_data.size()}');

                # Compute the discriminator loss
                if False:
                    print('real_labels, fake_labels, fake_data: ',real_labels.shape,fake_labels.shap,fake_data.shape)
                
                real_predictions = discriminator( real_data[0].to(device) )
                fake_predictions = discriminator(fake_data.detach())  # Detach fake_data from the generator
                if False:
                    print(f'real_predictions :{real_predictions.size()} | fake_predictions: {fake_predictions.size()} | fake_labels: {fake_labels.shape}')

                # exit(1)

                d_loss_real = criterion( (real_predictions), real_labels)
                d_loss_fake = criterion( (fake_predictions), fake_labels)
                d_loss = (d_loss_real + d_loss_fake)
                if False:
                    print(f'd_loss: {d_loss}')
                # train Discriminator for every ? epoches
                
                # Backpropagate and update discriminator parameters
                d_loss.backward() # (retain_graph=True)
                optimizer_D.step() # optimizer_D.zero_grad()  # Add this line to clear discriminator gradients
            
            if epoch %  k_step == 0:
                #----- Train the generator ----------------------------------------------------------------------
                optimizer_G.zero_grad()
                generator.zero_grad()
                
                # Generate new fake data using the updated generator
                if True:
                    # noise_z = genNoise_reduce(batch_len, seq_len, input_size_2, input_size, device)
                    noise_z = genNoise_CNN(batch_len, nz, device) #
                    fake_data = generator(noise_z)
                    
                # Compute the generator loss
                fake_predictions = discriminator(fake_data)
                outputs = fake_predictions
                mod_vanishing_grad = False
                if mod_vanishing_grad:
                    g_loss = -torch.log( outputs ).mean()
                else:
                    g_loss = criterion( (outputs), real_labels)
                    # g_loss = criterion( (outputs), real_labels_gen)
                
                # Backpropagate and update generator parameters
                g_loss.backward() # (retain_graph=True)
                optimizer_G.step() # optimizer_G.zero_grad()  # Add this line to clear generator gradients
                
            #----------------------------------------------------------------------------------------
            # Print losses and other metrics for monitoring
            if batch_idx == len(dataloader)-1 and (epoch % 5) == 0:
                # Evaluate generator performance
                test_accuracy = 0
                num_test_acc = 100
                false_positives, false_negatives = 0, 0
                true_positives, true_negatives = 0, 0
                with torch.no_grad():
                    for _ in range(num_test_acc):
                        for _, real_data_inner in enumerate(dataloader):
                            real_predictions = discriminator( real_data_inner[0].to(device) )
                            
                            # noise_z = genNoise_reduce(batch_len, seq_len, input_size_2, input_size, device)
                            noise_z = genNoise_CNN(batch_len, nz, device) #
                            generated_fake_data = generator(noise_z)
                            test_fake_output = discriminator(generated_fake_data)
                            
                            true_positives += torch.sum((real_predictions > 0.5).float())
                            true_negatives += torch.sum((test_fake_output < 0.5).float())
                            false_positives += torch.sum((test_fake_output > 0.5).float())
                            false_negatives += torch.sum((real_predictions < 0.5).float())
                accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
                false_positive_rate = false_positives / (false_positives + true_negatives)
                false_negative_rate = false_negatives / (false_negatives + true_positives)

                # accuracy = test_accuracy / num_test_acc
                error_output.append([int(epoch+1), float(g_loss.item()), float(d_loss.item()), float(accuracy.item()), float(false_positive_rate.item()), float(false_negative_rate.item())])
                print('Epoch [{}/{}], Step [{}/{}], G-loss: {:.4f}, D-loss: {:.4f}, D-loss-real: {:.4f}, D-loss-fake: {:.4f}, acc: {:.4f}, FP: {:.3f}, FN: {:.3f}'
                      .format(epoch+1, num_epochs, batch_idx+1, len(dataloader), \
                              g_loss.item(), d_loss.item(),d_loss_real.item(), d_loss_fake.item(), accuracy.item(), false_positive_rate.item(),false_negative_rate.item()))
        # endfor batch_loop
    # endfor training loop
    #----------------------------------------------------------------------------------------
    if True:
        # Generate fake time-series data using the trained generator
        batch_len = num_GAN_data
        # noise_z = genNoise_reduce(batch_len,lookback, seq_len, input_size_2, input_size, device)
        # noise_z = genNoise_reduce(batch_len, seq_len+1, input_size_2, input_size, device)
        # noise_z = genNoise_CNN(batch_size, nz, device) #
        noise_z = genNoise_CNN(batch_len, nz, device) 

        print(f'noise_z:{noise_z.shape} | seq_len={seq_len}')
        
        fake_data = generator(noise_z)
        fake_data = image_to_seq(fake_data, seq_length=lookback+1, seq_numFeatures=X_train.size(2)+1)
        seq_len = lookback
        print('fake_data: ', fake_data.shape)
        
        # Convert tensor to NumPy array
        A = fake_data
        seq_len_fake = seq_len + 1
        print(f'num_GAN_data:{num_GAN_data} | seq_len_fake:{seq_len_fake} ')
        B = A.reshape(num_GAN_data * seq_len_fake, -1) 
        # B = A.view(num_GAN_data * seq_len_fake, -1) 
        print(f'A:{A.shape} | B:{B.shape}')

        site_id = 0 
        # Create a tensor with repeated values
        site_ids = torch.arange(1+site_id*100, num_GAN_data + 1+site_id*100, dtype=torch.int32).unsqueeze(1).repeat(1, seq_len_fake).view(-1,1).to(device)
        days = torch.arange(0,seq_len_fake).unsqueeze(1).repeat( num_GAN_data,1).view(-1,1)
        days_plant = days.to(device)
        B = torch.cat((days_plant,B), dim=1 )
        B = torch.cat((site_ids,B), dim=1 )
        
        # Add column names to the DataFrame
        cols_total = ['site_id', 'plantation_day', 'inorganic_N']
        col_names = cols_total[:2] + cols_feature + cols_total[2:]

        df = pd.DataFrame(B.cpu().detach()) # Create a DataFrame from the array
        df.columns = col_names
        df.to_csv(f'data/fake_data.csv', index=False) # Save the array to a CSV file
        print( f'fake_dataset.csv exported')
    
    # plot_fakedata(site_id, df)
    return error_output


def trainGAN_RNN(batch_size, lookback, num_epochs, num_GAN_data, GAN_train_rate, \
        GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_lr, DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_lr, k_step):
    print(f'batch_size:{batch_size}, lookback:{lookback}, num_epochs:{num_epochs}, num_GAN_data:{num_GAN_data}, GAN_train_rate:{GAN_train_rate}')
    gan_dl = GAN_dataloader()    

    # X_train, y_train, X_test, y_test = gan_dl.create_dataset_GAN_slidingWindow_all(lookback,GAN_train_rate)
    X_train, y_train, _,_ = gan_dl.create_dataset_GAN_slidingWindow_all(lookback,GAN_train_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if False:
        print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_train_pair = torch.cat((X_train, y_train),dim=2) # combine feature and label as one entity for GAN training

    dataloader = data.DataLoader(data.TensorDataset(X_train_pair, y_train), shuffle=True, batch_size=batch_size)
    
    # Set the parameters for the GAN
    cols_feature = gan_dl.get_cols_feature()
    input_size = len(cols_feature) + 1 + 0  # Dimensionality of the input noise
    
    input_size_1 = 2 # day_cos, day_sin
    input_size_2 = 2 # input_size_1 = 2
    input_size_3 = input_size - input_size_1 - input_size_2 # exclude (1) init_N, (2) other_N, (3) day_cos, (4) day_sin
    hidden_size1 = 16  # Number of hidden units in fc1
    
    GEN_bidirectional = True
    DIS_bidirectional = True

    output_size = len(cols_feature)+1  # Dimensionality of the output sequence
    
    # Instantiate the generator and discriminator models
    # generator = Generator_st(input_size_1, input_size_2, input_size_3, hidden_size1, hidden_size, output_size).to(device)
    
    # LSTM-based Generator
    # generator = Generator_simple(input_size, GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_bidirectional, output_size).to(device)
    
    # GRU-based Generator
    generator = Generator_GRU(input_size, GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_bidirectional, output_size).to(device)
    # GRU-based Discriminator
    discriminator = Discriminator_GRU(output_size, DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_bidirectional).to(device)
    
    # LSTM-based Discriminator
    #discriminator = Discriminator(output_size, DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_bidirectional).to(device)
    
    # Define the loss functions and optimizers
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=GEN_lr, weight_decay=1e-5, betas = (0.5, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=DIS_lr, weight_decay=1e-5, betas = (0.5, 0.9))

    error_output = []
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, real_data in enumerate(dataloader): # (real_data, _) to discard the target labels
            batch_len,seq_len,feature_dim = real_data[0].shape
            
            if False:
                print('batch_len,seq_len,feature_dim = ', real_data[0].shape)
            
            real_labels     = torch.ones(batch_len, 1, device=device)
            real_labels_gen = torch.ones(batch_len, 1, device=device)
            fake_labels     = -torch.zeros(batch_len, 1, device=device)
            
            #----- Train the discriminator ----------------------------------------------------------------------
            discriminator.zero_grad()
            optimizer_D.zero_grad()
            
            # # Apply label smoothing
            if False:
                delta_smooth = 0.1
                real_labels = torch.FloatTensor(real_labels.size()).uniform_(1.0-delta_smooth, 1.0+delta_smooth).to(device)
                fake_labels = torch.FloatTensor(fake_labels.size()).uniform_(0.0, 0.0+delta_smooth).to(device)
    
            # Generate fake data using the generator
            noise_z = genNoise_reduce(batch_len, seq_len, input_size_2, input_size, device)
            fake_data = generator(noise_z)
            
            # Compute the discriminator loss
            if False:
                print('real_labels, fake_labels, fake_data: ',real_labels.shape,fake_labels.shap,fake_data.shape)
            
            real_predictions = discriminator( real_data[0].to(device) )
            fake_predictions = discriminator(fake_data.detach())  # Detach fake_data from the generator

            d_loss_real = criterion( (real_predictions), real_labels)
            d_loss_fake = criterion( (fake_predictions), fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            # trained Discriminator for every ? epoches
            if epoch % k_step == 0: 
                # Backpropagate and update discriminator parameters
                d_loss.backward() # (retain_graph=True)
                optimizer_D.step() # optimizer_D.zero_grad()  # Add this line to clear discriminator gradients
            
            #----- Train the generator ----------------------------------------------------------------------
            optimizer_G.zero_grad()
            generator.zero_grad()
            
            # Generate new fake data using the updated generator
            if True:
                noise_z = genNoise_reduce(batch_len, seq_len, input_size_2, input_size, device)
                fake_data = generator(noise_z)
                
            # Compute the generator loss
            fake_predictions = discriminator(fake_data)
            outputs = fake_predictions
            mod_vanishing_grad = False
            if mod_vanishing_grad:
                g_loss = -torch.log( outputs ).mean()
            else:
                g_loss = criterion( (outputs), real_labels)
                # g_loss = criterion( (outputs), real_labels_gen)
            
            # Backpropagate and update generator parameters
            g_loss.backward() # (retain_graph=True)
            optimizer_G.step() # optimizer_G.zero_grad()  # Add this line to clear generator gradients
            
            #----------------------------------------------------------------------------------------
            # Print losses and other metrics for monitoring
            if batch_idx % 20 == 0 and epoch %20 == 0:
                # Evaluate generator performance
                test_accuracy = 0
                num_test_acc = 100
                false_positives, false_negatives = 0, 0
                true_positives, true_negatives = 0, 0
                with torch.no_grad():
                    for _ in range(num_test_acc):
                        for _, real_data_inner in enumerate(dataloader):
                            real_predictions = discriminator( real_data_inner[0].to(device) )
                            
                            noise_z = genNoise_reduce(batch_len, seq_len, input_size_2, input_size, device)
                            generated_fake_data = generator(noise_z)
                            test_fake_output = discriminator(generated_fake_data)
                            
                            true_positives += torch.sum((real_predictions > 0.5).float())
                            true_negatives += torch.sum((test_fake_output < 0.5).float())
                            false_positives += torch.sum((test_fake_output > 0.5).float())
                            false_negatives += torch.sum((real_predictions < 0.5).float())
                accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
                false_positive_rate = false_positives / (false_positives + true_negatives)
                false_negative_rate = false_negatives / (false_negatives + true_positives)

                # accuracy = test_accuracy / num_test_acc
                error_output.append([int(epoch+1), float(g_loss.item()), float(d_loss.item()), float(accuracy.item()), float(false_positive_rate.item()), float(false_negative_rate.item())])
                print('Epoch [{}/{}], Step [{}/{}], G-loss: {:.4f}, D-loss: {:.4f}, D-loss-real: {:.4f}, D-loss-fake: {:.4f}, acc: {:.4f}, FP: {:.3f}, FN: {:.3f}'
                      .format(epoch+1, num_epochs, batch_idx+1, \
                              len(dataloader), g_loss.item(), d_loss.item(),d_loss_real.item(), d_loss_fake.item(), accuracy.item(), false_positive_rate.item(),false_negative_rate.item()))
        # endfor batch_loop
    # endfor training loop
    #----------------------------------------------------------------------------------------
    # Generate fake time-series data using the trained generator
    batch_len = num_GAN_data
    # noise_z = genNoise_reduce(batch_len,lookback, seq_len, input_size_2, input_size, device)
    noise_z = genNoise_reduce(batch_len, seq_len+1, input_size_2, input_size, device)
    print(f'noise_z:{noise_z.shape} | seq_len={seq_len}')
    
    fake_data = generator(noise_z)
    print('fake_data: ', fake_data.shape)
    
    # Convert tensor to NumPy array
    A = fake_data
    seq_len_fake = seq_len + 1
    B = A.view(num_GAN_data * seq_len_fake, -1) 
    print(f'A:{A.shape} | B:{B.shape}')

    site_id = 0 
    # Create a tensor with repeated values
    site_ids = torch.arange(1+site_id*100, num_GAN_data + 1+site_id*100, dtype=torch.int32).unsqueeze(1).repeat(1, seq_len_fake).view(-1,1).to(device)
    days = torch.arange(0,seq_len_fake).unsqueeze(1).repeat( num_GAN_data,1).view(-1,1)
    days_plant = days.to(device)
    B = torch.cat((days_plant,B), dim=1 )
    B = torch.cat((site_ids,B), dim=1 )
    
    # Add column names to the DataFrame
    cols_total = ['site_id', 'plantation_day', 'inorganic_N']
    col_names = cols_total[:2] + cols_feature + cols_total[2:]

    df = pd.DataFrame(B.cpu().detach()) # Create a DataFrame from the array
    df.columns = col_names
    df.to_csv(f'data/fake_data.csv', index=False) # Save the array to a CSV file
    print( f'fake_dataset.csv exported')
    
    # plot_fakedata(site_id, df)
    return error_output


#######################################################################
def trainGPGAN_RNN(batch_size, lookback, num_epochs, num_GAN_data, GAN_train_rate, \
        GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_lr, DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_lr, k_step):
    print(f'batch_size:{batch_size}, lookback:{lookback}, num_epochs:{num_epochs}, num_GAN_data:{num_GAN_data}, GAN_train_rate:{GAN_train_rate}')
    gan_dl = GAN_dataloader()    

    # X_train, y_train, X_test, y_test = gan_dl.create_dataset_GAN_slidingWindow_all(lookback,GAN_train_rate)
    # X_train, y_train, _,_ = gan_dl.create_dataset_GAN_slidingWindow_all(lookback,GAN_train_rate)
    X_train, y_train, X_test, y_test = gan_dl.create_dataset_slidingWindow_all(lookback, GAN_train_rate, False)
    print(f'line 713: X={X_train[1,:,:]}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if False:
        print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
    X_train, y_train = X_train.to(device), y_train.to(device)
    # X_train_pair = torch.cat((X_train, y_train),dim=2) # combine feature and label as one entity for GAN training
    X_train_pair = X_train

    dataloader = data.DataLoader(data.TensorDataset(X_train_pair, y_train), shuffle=True, batch_size=batch_size)
    
    # Set the parameters for the GAN
    cols_feature = gan_dl.get_cols_feature()
    input_size = len(cols_feature) + 1 + 0  # Dimensionality of the input noise
    
    input_size_1 = 2 # day_cos, day_sin
    input_size_2 = 2 # input_size_1 = 2
    input_size_3 = input_size - input_size_1 - input_size_2 # exclude (1) init_N, (2) other_N, (3) day_cos, (4) day_sin
    hidden_size1 = 16  # Number of hidden units in fc1
    
    GEN_bidirectional = True
    DIS_bidirectional = True

    output_size = len(cols_feature)+1  # Dimensionality of the output sequence
    
    # Instantiate the generator and discriminator models
    # generator = Generator_st(input_size_1, input_size_2, input_size_3, hidden_size1, hidden_size, output_size).to(device)
    
    # LSTM-based Generator
    # generator = Generator_simple(input_size, GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_bidirectional, output_size).to(device)
    
    # GRU-based Generator
    generator = Generator_GRU(input_size, GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_bidirectional, output_size).to(device)
    # GRU-based Discriminator
    # discriminator = Discriminator_GRU(output_size, DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_bidirectional).to(device)
    # GRU-based Predictor
    discriminator = Prediction_GRU(len(cols_feature), DIS_num_layer, DIS_dropout_prob).to(device)

    # LSTM-based Discriminator
    #discriminator = Discriminator(output_size, DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_bidirectional).to(device)
    
    # Define the loss functions and optimizers
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=GEN_lr, weight_decay=1e-5, betas = (0.5, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=DIS_lr, weight_decay=1e-5, betas = (0.5, 0.9))

    error_output = []
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, real_data in enumerate(dataloader): # (real_data, _) to discard the target labels
            batch_len,seq_len,feature_dim = real_data[0].shape
            
            if False:
                print('batch_len,seq_len,feature_dim = ', real_data[0].shape)
            
            real_labels     = torch.ones(batch_len, 1, device=device)
            real_labels_gen = torch.ones(batch_len, 1, device=device)
            fake_labels     = -torch.zeros(batch_len, 1, device=device)
            
            #----- Train the discriminator ----------------------------------------------------------------------
            discriminator.zero_grad()
            optimizer_D.zero_grad()
            
            # # Apply label smoothing
            if False:
                delta_smooth = 0.1
                real_labels = torch.FloatTensor(real_labels.size()).uniform_(1.0-delta_smooth, 1.0+delta_smooth).to(device)
                fake_labels = torch.FloatTensor(fake_labels.size()).uniform_(0.0, 0.0+delta_smooth).to(device)
    
            # Generate fake data using the generator
            noise_z = genNoise_reduce(batch_len, seq_len, input_size_2, input_size, device)
            fake_data = generator(noise_z)
            
            # Compute the discriminator loss
            if True:
                print(f'real_labels{real_labels.shape}, fake_labels{fake_labels.shape}, fake_data:{fake_data.shape}')
            
            real_predictions = discriminator( real_data[0].to(device) )
            fake_predictions = discriminator(fake_data.detach())  # Detach fake_data from the generator

            d_loss_real = criterion( (real_predictions), real_labels)
            d_loss_fake = criterion( (fake_predictions), fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            # trained Discriminator for every ? epoches
            if epoch % k_step == 0: 
                # Backpropagate and update discriminator parameters
                d_loss.backward() # (retain_graph=True)
                optimizer_D.step() # optimizer_D.zero_grad()  # Add this line to clear discriminator gradients
            
            #----- Train the generator ----------------------------------------------------------------------
            optimizer_G.zero_grad()
            generator.zero_grad()
            
            # Generate new fake data using the updated generator
            if True:
                noise_z = genNoise_reduce(batch_len, seq_len, input_size_2, input_size, device)
                fake_data = generator(noise_z)
                
            # Compute the generator loss
            fake_predictions = discriminator(fake_data)
            outputs = fake_predictions
            mod_vanishing_grad = False
            if mod_vanishing_grad:
                g_loss = -torch.log( outputs ).mean()
            else:
                g_loss = criterion( (outputs), real_labels)
                # g_loss = criterion( (outputs), real_labels_gen)
            
            # Backpropagate and update generator parameters
            g_loss.backward() # (retain_graph=True)
            optimizer_G.step() # optimizer_G.zero_grad()  # Add this line to clear generator gradients
            
            #----------------------------------------------------------------------------------------
            # Print losses and other metrics for monitoring
            if batch_idx % 20 == 0 and epoch %20 == 0:
                # Evaluate generator performance
                test_accuracy = 0
                num_test_acc = 100
                false_positives, false_negatives = 0, 0
                true_positives, true_negatives = 0, 0
                with torch.no_grad():
                    for _ in range(num_test_acc):
                        for _, real_data_inner in enumerate(dataloader):
                            real_predictions = discriminator( real_data_inner[0].to(device) )
                            
                            noise_z = genNoise_reduce(batch_len, seq_len, input_size_2, input_size, device)
                            generated_fake_data = generator(noise_z)
                            test_fake_output = discriminator(generated_fake_data)
                            
                            true_positives += torch.sum((real_predictions > 0.5).float())
                            true_negatives += torch.sum((test_fake_output < 0.5).float())
                            false_positives += torch.sum((test_fake_output > 0.5).float())
                            false_negatives += torch.sum((real_predictions < 0.5).float())
                accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
                false_positive_rate = false_positives / (false_positives + true_negatives)
                false_negative_rate = false_negatives / (false_negatives + true_positives)

                # accuracy = test_accuracy / num_test_acc
                error_output.append([int(epoch+1), float(g_loss.item()), float(d_loss.item()), float(accuracy.item()), float(false_positive_rate.item()), float(false_negative_rate.item())])
                print('Epoch [{}/{}], Step [{}/{}], G-loss: {:.4f}, D-loss: {:.4f}, D-loss-real: {:.4f}, D-loss-fake: {:.4f}, acc: {:.4f}, FP: {:.3f}, FN: {:.3f}'
                      .format(epoch+1, num_epochs, batch_idx+1, \
                              len(dataloader), g_loss.item(), d_loss.item(),d_loss_real.item(), d_loss_fake.item(), accuracy.item(), false_positive_rate.item(),false_negative_rate.item()))
        # endfor batch_loop
    # endfor training loop
    #----------------------------------------------------------------------------------------
    # Generate fake time-series data using the trained generator
    batch_len = num_GAN_data
    # noise_z = genNoise_reduce(batch_len,lookback, seq_len, input_size_2, input_size, device)
    noise_z = genNoise_reduce(batch_len, seq_len+1, input_size_2, input_size, device)
    print(f'noise_z:{noise_z.shape} | seq_len={seq_len}')
    
    fake_data = generator(noise_z)
    print('fake_data: ', fake_data.shape)
    
    # Convert tensor to NumPy array
    A = fake_data
    seq_len_fake = seq_len + 1
    B = A.view(num_GAN_data * seq_len_fake, -1) 
    print(f'A:{A.shape} | B:{B.shape}')

    site_id = 0 
    # Create a tensor with repeated values
    site_ids = torch.arange(1+site_id*100, num_GAN_data + 1+site_id*100, dtype=torch.int32).unsqueeze(1).repeat(1, seq_len_fake).view(-1,1).to(device)
    days = torch.arange(0,seq_len_fake).unsqueeze(1).repeat( num_GAN_data,1).view(-1,1)
    days_plant = days.to(device)
    B = torch.cat((days_plant,B), dim=1 )
    B = torch.cat((site_ids,B), dim=1 )
    
    # Add column names to the DataFrame
    cols_total = ['site_id', 'plantation_day', 'inorganic_N']
    col_names = cols_total[:2] + cols_feature + cols_total[2:]

    df = pd.DataFrame(B.cpu().detach()) # Create a DataFrame from the array
    df.columns = col_names
    df.to_csv(f'data/fake_data.csv', index=False) # Save the array to a CSV file
    print( f'fake_dataset.csv exported')
    
    # plot_fakedata(site_id, df)
    return error_output