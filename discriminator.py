import torch.nn as nn
import torch




#######################################
# Discriminator Model (GRU based)
# Prediction Model (GRU based)
class Discriminator_GRU(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,dropout_prob, bidirectional, output_size, batch_size, seq_len):
        super(Discriminator_GRU, self).__init__()
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

        self.rnn2 = nn.GRU(self.fc1.out_features, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=self.bidirectional)
        self.fc2 = nn.Linear(self.num_states * self.rnn2.hidden_size, self.output_dim)
        
        self.tanh = nn.Tanh()
        
        self.fc3 = nn.Linear(self.seq_len, self.output_dim)
        
        self.relu  = nn.ReLU()
        self.leakReLu = nn.LeakyReLU(0.2)
        # self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

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

        forward_output, backward_output = out[:, :, :self.hidden_dim] ,  out[:, :, self.hidden_dim:]
        # forward_output, backward_output = torch.split(pred, split_size_or_sections=2, dim=2)
        
        out = self.fc1(out)
        out = self.leakReLu(out)
        # out = self.tanh(out)

        out, (_,_) = self.rnn2(out,(h1))
        # out, (_,_) = self.rnn2(out,(h1,c1))

        out = self.fc2( out )
        out = self.leakReLu(out)
        # out = self.tanh(out)

        out = self.fc3( out.squeeze(dim=-1) )
        # out = self.leakReLu(out)
        out = self.sigmoid(out)
        return out

# class Discriminator_GRU(nn.Module):
#     def __init__(self, input_size, hidden_size, lstm_layers, dropout_prob, bidirectional=True):
#         super(Discriminator_GRU, self).__init__()
        
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = lstm_layers  # Number of LSTM layers
#         self.dropout_p = dropout_prob
#         self.bidirectional = bidirectional
#         self.num_states = 2 if self.bidirectional else 1
        
#         self.dropout = nn.Dropout(p=self.dropout_p)  # Adjust the dropout probability as needed
        
#         # self.lstm = nn.LSTM(input_size, hidden_size, \
#         #                     num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional,\
#         #                    dropout=self.dropout_p)
        

#         self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,\
#                         batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout_p)

#         self.fc   = nn.Linear(self.num_states*self.hidden_size, 1)  # Adjust the linear layer input size to match bidirectional LSTM output
#         self.tanh = nn.Tanh()
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         batch_size = x.size(0)
        
#         # Initialize the hidden and cell states to zero
#         h0 = torch.zeros(self.num_states * self.num_layers, batch_size, self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_states * self.num_layers, batch_size, self.hidden_size).to(x.device)
#         # out, _ = self.lstm(x, (h0, c0))
#         out,_ = self.gru(x, h0)
#         # out = self.dropout(out)
#         out = self.fc(out[:, -1, :])  # Output of the last timestep
#         out = self.sigmoid(out)
#         return out

#######################################
# Discriminator Model (LSTM based)
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers, dropout_prob, bidirectional=True):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = lstm_layers  # Number of LSTM layers
        self.dropout_p = dropout_prob
        self.dropout = nn.Dropout(p=self.dropout_p)  # Adjust the dropout probability as needed
        
        self.lstm = nn.LSTM(input_size, hidden_size, \
                            num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional,\
                           dropout=self.dropout_p)
        
        self.num_states = 2 if self.bidirectional else 1
        self.fc   = nn.Linear(self.num_states*self.hidden_size, 1)  # Adjust the linear layer input size to match bidirectional LSTM output
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize the hidden and cell states to zero
        h0 = torch.zeros(self.num_states * self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_states * self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # Output of the last timestep
        out = self.sigmoid(out)
        return out


#######################################
# Discriminator Model (CNN based)
class Discriminator_CNN(nn.Module):
    def __init__(self, nz, ngf, nc, ndf, ngpu=1):
        super(Discriminator_CNN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()

        )

        # # Add a fully-connected layer for classification
        # self.fc_layer = nn.Sequential(
        #     nn.Linear(ndf *  4*4, out_features=1),  # Adjust the input size accordingly
        #     nn.Tanh()
        # )

    def forward(self, input):
        features = self.main(input)
        # return features
        output = features.view(input.size(0), -1)  # Flatten the features to a 2D tensor
        # output = self.fc_layer(output)
        return output