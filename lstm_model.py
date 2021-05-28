import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader

class SimpleLSTM(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size, num_layers=1, batch_first=True, bidirectional=False):
        
        super(SimpleLSTM, self).__init__()        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional 
        
        if bidirectional == True:
            num_directions = 2
            self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, batch_first=batch_first, bidirectional=True)
        else:
            num_directions= 1
            self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, batch_first=batch_first, bidirectional=False)
        # Define the output layer
        self.hidden2out = nn.Linear(hidden_dim*num_directions, output_size)

    def forward(self, seq, init_state=None):
        '''
        @param:  seq = input data [flux] 
        @param:  init_state = (h_0,c_0) inital hidden and cell state 
        @output: final_seq = predicted data [flux at t+lag]
        @output: final_state = (h_n, c_n) final hidden and cell state 
        '''       
        lstm_out, final_state = self.lstm(seq, init_state)
        
        final_seq = self.hidden2out(lstm_out)
        
        return final_seq, final_state