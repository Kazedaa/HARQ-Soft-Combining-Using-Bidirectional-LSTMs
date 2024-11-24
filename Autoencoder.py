import torch

class AutoencoderModel(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_layers):
        super(AutoencoderModel, self).__init__()
        self.hidden_size= hidden_size
        self.num_layers = num_layers
        
        self.bilstm_block1 = torch.nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, bidirectional = True)
        self.dense1 = torch.nn.Linear(2*hidden_size, input_size // 2)
        self.dense2 = torch.nn.Linear(hidden_size, input_size // 2)
        
    def forward(self,x):
      
        x,_ = self.bilstm_block1(x)
        x = torch.nn.functional.relu(x)
        x = self.dense1(x)
        x = torch.nn.functional.relu(x)

        return x
        