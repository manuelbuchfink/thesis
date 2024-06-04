import numpy as np

import torch
import torch.nn as nn



############ Input Positional Encoding ############
class Positional_Encoder(): 
    def __init__(self, params):
        if params['embedding'] == 'gauss':
            
            
            '''
            MAMEL # added standard deviation to B becaue i think their implementation was sigma = 1 only

                    original: 
                    #
                    self.B = torch.empty((params['embedding_size'], params['coordinates_size'])).normal_(mean=0, std=params['standard_deviation'])
            '''
            self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
            self.B = self.B.cuda()
        else:
            raise NotImplementedError

    def embedding(self, x):
        x_embedding = (2. * np.pi * x).to("cuda") @ self.B.t().to("cuda") # @ is dot product, .t() is transpose
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1).to("cuda")
        return x_embedding



############ Fourier Feature Network ############
class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FFN(nn.Module):
    def __init__(self, params):
        super(FFN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out
