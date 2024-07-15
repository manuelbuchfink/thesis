import numpy as np
import torch
import torch.nn as nn



############ Input Positional Encoding ############
class Positional_Encoder(): 
    def __init__(self, params, bb_embedding_size):
        if params['embedding'] == 'gauss':
            
            #self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
            self.B = torch.randn((int(bb_embedding_size / 2), params['coordinates_size'])) * params['scale']
            self.B = self.B.cuda()
        else:
            raise NotImplementedError

    def embedding(self, x):
        x_embedding = (2. * np.pi * x).to("cuda") @ self.B.t().to("cuda") # @ is dot product, .t() is transpose
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1).to("cuda")
        return x_embedding



                    
############ Feed Forward Network ############
class FFN(nn.Module):
    def __init__(self, params, bb_input_dim):
        super(FFN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = bb_input_dim
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

    
###########Projection Layer###################
class Projection(nn.Module):
    def __init__(self, params, input_dim, resize_dim):        
        super(Projection, self).__init__()

        input_dim = input_dim  
        layers = [nn.Linear(input_dim, resize_dim), nn.ReLU()]        

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out