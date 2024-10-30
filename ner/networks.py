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

    # def embedding(self, x):
    #     x_embedding = (2. * np.pi * x).to("cuda") @ self.B.t().to("cuda") # @ is dot product, .t() is transpose
    #     x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1).to("cuda")
    #     return x_embedding
    def embedding(self, x):
        #x_embedding = ((2. * np.pi * x).to("cuda") @ self.B.t().to("cuda")).int() # @ is dot product, .t() is transpose
        x_embedding = torch.matmul(((2. * np.pi * x).bfloat16().to("cuda")) ,(self.B.t().bfloat16().to("cuda"))).bfloat16().to("cuda") # @ is dot product, .t() is transpose
        x_embedding = torch.cat([torch.sin(x_embedding).bfloat16(), torch.cos(x_embedding).bfloat16()], dim=-1).bfloat16().to("cuda")
        return x_embedding
############ Input Positional Encoding 3D ############
class Positional_Encoder_3D():
    def __init__(self, params):
        if params['embedding'] == 'gauss':
            #[1, x, y, z, 3] * (3 , 128) = [1, x, y, z, 128]
            self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
            #self.B = torch.randn((int(bb_embedding_size / 2), params['coordinates_size'])) * params['scale']
            self.B = self.B.cuda()
        else:
            raise NotImplementedError
    # def embedding(self, x):
    #     x_embedding = (2. * np.pi * x).to("cuda") @ self.B.t().to("cuda") # @ is dot product, .t() is transpose
    #     x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1).to("cuda")
    #     return x_embedding
    def embedding(self, x):
        #x_embedding = ((2. * np.pi * x).to("cuda") @ self.B.t().to("cuda")).int() # @ is dot product, .t() is transpose
        x_embedding = torch.matmul(((2. * np.pi * x).bfloat16().to("cuda")) ,(self.B.t().bfloat16().to("cuda"))).bfloat16().to("cuda") # @ is dot product, .t() is transpose
        x_embedding = torch.cat([torch.sin(x_embedding).bfloat16(), torch.cos(x_embedding).bfloat16()], dim=-1).bfloat16().to("cuda")
        return x_embedding
    # def embedding(self, x):
    #     x_embedding = (2. * np.pi * x).to(torch.bfloat16).to("cuda") @ self.B.t().to(torch.bfloat16).to("cuda") # @ is dot product, .t() is transpose
    #     x_embedding = torch.cat([torch.sin(x_embedding).to(torch.bfloat16), torch.cos(x_embedding).to(torch.bfloat16)], dim=-1).to("cuda")
    #     return x_embedding
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

############ Feed Forward Network ############
class FFN_3D(nn.Module):
    def __init__(self, params):
        super(FFN_3D, self).__init__()

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
