import torch.nn as nn
import torch
    
class ReLU_SIRENs(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.w_sin = nn.Embedding(num_embeddings=1, embedding_dim=dim)
        self.w_relu = nn.Embedding(num_embeddings=1, embedding_dim=1)

    def forward(self, x):
        index = torch.zeros(1, device=x.device, dtype=int)
        w_sin = self.w_sin(index)[None, :]
        w_relu = self.w_relu(index)[None, :]
        return w_relu * self.relu(x) + torch.sin(w_sin * x)

class DMM_Decoder(nn.Module):
    def __init__(self, v_dim, layer_dim=[512]*8):
        super(DMM_Decoder, self).__init__()
        self.v_dim = v_dim
        self.layers = []
        self.layers.append(nn.utils.weight_norm(nn.Linear(v_dim, layer_dim[0])))
        acti_fun = ReLU_SIRENs 
        self.layers.append(acti_fun(layer_dim[0]))
        for i in range(len(layer_dim) - 1):
            self.layers.append(nn.utils.weight_norm(nn.Linear(layer_dim[i], layer_dim[i+1])))
            self.layers.append(acti_fun(layer_dim[i+1]))
        self.layers = nn.Sequential(*self.layers)
        self.last_hidden_dim = layer_dim[-1]
        self.output_layer_delta = nn.Linear(self.last_hidden_dim, self.v_dim)
        
    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.001)
        self.output_layer_delta.weight.data.fill_(1e-8)
        self.output_layer_delta.bias.data.fill_(0)
        
    def re_init_weights(self):
        self.output_layer_delta.weight.data.fill_(1e-8)
        self.output_layer_delta.bias.data.fill_(0)
        
    # v: [bs, num_v, v_dim]
    def forward(self, v):
        v_ = v.clone()
        feat = self.layers(v_)
        delta = self.output_layer_delta(feat)
        return delta
    
class LSM_Decoder(torch.nn.Module):
    def __init__(self, v_dim, latent_dim=256, layer_dim=[512]*8):
        super(LSM_Decoder, self).__init__()
        self.v_dim = v_dim
        acti_fun = ReLU_SIRENs
        
        self.latent_projection = nn.utils.weight_norm(nn.Linear(latent_dim, layer_dim[0]))
        self.coordinate_projection = nn.utils.weight_norm(nn.Linear(v_dim, layer_dim[0]))
        
        self.layers = []
        self.layers.append(acti_fun(layer_dim[0]))
        for i in range(len(layer_dim) - 1):
            self.layers.append(nn.utils.weight_norm(nn.Linear(layer_dim[i], layer_dim[i+1])))
            self.layers.append(acti_fun(layer_dim[i+1]))
        self.layers = nn.Sequential(*self.layers)
        self.last_hidden_dim = layer_dim[-1]
        
        self.output_layer_delta = nn.utils.weight_norm(nn.Linear(self.last_hidden_dim, self.v_dim)) 
                                                      
    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.001)
        nn.init.xavier_uniform_(self.latent_projection.weight)
        self.latent_projection.bias.data.fill_(0.001)
        self.output_layer_delta.weight.data.fill_(0)
        self.output_layer_delta.bias.data.fill_(0)
        
    # latent_z: [bs, latent_dim]
    # v: [bs, num_v, v_dim]
    def forward(self, latent_z, v):
        bs = v.shape[0]
        # forward latent code projection
        latent_z = latent_z[:, None, :]
        f_latent = self.latent_projection(latent_z)
        f_coord = self.coordinate_projection(v.clone())
        feat = f_latent + f_coord
        # forward stem
        feat = self.layers(feat)
        delta = self.output_layer_delta(feat)
        
        return delta
