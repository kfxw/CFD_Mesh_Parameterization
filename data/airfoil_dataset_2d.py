import torch
import numpy as np

from data.naca_2d import naca
from data.uiuc_2d import uiuc

class AirfoilDataset2D(torch.utils.data.Dataset):
    def __init__(self, airfoil_list, nPoints = 50):
        self.airfoil_list = open(airfoil_list, 'r').readlines()
        self.nPoints = nPoints
        
    def __len__(self):
        return len(self.airfoil_list)
    
    def __getitem__(self, idx):
        target_shape_str = self.airfoil_list[idx].strip()
        
        if '-' in target_shape_str:
            airfoil_pts = uiuc('./data/UIUC_airfoils/dat', target_shape_str, 300)[0]  # [2n+2] points
        else:
            airfoil_x_cos, airfoil_y_cos = naca(target_shape_str, n=300, finite_TE=False, half_cosine_spacing=True)
            airfoil_pts = np.stack([airfoil_x_cos, airfoil_y_cos], axis=1)         # [2n+1] points
            airfoil_pts = np.concatenate([airfoil_pts, airfoil_pts[0:1, :]], axis=0)
            
        if type(airfoil_pts) == np.ndarray:
            airfoil_pts = torch.tensor(airfoil_pts)
        return airfoil_pts, idx
    
    def get_item_name(self, idx):
        return self.airfoil_list[idx]