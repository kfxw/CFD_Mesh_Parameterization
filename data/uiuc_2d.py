import torch
import numpy as np
from math import pi, cos, sin, tan, atan, pow, sqrt
from scipy.interpolate import CubicSpline

def linspace(start,stop,np):
    """
    Emulate Matlab linspace
    """
    return [start+(stop-start)*i/(np-1) for i in range(np)]

def uiuc(profile_dir, profile, nPoints):
    # dat files need preprocessing to remove header, replicated points, extra spaces and empty lines
    # make points in clock-wise or anti-clockwise order
    # it should be like:
    # x1 y1
    # x2 y2
    # ...
    profile_raw = open('{}/{}.dat'.format(profile_dir, profile), 'r').readlines()
    upper_num_v = int(profile_raw[1].split('.')[0])
    lower_num_v = int(profile_raw[1].split('.')[1])
    upper = ''.join(profile_raw[3:3+upper_num_v])
    upper.replace('\n', '')
    import re
    upper = np.array(re.split('  | ', upper)) # delimiter: 1st double spaces, 2nd single space
    upper = upper[1:].astype(float).reshape([-1, 2])
    lower = ''.join(profile_raw[3+upper_num_v+1 : 3+upper_num_v+1+lower_num_v])
    lower.replace('\n', '')
    lower = np.array(re.split('  | ', lower)) # delimiter: 1st double spaces, 2nd single space
    lower = lower[1:].astype(float).reshape([-1, 2])

    if upper[1,0] <= upper[0,0]:   # x in ascending order, some airfoils have outlier definition on first two points
        upper_cs = CubicSpline(upper[1:,0], upper[1:,1])
    else:
        upper_cs = CubicSpline(upper[:,0], upper[:,1])
    if lower[1,0] <= lower[0,0]:
        lower_cs = CubicSpline(lower[1:,0], lower[1:,1])
    else:
        lower_cs = CubicSpline(lower[:,0], lower[:,1])

    beta = linspace(0.0,pi,nPoints+1)
    half_cosine_spacing_X = np.array([(0.5*(1.0-cos(xx))) for xx in beta])  # Half cosine based spacing
    upper_cosine_spacing_Y = upper_cs(half_cosine_spacing_X)
    lower_cosine_spacing_Y = lower_cs(half_cosine_spacing_X)
    half_cosine_spacing_X = torch.tensor( np.concatenate([half_cosine_spacing_X, half_cosine_spacing_X[::-1]]) )
    half_cosine_spacing_Y = torch.tensor( np.concatenate([upper_cosine_spacing_Y, lower_cosine_spacing_Y[::-1]]) )

    uniform_spacing_X = np.array(linspace(0.0,1.0,nPoints+1))  # uniform spacing
    upper_uniform_spacing_X = upper_cs(uniform_spacing_X)
    lower_uniform_spacing_X = lower_cs(uniform_spacing_X)
    uniform_spacing_X = torch.tensor( np.concatenate([uniform_spacing_X, uniform_spacing_X[::-1]]) )
    uniform_spacing_Y = torch.tensor( np.concatenate([upper_uniform_spacing_X, lower_uniform_spacing_X[::-1]]) )

    return torch.stack([half_cosine_spacing_X, half_cosine_spacing_Y], dim=1), \
         torch.stack([uniform_spacing_X, uniform_spacing_Y], dim=1)
