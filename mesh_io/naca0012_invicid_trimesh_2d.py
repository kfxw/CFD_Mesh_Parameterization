import torch
import numpy as np
import os

from mesh_io.template_mesh import Template_Mesh
from mesh_io.geometry import Edge_Set

point_number = 5233
face_number = 10216

face_start_line = 2
face_end_line = 10218
point_start_line = 10219
point_end_line = 15452
contour_start_line = 15455
contour_end_line = 15655
boundary_start_line = 15657
boundary_end_line = 15707

def read_points(filename):
    lines = open(filename, 'r').readlines()
    points = np.zeros([point_number, 2], dtype=float)
    for i, line in enumerate(lines[point_start_line : point_end_line]):
        line = line.strip().split('\t')
        points[i,0] = float(line[0])
        points[i,1] = float(line[1])
    return points
    
def read_faces(filename):
    lines = open(filename, 'r').readlines()
    faces = np.zeros([face_number, 3], dtype=int)
    for i, line in enumerate(lines[face_start_line : face_end_line]):
        line = line.strip().split('\t')
        faces[i,0] = int(line[1])
        faces[i,1] = int(line[2])
        faces[i,2] = int(line[3])
    return faces

def read_contour_id_list(filename):
    lines = open(filename, 'r').readlines()
    contour_id_list = []
    for line in lines[contour_start_line : contour_end_line]:
        line = line.strip().split('\t')
        contour_id_list.append([int(line[1]), int(line[2])])
    return np.array(contour_id_list, dtype=int)

def read_boundary_id_list(filename):
    lines = open(filename, 'r').readlines()
    boundary_id_list = []
    for line in lines[boundary_start_line : boundary_end_line]:
        line = line.strip().split('\t')
        boundary_id_list.append([int(line[1]), int(line[2])])
    return np.array(boundary_id_list, dtype=int)
    

class Naca0012_Invicid_Trimesh_2d(Template_Mesh):
    def __init__(self, cache_name=None, su2Mesh_dir=None):
        super().__init__()
        self.cache_name = cache_name
        self.su2Mesh_dir = su2Mesh_dir
        
    # init CFD mesh, make or use cache file
    # the structure of cache file:
    # dict
    #   deformed_area
    #     points [num_deformed_pt, 2]
    #     edge_set
    #     faces [num_deformed_faces, 3]
    #     x/y_movable_mask [num_deformed_pt]
    #     airfoil_contour_id_list [num_pt_af, 2]
    #     boundary_contour_id_list [num_pt_b, 2]
    def init_mesh(self, use_cache = True):
        if self.cache_name is None:
            self.cache_name = './mesh_io/naca0012_su2/init_mesh_cache.pymesh'
            
        # if use cache, load cache file directly
        if (use_cache == True) and (os.path.isfile(self.cache_name) == True):
            self.CFD_mesh = torch.load(self.cache_name)
            return self.CFD_mesh
        
        # else not use cache, then init a cache file
        if self.su2Mesh_dir is None:
            raw_file = './mesh_io/naca0012_su2/mesh_NACA0012_inv.su2'
        else:
            raw_file = os.path.join(self.su2Mesh_dir, 'mesh_NACA0012_inv.su2')
        
        points = read_points(raw_file)
        faces = read_faces(raw_file)
        edge_set = Edge_Set()
        for i in range(faces.shape[0]):
            for j in range(faces.shape[1]):
                edge_set.add_edge(faces[i,j], faces[i,j-1])
        contour_id_list = read_contour_id_list(raw_file)
        boundary_id_list = read_boundary_id_list(raw_file)
        
        movable_mask = np.ones(points.shape).astype(bool)
        movable_mask[np.unique(boundary_id_list.flatten()), :] = 0
        
        self.CFD_mesh = dict()
        self.CFD_mesh['deformed_area'] = dict()
        self.CFD_mesh['deformed_area']['points'] = points
        self.CFD_mesh['deformed_area']['faces'] = faces
        self.CFD_mesh['deformed_area']['edge_set'] = edge_set
        self.CFD_mesh['deformed_area']['airfoil_contour_id_list'] = contour_id_list
        self.CFD_mesh['deformed_area']['boundary_contour_id_list'] = boundary_id_list
        self.CFD_mesh['deformed_area']['movable_mask'] = movable_mask
        
        # save and return cache file
        torch.save(self.CFD_mesh, self.cache_name)
        return self.CFD_mesh
        
    # read raw .su2 file and replace the points coordinates with new ones
    # points, [num_pt, 2]
    def write_to_su2Mesh(self, points, output_point_file):
        assert(points.ndim == 2)
        assert(points.shape[0] == point_number)
        
        if self.su2Mesh_dir is None:
            raw_file = './mesh_io/naca0012_su2/mesh_NACA0012_inv.su2'
        else:
            raw_file = os.path.join(self.su2Mesh_dir, 'mesh_NACA0012_inv.su2')
        
        lines = open(raw_file, 'r').readlines()
        for i, j in enumerate(np.arange(point_start_line, point_end_line)):
            lines[j] = '\t{}\t{}\t{}\n'.format(points[i,0], points[i,1], i)
            
        f = open(output_point_file, 'w')
        f.write(''.join(lines))
        f.close()