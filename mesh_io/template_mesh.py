class Template_Mesh():
    def __init__(self, cached_mesh_key = 'deformed_area'):
        self.cached_mesh_key = cached_mesh_key
    
    def init_mesh(self):
        pass
    
    def write_to_polyMesh(self):
        pass
    
    def write_to_openfoam_probe_file(self):
        pass
    
    def write_to_write_to_su2Mesh(self):
        pass
    
    # find and return the edge index and face of surface
    #   does not update self attributes if they exist
    # self.surface_edge_index: [num_surface_e, 2]
    # self.surface_faces: [num_surface_f, <unknown>]
    def get_surface_edges_and_faces(self):
        if hasattr(self, 'surface_edge_index') == False: 
            unique_contour_id = self.contour_id_list.flatten().unique()
            is_surface_edge = sum(
                [self.edge_index[:,0] == _ for _ in unique_contour_id] +\
                [self.edge_index[:,1] == _ for _ in unique_contour_id]
            ) > 1 # 1=2-1                                   # [num_e]
            self.surface_edge_index = self.edge_index[is_surface_edge, :]
            
        if hasattr(self, 'surface_edge_faces') == False:
            num_face_edges = self.faces.shape[1]
            unique_contour_id = self.contour_id_list.flatten().unique()
            is_surface_face = 0
            for i in range(self.faces.shape[1]):
                is_surface_face += sum([self.faces[:,i] == _ for _ in unique_contour_id])
            is_surface_face = is_surface_face > (num_face_edges - 1)
            self.surface_faces = self.faces[is_surface_face, :]
            
        return self.surface_edge_index, self.surface_faces
    
    # find and return the faces that contain the specified query vertex
    # query_v_id_list: int, [num_query]
    # faces: int, [num_f, <unknown>], the face list that one wants to query
    # queried_faces_list: list of int tensor, [num_queried_faces, <unknow>] list of [<unknown>]  tensors
    @staticmethod
    def get_connected_faces(query_v_id_list, faces):
        queried_faces_list = []
        for query_v_id in query_v_id_list:
            is_queried_face = 0
            for i in range(faces.shape[1]):
                is_queried_face += (faces[:,i] == query_v_id)
            is_queried_face = is_queried_face > 0
            queried_faces_list.append(faces[is_queried_face, :])
        return queried_faces_list
    
    # find and return the edges that contain the specified query vertex
    # query_v_id_list: int, [num_query]
    # edge_index: int, [num_e, 2], the edge list that one wants to query
    # queried_edge_index: list of int tensor, [num_queried_edges, <unknow>] list of [2]  tensors
    @staticmethod
    def get_connected_edge_index(query_v_id_list, edge_index):
        queried_edge_list = []
        for query_v_id in query_v_id_list:
            is_queried_edge = 0
            for i in range(edge_index.shape[1]):
                is_queried_edge += (edge_index[:,i] == query_v_id)
            is_queried_edge = is_queried_edge > 0
            queried_edge_list.append(edge_index[is_queried_edge, :])
        return queried_edge_list
    
    def parse_mesh_dict(self, to_tensor=False):
        # self.cached_mesh_key = 'deformed_area' by default
        self.v = self.CFD_mesh[self.cached_mesh_key]['points']               # [num_v, v_dim]
        self.v_movable_mask = self.CFD_mesh[self.cached_mesh_key]['movable_mask']  # True for movable, False for fixed, [num_v, v_dim]
        self.edge_set = self.CFD_mesh[self.cached_mesh_key]['edge_set']
        self.edge_index = self.edge_set.get_edge()                   # [num_e, 2]
        self.faces = self.CFD_mesh[self.cached_mesh_key]['faces']            # [num_f, <unknown>]
        if hasattr(self.edge_set, 'neighbor'):
            delattr(self.edge_set, 'neighbor')
        self.num_vertices = self.v.shape[0]
        self.contour_id_list = self.CFD_mesh[self.cached_mesh_key]['airfoil_contour_id_list']
        
        if hasattr(self.CFD_mesh[self.cached_mesh_key], 'boundary_contour_id_list'):
            self.boundary_id_list = self.CFD_mesh[self.cached_mesh_key]['boundary_contour_id_list']
        else:
            # return True if at least one False appears in any movable_mask
            # return False for totally movable vertices
            self.boundary_id_list = self.v_movable_mask.sum(axis=1) < self.v_movable_mask.shape[1]  
        
        if to_tensor == True:
            import torch
            self.v = torch.tensor(self.v)
            self.v_movable_mask = torch.tensor(self.v_movable_mask)
            self.edge_index = torch.tensor(self.edge_index)
            self.faces = torch.tensor(self.faces)
            self.contour_id_list = torch.tensor(self.contour_id_list)
            self.boundary_id_list = torch.tensor(self.boundary_id_list)
        
        return self.v, self.v_movable_mask, \
             self.edge_set, self.edge_index, \
             self.faces, \
             self.num_vertices, \
             self.contour_id_list, self.boundary_id_list