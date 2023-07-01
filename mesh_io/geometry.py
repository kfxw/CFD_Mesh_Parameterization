import numpy as np

class Edge_Set():
    def __init__(self):
        self.edge = []
    
    def add_edge(self, p1, p2):
        if (p1 is None) or (p2 is None):
            return
        self.edge.append(np.array([ min(p1,p2), max(p1,p2) ], dtype=int))
        
    def get_edge(self):
        return np.unique(np.array(list(self.edge)), axis=0)
            
    # make a neighborhood np array of size [num_p, max_neighbor_num]
    # -1 means empty neighbors by default
    # at every call it will overwrite the self.neighbor item
    def get_point_neighbor(self):
        edges = self.get_edge()
        num_v = edges.max() + 1
        neighbor_list = [set() for _ in range(num_v)]
        max_neighbor_num = 0
        for i in range(edges.shape[0]):
            neighbor_list[ edges[i,0] ].add(edges[i,1])
            neighbor_list[ edges[i,1] ].add(edges[i,0])
            max_neighbor_num = len(neighbor_list[ edges[i,0] ]) if len(neighbor_list[edges[i,0]]) > max_neighbor_num else max_neighbor_num
            max_neighbor_num = len(neighbor_list[ edges[i,1] ]) if len(neighbor_list[edges[i,1]]) > max_neighbor_num else max_neighbor_num
        self.neighbor = np.zeros([num_v, max_neighbor_num]) - 1     # -1 for empty element
        for i in range(num_v):
            self.neighbor[i, :len(neighbor_list[i])] = np.array(list(neighbor_list[i]))
        return self.neighbor
            
    def __len__(self):
        return len(self.edge)