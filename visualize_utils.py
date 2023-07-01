import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc

def plot_airfoil_2d_details(vertices, edge_index, airfoil_pts, prefix, visualize_full=True, visualize_zoom=True):
    v_np = vertices.data.cpu().numpy().squeeze()
    vx_np = v_np[:,0]
    vy_np = v_np[:,1]
    edge_index_np = edge_index.data.cpu().numpy().squeeze()
    if airfoil_pts is not None:
        airfoil_pts = airfoil_pts.squeeze()
        airfoil_x_np = airfoil_pts[:,0].data.cpu().numpy().squeeze()
        airfoil_y_np = airfoil_pts[:,1].data.cpu().numpy().squeeze()
    
    # matplotlib list of colors: https://matplotlib.org/stable/gallery/color/named_colors.html
    fig = plt.gcf()
    fig.set_size_inches(15,8)
    if airfoil_pts is not None:
        plt.scatter(airfoil_x_np, airfoil_y_np, s=2.75, color='red')
    # plot mesh
    plt.scatter(vx_np, vy_np, s=0.05, color='dimgrey')
    edges = np.stack([vx_np[edge_index_np[:,0]], vy_np[edge_index_np[:,0]], vx_np[edge_index_np[:,1]], vy_np[edge_index_np[:,1]]], axis=1)
    edges = edges.reshape(-1, 2, 2)
    lc = mc.LineCollection(edges, colors='silver', linewidths=0.25)
    axes = plt.gca()
    axes.add_collection(lc)
    if visualize_full == True:
        plt.savefig('{}_full_size.png'.format(prefix), dpi=150)
    
    if visualize_zoom == True:
        fig = plt.gcf()
        fig.set_size_inches(9,9)#(15,8)
        axes = plt.gca()
        axes.set_xlim([-0.3,1.5]) #([-0.05,1.075]) 
        axes.set_ylim([-0.9,0.9]) #([-0.25,0.25])
        plt.savefig('{}_zoom.png'.format(prefix), dpi=150)
    plt.close()
    
def plot_airfoil_2d_optimize_comparison(vertices_1, edge_index_1, vertices_2, edge_index_2, prefix, title=None, visualize_full=False):
    for vertices, edge_index, v_color, e_color in zip(
                                        [vertices_1, vertices_2], \
                                        [edge_index_1, edge_index_2], \
                                        ['dimgrey', 'red'], ['silver', 'red']\
                                    ):
        v_np = vertices.data.cpu().numpy().squeeze()
        vx_np = v_np[:,0]
        vy_np = v_np[:,1]
        edge_index_np = edge_index.data.cpu().numpy().squeeze()
        # matplotlib list of colors: https://matplotlib.org/stable/gallery/color/named_colors.html
        fig = plt.gcf()
        fig.set_size_inches(15,8)
        # plot mesh
        plt.scatter(vx_np, vy_np, s=0.05, color=v_color)
        edges = np.stack([vx_np[edge_index_np[:,0]], vy_np[edge_index_np[:,0]], vx_np[edge_index_np[:,1]], vy_np[edge_index_np[:,1]]], axis=1)
        edges = edges.reshape(-1, 2, 2)
        lc = mc.LineCollection(edges, colors=e_color, linewidths=0.25)
        axes = plt.gca()
        axes.add_collection(lc)
    
    if title is not None:
        plt.suptitle(title)
    if visualize_full == True:
        plt.savefig('{}_full_size.png'.format(prefix), dpi=150)
        
    fig = plt.gcf()
    fig.set_size_inches(15,8)
    axes = plt.gca()
    axes.set_xlim([-0.05,1.075]) 
    axes.set_ylim([-0.5,0.5])#[-0.25,0.25])
    plt.savefig('{}_zoom.png'.format(prefix), dpi=150)
    plt.close()