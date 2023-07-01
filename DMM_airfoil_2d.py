import numpy as np
import os
import time
import torch
import sys

from network import DMM_Decoder
from data.naca_2d import naca
from data.uiuc_2d import uiuc
from mesh_io.naca0012_invicid_trimesh_2d import Naca0012_Invicid_Trimesh_2d
from loss_utils import batched_chamfer_distance, reg_loss
from visualize_utils import plot_airfoil_2d_details

def do_minimization_iter(
            decoder,
            v,
            unique_surface_id_list,
            boundary_id_list,
            target_pts,
            optimizer,
            regular_sampling_ratio
           ):
    batch_size = target_pts.shape[0] if target_pts is not None else 1
    decoder.train()
    
    v_contour = v[unique_surface_id_list].repeat(batch_size, 1, 1)
    v_boundary = v[boundary_id_list].repeat(batch_size, 1, 1) 
    num_sampled = int(regular_sampling_ratio * v.shape[0])
    rand_id = torch.randint(low=0, high=v.shape[0], device=v.device, size=[num_sampled])
    combined_id = torch.cat([unique_surface_id_list, rand_id])
    uniques, counts = combined_id.unique(return_counts=True)
    rand_id = uniques[counts == 1]
    v_sampled = v[rand_id].repeat(batch_size, 1, 1) 
    v_sampled.requires_grad = True
    v_ = torch.cat([v_contour, v_sampled, v_boundary], dim=1)
    
    delta = decoder(v_.float())
    delta_contour = delta[:, :v_contour.shape[1], :]
    delta_sampled = delta[:, v_contour.shape[1]:v_contour.shape[1] + v_sampled.shape[1], :]
    delta_boundary = delta[:, v_contour.shape[1] + v_sampled.shape[1]:, :]

    v_contour_deformed = v_contour + delta_contour
    loss_chamfer = batched_chamfer_distance(
        v_contour_deformed, target_pts, 
        single_sided_argmin_on_pt2=True, 
        single_sided_argmin_on_pt1=True
    )
    loss_reg = reg_loss(v_sampled, delta_sampled)
    loss_boundary = ((delta_boundary ** 2).sum(dim=-1) ** 0.5).mean()

    loss = loss_chamfer + loss_reg + 0.05 * loss_boundary

    loss.backward()
    optimizer.step()

    return delta_contour, loss_chamfer, loss_reg, loss_boundary

#    
# main
#
if __name__ == '__main__':
    #
    # init grid
    #
    CFD_mesh = Naca0012_Invicid_Trimesh_2d(
        cache_name = './mesh_io/naca0012_su2/init_mesh_cache.pymesh', 
        su2Mesh_dir = './mesh_io/naca0012_su2'
    )
    _ = CFD_mesh.init_mesh(use_cache = True)
    v, _, edge_set, edge_index, faces, num_vertices, contour_id_list, boundary_id_list = \
                                            CFD_mesh.parse_mesh_dict(to_tensor=True)
    v = v[:,:2]
    unique_contour_id_list = contour_id_list.flatten().unique()

    #
    # init model
    #
    decoder = DMM_Decoder(
        v_dim=2, 
        layer_dim=[256, 512]
    )
    print('Model info')
    print(decoder)
    decoder.init_weights()

    #
    # init cuda
    #
    use_cuda = True
    if use_cuda:
        unique_contour_id_list = unique_contour_id_list.cuda()
        boundary_id_list = boundary_id_list.cuda()
        decoder = decoder.cuda()
        v = v.cuda()
    
    # main
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', type=str, required=True, default='naca')
    parser.add_argument('-profile', type=str, default='3413')
    parser.add_argument('-reconIter', type=int, default='600')
    parser.add_argument('-workspaceDir', type=str, default='exp_dmm_2d/')
    parser.add_argument('-dumpModel', type=str, default='True')
    args = parser.parse_args()
    workspace_dir = args.workspaceDir
    dump_model = args.dumpModel
    assert(dump_model in ['True', 'False'])
    dump_model = True if dump_model == 'True'else False
    
    target_shape_str = ''
    if args.type == 'naca':
        target_shape_str = args.profile
        airfoil_x_cos, airfoil_y_cos = naca(target_shape_str, n=300, finite_TE=False, half_cosine_spacing=True)
        airfoil_pts = np.stack([airfoil_x_cos, airfoil_y_cos], axis=1)
    elif args.type == 'uiuc':
        target_shape_str = args.profile
        airfoil_pts = uiuc('./data/UIUC_airfoils/dat', target_shape_str, 300)
    else:
        raise Exception('target shape type is not supported')
        
    airfoil_pts = torch.tensor(airfoil_pts, device=v.device)[None,:]
    plot_airfoil_2d_details(v, edge_index, airfoil_pts, './{}/{}_and_template_mesh'.format(workspace_dir, target_shape_str))
    
    num_recon_iters = args.reconIter
    optimizer = torch.optim.Adam(params = decoder.parameters(), lr = 5e-3)

    print('Reconstruct {}...'.format(target_shape_str))
    tic = time.time()
    for i in range(num_recon_iters):
        optimizer.zero_grad()
        delta_contour, loss_chamfer, loss_reg, loss_boundary = do_minimization_iter(
            decoder,
            v, 
            unique_contour_id_list,
            boundary_id_list,
            airfoil_pts,
            optimizer, 
            regular_sampling_ratio = 0.02
        )
        if i % 20 == 0:
            print(' iter {}, chamfer distance: {:.6f}, reg loss: {:.4e}, boundary loss: {:.4e}'.format(i, loss_chamfer.item(), loss_reg.item(), loss_boundary.item()))
    toc = time.time()
    print('Done in {}s'.format(toc - tic))
        
    decoder.eval()
    v_ = v.repeat(airfoil_pts.shape[0], 1, 1)
    v_ += decoder(v_.float())
    
    plot_airfoil_2d_details(v_, edge_index, airfoil_pts, '{}/{}'.format(workspace_dir, target_shape_str))
    if dump_model == True:
        # save model
        torch.save({
            'args': args,
            'profile_info': workspace_dir,
            'model_info': 'decoder:\n{}\n '.format(decoder.__str__()),
            'decoder': decoder.cpu(),
        }, workspace_dir + '/dmm_{}_model.pth'.format(target_shape_str))        