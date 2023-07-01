import numpy as np
import os
import time
import torch
import sys

from network import LSM_Decoder
from data.naca_2d import naca
from data.uiuc_2d import uiuc
from data.airfoil_dataset_2d import AirfoilDataset2D
from mesh_io.naca0012_invicid_trimesh_2d import Naca0012_Invicid_Trimesh_2d
from visualize_utils import plot_airfoil_2d_details
from loss_utils import batched_chamfer_distance, reg_loss

def do_minimization_iter(
            decoder,
            v,
            latent_z,
            unique_surface_id_list,
            boundary_id_list,
            target_pts,
            optimizer: list,
            lr_scheduler: list,
            regular_sampling_ratio
           ):
    batch_size = target_pts.shape[0] if target_pts is not None else 1
    
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
    
    delta = decoder(latent_z, v_.float())
    
    delta_contour = delta[:, :v_contour.shape[1], :]
    v_contour_deformed = v_contour + delta_contour
    delta_sampled = delta[:, v_contour.shape[1]:v_contour.shape[1] + v_sampled.shape[1], :]

    delta_boundary = delta[:, v_contour.shape[1] + v_sampled.shape[1]:, :]

    loss_chamfer = batched_chamfer_distance(
        v_contour_deformed, target_pts, 
        use_squared_loss=False, 
        single_sided_argmin_on_pt2=True, 
        single_sided_argmin_on_pt1=True
    )
    loss_reg = reg_loss(v_sampled, delta_sampled)
    loss_boundary = ((delta_boundary ** 2).sum(dim=-1) ** 0.5).mean()
    loss_code_reg = torch.mean(latent_z ** 2)

    loss = loss_chamfer + loss_reg + 0.05 * loss_boundary + 1e-4 * loss_code_reg

    #
    # optimization
    #
    loss.backward()
    for optim in optimizer:
        optim.step()
    for lr_sched in lr_scheduler:
        lr_sched.step()

    return loss_chamfer, loss_reg, loss_boundary

def reconstruct_latent_model(decoder, latent_dim, v, target_pts, unique_surface_id_list, boundary_id_list):
    latent_z = torch.ones([1, latent_dim]).normal_(mean=0, std=1.0 / latent_dim**0.5).to(v.device)
    latent_z.requires_grad = True
    optimizer_z_test = torch.optim.Adam(params = [latent_z], lr = 1e-4)
    lr_scheduler_z_test = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_z_test, factor = 0.3, patience = 5)

    decoder.eval()
    max_reconstruct_iter = 1001
    for recon_iter_num in range(max_reconstruct_iter):
        optimizer_z_test.zero_grad()
        loss_all = do_minimization_iter(
            decoder, v, latent_z,
            unique_surface_id_list, 
            boundary_id_list, 
            target_pts,
            optimizer = [optimizer_z_test], 
            lr_scheduler = [],
            regular_sampling_ratio = 0.02
           )
        loss_chamfer = loss_all[0]
        lr_scheduler_z_test.step(loss_chamfer)
        if loss_chamfer < 5e-3:
            break
        if recon_iter_num % 50 == 0:
            print(' iter {}, chamfer distance: {:.5f}, reg loss: {:.4e}'.format(recon_iter_num, loss_chamfer.item(), loss_all[1].item()))
    return latent_z

#    
# main
#
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', type=str, required=True, default='naca')
    parser.add_argument('-profile', type=str, default='3413')
    parser.add_argument('-workspaceDir', type=str, default='exp_dmm_2d/')
    parser.add_argument('-latentModelPath', type=str, default='exp_lsm_2d/lsm_model.pth')
    args = parser.parse_args()
    workspace_dir = args.workspaceDir
    latent_model_path = args.latentModelPath
    
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
    save_dict = torch.load(latent_model_path)
    decoder = save_dict['decoder']
    latent_dim = save_dict['latent_z_dim']

    #
    # init cuda
    #
    use_cuda = True
    if use_cuda:
        unique_contour_id_list = unique_contour_id_list.cuda()
        boundary_id_list = boundary_id_list.cuda()
        decoder = decoder.cuda()
        v = v.cuda()
        
    #
    # get target geometry
    #
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
    
    #
    # parameterize
    #
    print('Parameterize target airfoil with the latent space model...')
    tic = time.time()
    latent_z = reconstruct_latent_model(decoder, latent_dim, v, airfoil_pts, unique_contour_id_list, boundary_id_list)
    toc = time.time()
    print('Done in {}s'.format(toc - tic))
    
    # save latent code
    v_ = v[None,:,:]
    v_ += decoder(latent_z, v_.float())
    plot_airfoil_2d_details(v_, edge_index, airfoil_pts, '{}/{}'.format(workspace_dir, target_shape_str))
    torch.save({
        'latent_z': latent_z.data.cpu(),
        'profile': target_shape_str
    }, workspace_dir + '/latent_z_{}.pth'.format(target_shape_str))