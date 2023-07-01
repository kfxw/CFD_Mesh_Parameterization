import numpy as np
import os
import time
import torch
import sys

from network import LSM_Decoder
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


def do_minimization_epoch(
         epoch_id,
         workspace_dir,

         decoder,
         train_data, test_data, 
         v,
         latent_z_train, 
         unique_surface_id_list,
         boundary_id_list,

         optimizer_decoder,
         optimizer_z_train,
         lr_scheduler_decoder,
         lr_scheduler_z_train,

         train_display=True,
         training_observations=[],
         with_eval=True,
         eval_display=True,
        ):
    device = v.device
    
    #
    # training epoch
    #
    for iter_num, data in enumerate(train_data):
        target_pts, idx = data
        target_pts = target_pts.to(device)
        bs = target_pts.shape[0]
        
        optimizer_decoder.zero_grad()
        optimizer_z_train.zero_grad()
        latent_z_train.train()
        idx_ = torch.tensor(idx, device=device)
        latent_z = latent_z_train(idx_)
        
        decoder.train()
        loss_all = do_minimization_iter(
            decoder,
            v,
            latent_z,
            unique_surface_id_list,
            boundary_id_list,
            target_pts,
            optimizer = [optimizer_decoder, optimizer_z_train],
            lr_scheduler = [lr_scheduler_decoder, lr_scheduler_z_train],
            regular_sampling_ratio = 0.02
           )
        
        if iter_num % 40 == 0:
            print('epoch:{}, iter:{}, chamfer_loss={:.2g}, reg loss={:.2g}, boundary loss={:.2g}'.format(epoch_id, iter_num, *loss_all))
        
        if train_display:
            for i, id in enumerate(idx):
                if id in training_observations:
                    decoder.eval()
                    v_ = v.repeat(bs, 1, 1)
                    id = torch.tensor(id, device=device)
                    latent_z = latent_z_train(id).reshape(1, -1)
                    v_ += decoder(latent_z, v_.float())
                    
                    target_shape_str = train_data.dataset.get_item_name(id)
                    plot_airfoil_2d_details(v_[i], edge_index, target_pts[i], '{}/train_epoch_{}_{}'.format(workspace_dir, epoch_id, target_shape_str), visualize_full=False)
                    
    # print latent code statistics
    print('Latent code mean norm:{}'.format(latent_z_train.weight.norm(dim=1).mean().data.cpu().numpy()))
    print('Latent code mean:{}'.format(latent_z_train.weight.mean().data.cpu().numpy()))
        
    # 
    # testing
    #
    if with_eval:
        max_reconstruct_iter = 1001
        for iter_num, data in enumerate(test_data):
            target_pts, idx = data
            target_pts = target_pts.to(device)
            target_shape_str = test_data.dataset.get_item_name(idx).strip()
            bs = target_pts.shape[0]
            assert(bs == 1)
            
            latent_z_dim = latent_z_train.embedding_dim
            latent_z = torch.ones([bs, latent_z_dim]).normal_(mean=0, std=1.0 / latent_z_dim**0.5)
            latent_z = latent_z.to(device)
            latent_z.requires_grad = True
            optimizer_z_test = torch.optim.Adam(params = [latent_z], lr = 1e-4)
            lr_scheduler_z_test = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_z_test, factor = 0.3, patience = 5)
            
            decoder.eval()
            # reconstruct
            for recon_iter_num in range(max_reconstruct_iter):
                optimizer_z_test.zero_grad()
                loss_all = do_minimization_iter(
                    decoder,
                    v,
                    latent_z,
                    unique_surface_id_list,
                    boundary_id_list,
                    target_pts,
                    optimizer = [optimizer_z_test],
                    lr_scheduler = [],
                    regular_sampling_ratio = 0.02
                   )
                loss_chamfer = loss_all[0]
                lr_scheduler_z_test.step(loss_chamfer)
                
                is_reconstruct_converged = loss_chamfer < 5e-3
                if (recon_iter_num % 100 == 0) or is_reconstruct_converged:
                    print(' test epoch:{}, profile:{}, iter:{}, chamfer_loss={:.2g}, avm loss={:.2g}, BC loss={:.2g}'.format(epoch_id, target_shape_str, recon_iter_num, *loss_all))
                    if is_reconstruct_converged:
                        break
                        
            v_ = v.repeat(bs, 1, 1)
            v_ += decoder(latent_z, v_.float())
            plot_airfoil_2d_details(v_, edge_index, target_pts, '{}/test_epoch_{}_{}'.format(workspace_dir, epoch_id, target_shape_str))

#    
# main
#
if __name__ == '__main__':
    #
    # datasets
    #
    train_list_file = './data/train_latent.txt'
    test_list_file = './data/test_latent.txt'
    train_set = torch.utils.data.DataLoader(
        AirfoilDataset2D(train_list_file, nPoints = 600),
        batch_size = 5,
        shuffle=True,
        num_workers=1,
        drop_last=False
    )
    test_set = torch.utils.data.DataLoader(
        AirfoilDataset2D(test_list_file, nPoints = 600),
        batch_size = 1,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )
    num_train = len(AirfoilDataset2D(train_list_file))
    num_test = len(AirfoilDataset2D(test_list_file))
    print('Training dataset size: {}'.format(num_train))

    #
    # latent code
    #
    latent_z_dim = 256
    latent_z_train = torch.nn.Embedding(num_train, latent_z_dim)
    torch.nn.init.xavier_uniform_(latent_z_train.weight.data)

    #
    # network
    #
    decoder = LSM_Decoder(
        v_dim=2, 
        latent_dim=latent_z_dim, 
        layer_dim=[256, 256, 512, 512]
    )
    print('Model info')
    print(decoder)
    decoder.init_weights()

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
    training_observations = [0,320,737,919] # radom IDs

    #
    # init cuda
    #
    use_cuda = True
    if use_cuda:
        unique_contour_id_list = unique_contour_id_list.cuda()
        boundary_id_list = boundary_id_list.cuda()
        decoder = decoder.cuda()
        latent_z_train = latent_z_train.cuda()
        v = v.cuda()
    
    # main
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-trainingEpoch', type=int, default='21')
    parser.add_argument('-workspaceDir', type=str)
    parser.add_argument('-visualize', type=bool, default=False)
    args = parser.parse_args()
    workspace_dir = args.workspaceDir
    num_epoch = args.trainingEpoch
    to_visualize = args.visualize

    #
    # optimizer
    #
    optimizer_decoder = torch.optim.Adam(params = decoder.parameters(), lr = 5e-4)
    optimizer_z_train = torch.optim.Adam(params = latent_z_train.parameters(), lr = 1e-3)
    stepsize = int(0.45 * num_epoch * num_train / train_set.batch_size)
    lr_scheduler_decoder = torch.optim.lr_scheduler.StepLR(optimizer_decoder, stepsize, gamma=0.3)
    lr_scheduler_z_train = torch.optim.lr_scheduler.StepLR(optimizer_z_train, stepsize, gamma=0.3)

    tic = time.time()
    for epoch_id in range(num_epoch):
        do_minimization_epoch(
             epoch_id,
             workspace_dir,
             decoder,
             train_set, test_set, 
             v,
             latent_z_train, 
             unique_contour_id_list,
             boundary_id_list,
    
             optimizer_decoder,
             optimizer_z_train,
             lr_scheduler_decoder,
             lr_scheduler_z_train,
    
             train_display= (to_visualize==True) and (epoch_id!=0) and ((epoch_id%4)==0),
             training_observations=training_observations,
             with_eval= (epoch_id>12) and ((epoch_id%4)==0),
             eval_display=(to_visualize==True)
            )
    toc = time.time()
    print('Training done in {}s'.format(toc - tic))
        
    torch.save({
        'args': args,
        'latent_z_dim': latent_z_dim,
        'profile_info': workspace_dir,
        'model_info': 'decoder:\n{}\n '.format(decoder.__str__()),
        'decoder': decoder.cpu(),
    }, workspace_dir + '/lsm_model.pth')