import torch

def batched_chamfer_distance(pt1, pt2, use_squared_loss=True, single_sided_argmin_on_pt2=True, single_sided_argmin_on_pt1=True, epsilon=1e-20):
    if (single_sided_argmin_on_pt2 or single_sided_argmin_on_pt1) == False:
        return 0
    assert(pt1.ndim == pt2.ndim)
    assert(pt1.shape[0] == pt2.shape[0])
    assert(pt1.shape[2] == pt2.shape[2])
    
    bs = pt1.shape[0]
    num_pt1_v = pt1.shape[1]
    num_pt2_v = pt2.shape[1]
    
    squared_dist_matrix = ((pt1[:,:,None,:] - pt2[:,None,:,:]) ** 2).sum(dim=3)  
    loss = 0
    if single_sided_argmin_on_pt2:
        denominator = num_pt1_v + epsilon
        loss += (squared_dist_matrix.min(dim=2)[0] ** 0.5).sum() / denominator
    if single_sided_argmin_on_pt1:
        denominator = num_pt2_v + epsilon
        loss += (squared_dist_matrix.min(dim=1)[0] ** 0.5).sum() / denominator
    return loss / bs    
    
counter = 0
def reg_loss(v, delta_v):
    v_dim = v.shape[-1]
    delta_v_dim = delta_v.shape[-1]
    global counter
    counter_1 = counter % delta_v_dim
    counter_2 = int((counter % (v_dim*delta_v_dim)) / delta_v_dim)
    counter_3 = int((counter % (v_dim*v_dim*delta_v_dim)) / (v_dim*delta_v_dim))
    
    grad_outputs = torch.ones(delta_v.shape[:-1], device=delta_v.device)
    dDv_dv  = torch.autograd.grad(delta_v[:,:,counter_1], v, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
    
    grad_outputs = torch.ones(v.shape[:-1], device=delta_v.device)
    d2Dv_dv2 = torch.autograd.grad(dDv_dv[:,:,counter_2], v, grad_outputs, retain_graph=True,create_graph=True)[0][:,:,counter_3]
    counter += 1
    
    loss = - (d2Dv_dv2 ** 2).mean()
    
    if counter_2 != counter_3:
        loss *= 0.5
    return loss.abs()