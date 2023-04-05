import torch
import torch.nn as nn

class MisOrientation(nn.Module):
    """Misoreintation loss."""
    def __init__(self, args):
        super(MisOrientation, self).__init__()
        #import pdb; pdb.set_trace()
        dist_type = args.dist_type
        act = args.act_loss
        syms_req = args.syms_req
        syms_type = args.syms_type
        
        print(f'Parameters for Training Loss')
        print('+++++++++++++++++++++++++++++++++++++++++')
        print(f'dist_type: {dist_type}  activation:{act}  Symmetry:{syms_req} Symmetry Type:{syms_type}')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++')
        from mat_sci_torch_quats.losses import ActAndLoss, Loss
        from mat_sci_torch_quats.symmetries import hcp_syms, fcc_syms

        if syms_req:
            if syms_type == 'HCP':
                syms = hcp_syms
            elif syms_type == 'FCC':
                syms = fcc_syms
        else:
            syms = None
        
        self.act_loss = ActAndLoss(act, Loss(dist_func=dist_type, syms=syms), quat_dim=1)
            
    def forward(self, sr, hr):
        #import pdb; pdb.set_trace()
        loss = self.act_loss(sr, hr)
        loss = torch.mean(loss)
        
        return loss
