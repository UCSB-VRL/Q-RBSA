import torch
from mat_sci_torch_quats.quats import rand_quats, outer_prod, rot_dist, scalar_first2last,  scalar_last2first
from mat_sci_torch_quats.rot_dist_approx import RotDistLoss
import torch.nn as nn
import torch.nn.functional as F


def l1(q1,q2):
        """ Basic L1 loss """
        return torch.mean(abs(q1-q2),dim=-1)

def l2(q1,q2):
        """ Basic L2 loss """
        return torch.sqrt(torch.mean((q1-q2)**2,dim=-1))


class Laplacian_Kernel(nn.Module):
    def __init__(self):
        super(Laplacian_Kernel, self).__init__()
        #import pdb; pdb.set_trace()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(4,1,1,1)
        #self.kernel = torch.matmul(k.t(),k).unsqueeze(0).unsqueeze(0).repeat(4,1,1,1,1)

        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda() 

    def conv_gauss(self, img):
        #import pdb; pdb.set_trace()
        n_channels, _,kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        #import pdb; pdb.set_trace()
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = x.permute(3,0,4,1,2)
        delta_x = torch.zeros_like(x)
        symms_count, _, _, _, _ =x.shape
        for i in range(symms_count): 
            delta_x[i, :, :, :, :] = self.laplacian_kernel(x[i])
        return delta_x


class Edge_Loss:
        """ Wrapper for Edge loss. Inclues option for symmetry as well """
        def __init__(self,dist_func,syms=None):
                if dist_func == 'l1':
                    self.dist_func = l1 
                elif dist_func == 'l2':
                    self.dist_func = l2
                elif dist_func == 'rot_dist':
                    self.dist_func = rot_dist
                elif dist_func == 'rot_dist_approx':
                    self.dist_func = RotDistLoss()
                self.syms = syms
                
                self.gradient = Laplacian_Kernel()

                #self.quat_dim = quat_dim
        def __call__(self,q1,q2): 
                #import pdb; pdb.set_trace()
                if self.syms is not None:
                        self.syms = self.syms.cuda()
                        q1_w_syms = outer_prod(q1,self.syms)
                        if q2 is not None: q2 = q2[...,None,:]
                        # find gradients
                        delta_q1_w_syms = self.gradient(q1_w_syms)
                        delta_q2 = self.gradient(q2)

                        dists = self.dist_func(delta_q1_w_syms,delta_q2)
                        dist_min = dists.min(-1)[0]
                        return dist_min                         
                        #return torch.mean(dist_min)
                else:
                        #return torch.mean(self.dist_func(q1,q2))
                        delta_q1 = self.gradient(q1)
                        delta_q2 = self.gradient(q2)
                        return self.dist_func(delta_q1,delta_q2)

        def __str__(self):
                return f'Dist -> dist_func: {self.dist_func}, ' + \
                           f'syms: {self.syms is not None}'


class Loss:
        """ Wrapper for loss. Inclues option for symmetry as well """
        def __init__(self,dist_func,syms=None):
                if dist_func == 'l1':
                    self.dist_func = l1 
                elif dist_func == 'l2':
                    self.dist_func = l2
                elif dist_func == 'rot_dist':
                    self.dist_func = rot_dist
                elif dist_func == 'rot_dist_approx':
                    self.dist_func = RotDistLoss()
                self.syms = syms
                #self.quat_dim = quat_dim
        def __call__(self,q1,q2):                
                if self.syms is not None:
                        self.syms = self.syms.cuda()
                        q1_w_syms = outer_prod(q1,self.syms)
                        if q2 is not None: q2 = q2[...,None,:]
                        dists = self.dist_func(q1_w_syms,q2)
                        dist_min = dists.min(-1)[0]
                        return dist_min                         
                        #return torch.mean(dist_min)
                else:
                        #return torch.mean(self.dist_func(q1,q2))
                        return self.dist_func(q1,q2)

        def __str__(self):
                return f'Dist -> dist_func: {self.dist_func}, ' + \
                           f'syms: {self.syms is not None}'


def tanhc(x):
        """
        Computes tanh(x)/x. For x close to 0, the function is defined, but not
        numerically stable. For values less than eps, a taylor series is used.
        """
        eps = 0.05
        mask = (torch.abs(x) < eps).float()
        # clip x values, to plug into tanh(x)/x
        x_clip = torch.clamp(abs(x),min=eps)
        # taylor series evaluation
        output_ts = 1 - (x**2)/3 + 2*(x**4)/15 - 17*(x**6)/315
        # regular function evaluation for tanh(x)/x
        output_ht = torch.tanh(x_clip)/x_clip
        # use taylor series if x is close to 0, otherwise, use tanh(x)/x
        output = mask*output_ts + (1-mask)*output_ht
        return output


def tanh_act(q):
        """ Scale a vector q such that ||q|| = tanh(||q||) """
        return q*tanhc(torch.norm(q,dim=-1,keepdim=True))
        
def safe_divide_act(q,eps=10**-5):
        """ Scale a vector such that ||q|| ~= 1 """
        return q/(eps+torch.norm(q,dim=-1,keepdim=True))


class ActAndLoss:
        """ Wraps together activation and loss """
        def __init__(self,act,loss, quat_dim=-1):
                self.act = act
                self.loss = loss
                self.quat_dim = quat_dim
        def __call__(self,X,labels):
                # change to [b, ch, h, w] to [b, h, w, ch]
                #import pdb; pdb.set_trace()
                X = torch.movedim(X,self.quat_dim,-1)
                labels = torch.movedim(labels,self.quat_dim,-1)
                # scalar first convention for outer product
                X = scalar_last2first(X)
                labels = scalar_last2first(labels)

                if self.act == 'tanhc':
                    X_act = tanh_act(X)
                elif self.act is None:
                    X_act = X 
                return self.loss(X_act,labels)
        def __str__(self):
                return f'Act and Loss: ({self.act},{self.loss})'


# A simple script to test the quats class for numpy and torch
if __name__ == '__main__':

        from symmetries import hcp_syms

        torch.manual_seed(1)
        
        q1 = torch.randn(7,4,17,19)
        q2 = torch.randn(7,4,17,19)

        q1 /= torch.norm(q1,dim=1,keepdim=True)

        q2.requires_grad = True

        acts_and_losses = list()
        
        for act in [None,tanh_act,safe_divide_act]:
                for syms in [None,hcp_syms]:
                        for dist in [l1,l2,rot_dist]:
                                acts_and_losses.append(ActAndLoss(act,Loss(dist,syms,1)))
        

        for i in acts_and_losses:
                print(i)
                d = i(q1,q2)
                L = d.sum()
                print(L)
