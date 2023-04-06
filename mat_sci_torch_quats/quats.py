from math import pi
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Defines mapping from quat vector to matrix. Though there are many
# possible matrix representations, this one is selected since the
# first row, X[...,0], is the vector form.
# https://en.wikipedia.org/wiki/Quaternion#Matrix_representations
#import pdb; pdb.set_trace()
q1 = np.diag([1,1,1,1])
qj = np.roll(np.diag([-1,1,1,-1]),-2,axis=1)
qk = np.diag([-1,-1,1,1])[:,::-1]
qi = np.matmul(qj,qk)
Q_arr = torch.Tensor([q1,qi,qj,qk])
Q_arr_flat = Q_arr.reshape((4,16))


# Checks if 2 arrays can be broadcast together
def _broadcastable(s1,s2):
        if len(s1) != len(s2): return False
        else: return all((i==j) or (i==1) or (j==1) for i,j in zip(s1,s2))

# Converts an array of quats as vectors to matrices. Generally
# used to facilitate quat multiplication.
def vec2mat(X):
        #import pdb; pdb.set_trace()
        assert X.shape[-1] == 4, 'Last dimension must be of size 4'
        new_shape = X.shape[:-1] + (4,4)
        dtype = X.dtype
        Q = Q_arr_flat.type(X.dtype).to(X.device)
        #print('Q', Q.dtype)
        return torch.matmul(X,Q).reshape(new_shape)


# Performs element-wise multiplication, like the standard multiply in
# numpy. Equivalent to q1 * q2.
def hadamard_prod(q1,q2):
        assert _broadcastable(q1.shape,q2.shape), 'Inputs of shapes ' \
                        f'{q1.shape}, {q2.shape} could not be broadcast together'
        X1 = vec2mat(q1)
        X_out = (X1 * q2[...,None,:]).sum(-1)
        return X_out



# Performs outer product on ndarrays of quats
# Ex if X1.shape = (s1,s2,4) and X2.shape = (s3,s4,s5,4),
# output will be of size (s1,s2,s3,s4,s5,4)
def outer_prod(q1,q2):
        #import pdb; pdb.set_trace()
        X1 = vec2mat(q1)
        X2 = torch.movedim(q2,-1,0)
        X1_flat = X1.reshape((-1,4))
        X2_flat = X2.reshape((4,-1))
        X_out = torch.matmul(X1_flat,X2_flat)
        X_out = X_out.reshape(q1.shape + q2.shape[:-1])
        X_out = torch.movedim(X_out,len(q1.shape)-1,-1)
        return X_out


# Utilities to create random vectors on the L2 sphere. First produces
# random samples from a rotationally invariantt distibution (i.e. Gaussian)
# and then normalizes onto the unit sphere

# Produces random array of the same size as shape.
def rand_arr(shape,dtype=torch.FloatTensor):
        if not isinstance(shape,tuple): shape = (shape,)
        X = torch.randn(shape).type(dtype)
        X /= torch.norm(X,dim=-1,keepdim=True)
        return X

# Produces array of 3D points on the unit sphere.
def rand_points(shape,dtype=torch.FloatTensor):
        if not isinstance(shape,tuple): shape = (shape,)
        return rand_arr(shape + (3,), dtype)

# Produces random unit quaternions.
def rand_quats(shape,dtype=torch.FloatTensor):
        if not isinstance(shape,tuple): shape = (shape,)
        return rand_arr(shape+(4,), dtype)


# arccos, expanded from range [-1,1] to all real numbers
# values outside of [-1,1] and replaced with a line of slope pi/2, such that
# the function is continuous
def safe_arccos(x):
    mask = (torch.abs(x) < 1).float()
    x_clip = torch.clamp(x,min=-1,max=1)
    output_arccos = torch.arccos(x_clip)
    output_linear = (1 - x)*pi/2
    output = mask*output_arccos + (1-mask)*output_linear
    return output


def quat_dist(q1,q2=None):
        """
        Computes distance between two quats. If q1 and q2 are on the unit sphere,
        this will return the arc length along the sphere. For points within the
        sphere, it reduces to a function of MSE.
        """
        #import pdb; pdb.set_trace()
        if q2 is None: mse = (q1[...,0]-1)**2 + (q1[...,1:]**2).sum(-1)
        else: mse = ((q1-q2)**2).sum(-1)
        
        corr = 1 - (1/2)*mse
        # my fz version
        #corr = 1-2*(1- q1[...,0]**2)
    
        #corr_clamp = torch.clamp(corr,min=-1,max=1)
        #arccos = torch.arccos(corr_clamp)
        return safe_arccos(corr)
        #return arccos 
        
def rot_dist(q1,q2=None):
        """ Get dist between two rotations, with q <-> -q symmetry """
        #import pdb; pdb.set_trace()
        q1_w_neg = torch.stack((q1,-q1),dim=-2)
        if q2 is not None: q2 = q2[...,None,:]
        dists = quat_dist(q1_w_neg,q2)
        dist_min = dists.min(-1)[0]
        return dist_min

        
def fz_reduce(q,syms):
        #import pdb; pdb.set_trace()
        shape = q.shape
        q = q.reshape((-1,4))
        syms = syms.cuda()
        q_w_syms = outer_prod(q,syms)
        dists = rot_dist(q_w_syms)
        inds = dists.min(-1)[1]
        q_fz = q_w_syms[torch.arange(len(q_w_syms)),inds]
        q_fz *= torch.sign(q_fz[...,:1])
        q_fz = q_fz.reshape(shape)
        return q_fz


def scalar_first2last(X):
        return torch.roll(X,-1,-1)

def scalar_last2first(X):
        return torch.roll(X,1,-1)

def conj(q):
        q_out = q.clone()
        q_out[...,1:] *= -1
        return q_out


def rotate(q,points,element_wise=False):
        points = torch.as_tensor(points)
        P = torch.zeros(points.shape[:-1] + (4,),dtype=q.dtype,device=q.device)
        assert points.shape[-1] == 3, 'Last dimension must be of size 3'
        P[...,1:] = points
        if element_wise:
                X_int = hadamard_prod(q,P)
                X_out = hadamard_prod(X_int,conj(q))
        else:
                X_int = outer_prod(q,P)
                inds = (slice(None),)*(len(q.shape)-1) + \
                                (None,)*(len(P.shape)) + (slice(None),)
                X_out = (vec2mat(X_int) * conj(q)[inds]).sum(-1)
        return X_out[...,1:]


def linear_interpolation(q1, q2, frac):
    #inner product
    # scalar last convention
    lamda = torch.inner(q1, q2)
    mask = (lamda < 0)
    mask_ind = mask.nonzero()
    q2[mask_ind] = -q2[mask_ind]
    
    qin = q1 + frac * (q2 - q1)
    qin /= torch.norm(qin, dim=-1, keepdim=True) 
    
    return qin

def slerp_ebsd2d(input, scale_factor_h, scale_factor_w):
    import pdb; pdb.set_trace()
    source_H = input.shape[0]
    source_W = input.shape[1]
    source_C = input.shape[2]

    resized_H = int(source_H * scale_factor_h)
    resized_W = int(source_W * scale_factor_w)  
    
    output = np.zeros((resized_H, resized_W, source_C))
   
    def read_pixel(x, y):
        x = np.clip(x, 0, source_W - 1)
        y = np.clip(y, 0, source_H - 1)
        return input[y, x]   

    def quat_interpolate(x, y):
        #import pdb; pdb.set_trace()
        x1 = int(np.floor(x))
        x2 = x1 + 1

        y1 = int(np.floor(y))
        y2 = y1 + 1

        P11 = read_pixel(x1, y1)
        P12 = read_pixel(x1, y2)
        P21 = read_pixel(x2, y1)
        P22 = read_pixel(x2, y2)
        
        #return (P11 * (x2 - x) * (y2 - y) + 
        #        P12 * (x2 - x) * (y - y1) + 
        #        P21 * (x - x1) * (y2 - y) + 
        #        P22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))
   
        P_int = slerp_quat(slerp_quat(P11, P12, 0.5), slerp_quat(P21, P22, 0.5), 0.5)
        #P_int =0.25 * slerp_quat(P11,P12, 0.5) + 0.25 * slerp_quat(P11,P21, 0.5) + 0.25 * slerp_quat(P21,P22, 0.5) + 0.25 * slerp_quat(P12,P22, 0.5)

        #import pdb; pdb.set_trace()
        return P_int

    for dst_y in range(resized_H):
        for dst_x in range(resized_W):
            #print(dst_y, dst_x)
            src_x = (dst_x + 0.5) / scale_factor_w - 0.5
            src_y = (dst_y + 0.5) / scale_factor_h - 0.5
            output[dst_y, dst_x] = quat_interpolate(src_x, src_y)

    import pdb; pdb.set_trace()
    return output    

def slerp_quat(q0, q1, t):
    #import pdb; pdb.set_trace()
        
    epsilon = 0.0001  
    dot = np.sum(q0 * q1)

    if dot < 0:
        # quaternions are poinitng in opposite directions 
        # use equivalent alternative representation for q2
        q1 = -q1
     
    if np.absolute(1 - dot) < epsilon:   
        # quaternions are nearly parallel
        # linear interpolation
        qin = q0 + t*(q1 - q0)
    else:
        # spherical interpolation
        dot = np.clip(dot, -1.0, 1.0)
        omega = np.arccos(dot)
        so = np.sin(omega)
        qin = np.sin((1.0-t)*omega) / so * q0 + np.sin(t*omega)/so * q1

    #norm = np.linalg.norm(qin)
    #qin_norm = qin / norm

    return qin




def slerp_quat_multidimensional(q0, q1, t):
    import pdb; pdb.set_trace()
    q0 = torch.tensor(q0)
    q1 = torch.tensor(q1)
    
    epsilon = 0.0001  
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

    mask_ops = dot < 0
    # quaternions are poinitng in opposite directions 
    # use equivalent alternative representation for q2 
    mask_ind_ops = mask_ops.nonzero()
    q1[mask_ind_ops] = -q1[mask_ind_ops]

    # quaternions are nearly parallel
    # linear interpolation
    mask_prll = torch.norm(1 - dot, dim=-1, keepdim=True) < epsilon
    mask_prll = (mask_prll).float() 
    qin_maskprll = q0 + t*(q1 - q0)
   
    # spherical interpolation
    dot = torch.clamp(dot, -1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    qin_mask_noprll = torch.sin((1.0-t)*omega) / so * q0 + torch.sin(t*omega)/so * q1

    qin = mask_prll * qin_maskprll + (1-mask_prll) * qin_mask_noprll

    norm = torch.norm(qin, dim=-1, keepdim=True)
    qin_norm = qin / norm

    return qin_norm

def save_quat(arr_list):
    #import pdb; pdb.set_trace()
    postfix = ['LR', 'Slerp', 'HR']
    kwargs_imshow = {'vmin': -1, 'vmax': 1}
    channels = ['q1', 'q2', 'q3', 'q0']

    for ch_num, channel in enumerate(channels):
        fig, axes = plt.subplots((len(postfix)+1)//3,3, figsize=(14,12), constrained_layout = True)
        for ax, arr, title in zip(axes.reshape(-1), arr_list, postfix):
            if ch_num == 0:
                np.save(f'slerp_out_{title}.npy', arr)
                
            img_numpy = arr[:,:, ch_num]
            im = ax.imshow(img_numpy, **kwargs_imshow, cmap='jet')
            ax.set_title(title, fontweight="bold")

        cbar =fig.colorbar(im, ax = axes.ravel().tolist(), shrink=0.95)
        cbar.set_ticklabels(np.arange(0,1,0.2))
        cbar.set_ticklabels([-1 , 0, 1])
    
        plt.savefig(f'slerp_out_{channel}.png')
        
        plt.close()


def normalize(x):
        x_norm = torch.norm(x, dim=-1, keepdim=True)
                # make ||q|| = 1
        y_norm = torch.div(x, x_norm) 

        return y_norm
     

def post_process(x, org_shape):
    #import pdb; pdb.set_trace()
    #rotate by 0 or 180 degrees about x axis
    hcp_r1 = torch.eye(4)[:2]

    # rotate about 0, 60, ... 300 degrees about z axis
    hcp_r2 = torch.zeros((6,4))
    hcp_r2[:,0] = torch.cos(torch.arange(6)/6*np.pi)
    hcp_r2[:,3] = torch.sin(torch.arange(6)/6*np.pi)
    hcp_syms = outer_prod(hcp_r1,hcp_r2).reshape((-1,4))
    hcp_syms = hcp_syms.type(torch.DoubleTensor)

    x = torch.tensor(x)
    x = x.cuda()
    x = normalize(x)
    h, w, ch = org_shape
   
    #import pdb; pdb.set_trace() 
    x = scalar_last2first(x)
    x = fz_reduce(x, hcp_syms)
    
    x = scalar_first2last(x)
    x = x.detach().cpu().numpy()
   
    return x
    


# A simple script to test the quats class for numpy and torch
if __name__ == '__main__':

        import pdb; pdb.set_trace()
        np.random.seed(1)
        N = 6
        M = 6
        K = 13

        def test1(dtype,device):

                q1 = rand_quats(M,dtype).to(device)
                q2 = rand_quats(N,dtype).to(device)
                q3 = rand_quats(M,dtype).to(device)
                p1 = rand_points(K,dtype).to(device)

                p2 = rotate(q2,rotate(q1,p1))
                p3 = rotate(outer_prod(q2,q1),p1)
                p4 = rotate(conj(q1[:,None]),rotate(q1,p1),element_wise=True)

                print('Composition of rotation error:')
                err = abs(p2-p3).sum()/len(p2.reshape(-1))
                print('\t',err)

                print('Rotate then apply inverse rotation error:')
                err = abs(p4-p1).sum()/len(p1.reshape(-1))
                print('\t',err,'\n')

       
        def test2(dtype,device):
                import pdb; pdb.set_trace()
                q1 = rand_quats(M,dtype).to(device)
                q2 = rand_quats(N,dtype).to(device)
                #q_in = linear_interpolation(q1, q2, 0.5)
                q_in = slerp(q1, q2, 0.5)
                
                
        def test3(dtype, device):
            import pdb;pdb.set_trace()
            img = np.load(f'Ti64_orthogonal_sectioning_Test_X_Block_lr_0.npy')
            img_hr = np.load(f'Ti64_orthogonal_sectioning_Test_X_Block_hr_0.npy')
            scale_factor = 2
            img_upsample = slerp_ebsd2d(img, scale_factor, scale_factor)
            img_upsample = post_process(img_upsample, img_hr.shape)
            save_quat([img, img_upsample, img_hr]) 
            import pdb; pdb.set_trace() 
 
        print('CPU Float 32')
        test3(torch.cuda.FloatTensor,'cpu')

        print('CPU Float64')
        test2(torch.cuda.DoubleTensor,'cpu')     

        if torch.cuda.is_available():

                print('CUDA Float 32')
                test2(torch.cuda.FloatTensor,'cuda')

                print('CUDA Float64')
                test2(torch.cuda.DoubleTensor,'cuda') 

        else:
                print('No CUDA')

