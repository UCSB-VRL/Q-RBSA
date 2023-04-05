from math import pi

import torch
import numpy as np


# Defines mapping from quat vector to matrix. Though there are many
# possible matrix representations, this one is selected since the
# first row, X[...,0], is the vector form.
# https://en.wikipedia.org/wiki/Quaternion#Matrix_representations
q1 = np.diag([1,1,1,1])
qj = np.roll(np.diag([-1,1,1,-1]),-2,axis=1)
qk = np.diag([-1,-1,1,1])[:,::-1]
qi = np.matmul(qj,qk)
Q_arr = np.array([q1,qi,qj,qk])
Q_arr_flat_np = Q_arr.reshape((4,16))
Q_arr_flat_torch = torch.as_tensor(Q_arr_flat_np).float()
#Q_arr_flat_torch_cuda = Q_arr_flat_torch.cuda()  

is_cuda = True


# This section defines common array operations, which check whether or
# not the input is numpy or torch, then applies the corresponding operation

# Get L2 norm along last axis
def _norm(X):
    if _is_np(X): return np.linalg.norm(X,axis=-1,keepdims=True)
    else: return torch.norm(X,dim=-1,keepdim=True)

def _moveaxis(X,src,dst):
    return np.moveaxis(X,src,dst) if _is_np(X) else torch.movedim(X,src,dst)

def _array_copy(X):
    return np.copy(X) if _is_np(X) else X.clone()

def _matmul(X1,X2):
    if _is_np(X1) and _is_np(X2):
        return np.matmul(X1,X2)
    elif not _is_np(X1) and not _is_np(X2):
        return torch.matmul(X1,X2)
    else:
        str_types = ['numpy' if _is_np(X) else 'torch' for X in [X1,X2]]
        raise Exception(f'X1 is {str_types[0]} and X2 is {str_types[1]}')



# Other utility functions
def _is_np(X): return isinstance(X,np.ndarray)

# Checks if 2 arrays can be broadcast together
def _broadcastable(s1,s2):
    if len(s1) != len(s2):
        return False
    else:
        return all((i==j) or (i==1) or (j==1) for i,j in zip(s1,s2))

# Converts an array of quats as vectors to matrices. Generally
# used to facilitate quat multiplication.
def _vec2mat(X):
    #import pdb; pdb.set_trace()
    assert X.shape[-1] == 4, 'Last dimension must be of size 4'
    new_shape = X.shape[:-1] + (4,4)
    if _is_np(X): return np.matmul(X,Q_arr_flat_np).reshape(new_shape)
    else:
        device = X.device
        Q = Q_arr_flat_torch.to(device)
        return torch.matmul(X,Q).reshape(new_shape)

        #return torch.matmul(X,Q_arr_flat_torch_cuda).reshape(new_shape) if is_cuda else torch.matmul(X,Q_arr_flat_torch).reshape(new_shape)  



# Performs element-wise multiplication, like the standard multiply in
# numpy. Equivalent to q1 * q2.
def hadamard_prod(q1,q2):
    assert _broadcastable(q1.shape,q2.shape), 'Inputs of shapes ' \
            f'{q1.shape}, {q2.shape} could not be broadcast together'
    X1 = _vec2mat(q1.X)
    X_out = (X1 * q2.X[...,None,:]).sum(-1)
    return Quat(X_out)



# Performs outer product on ndarrays of quats
# Ex if X1.shape = (s1,s2,4) and X2.shape = (s3,s4,s5,4),
# output will be of size (s1,s2,s3,s4,s5,4)
def outer_prod(q1,q2):
    #import pdb; pdb.set_trace()
    X1 = _vec2mat(q1.X)
    X2 = _moveaxis(q2.X,-1,0)
    X1_flat = X1.reshape((-1,4))
    X2_flat = X2.reshape((4,-1))
    X_out = _matmul(X1_flat,X2_flat)
    X_out = X_out.reshape(q1.X.shape + q2.X.shape[:-1])
    X_out = _moveaxis(X_out,len(q1.X.shape)-1,-1)
    return Quat(X_out)



# Utilities to create random vectors on the L2 sphere. First produces
# random samples from a rotationally invariantt distibution (i.e. Gaussian)
# and then normalizes onto the unit sphere

# Produces random array of the same size as shape.
def rand_arr(shape,use_torch):
    if not isinstance(shape,tuple): shape = (shape,)
    if use_torch: X = torch.randn(shape)
    else: X = np.random.standard_normal(shape)
    X /= _norm(X)
    return X

# Produces array of 3D points on the unit sphere.
def rand_points(shape,use_torch=False):
    if not isinstance(shape,tuple): shape = (shape,)
    return rand_arr(shape + (3,),use_torch)

# Produces random unit quaternions.
def rand_quats(shape,use_torch=False):
    if not isinstance(shape,tuple): shape = (shape,)
    return Quat(rand_arr(shape+(4,),use_torch))



# Distance functions: get distance between q1 and q2. If q2 is unspecified,
# then q2 is assumed to be the identity quat <1,0,0,0>.
# Note that quat_dist(q1*q2.conjugate()) = quat_dist(q1,q2), but the second
# formula will be more efficient.

# Get distance between two sets of quats in radians
def quat_dist(q1,q2=None):
    #import pdb; pdb.set_trace()
    if q2 is None: corr = q1.X[...,0]
    else: corr = (q1.X*q2.X).sum(-1)
    
    # normalize corr to -1 to 1 
    corr = torch.clamp(corr, -0.99, 0.99)
    #b, h, w, sym = corr.shape
    #corr = corr.view(b,-1)
    #corr -= corr.min(1, keepdim=True)[0]
    #corr /= (corr.max(1, keepdim=True)[0] + 0.001)

    #corr = corr.view(b,h,w,sym)

    return np.arccos(corr) if _is_np(corr) else torch.arccos(corr)

# Get distance between two sets rotations, accounting for q <-> -q symmetry
def rot_dist(q1,q2=None):
    dq = quat_dist(q1,q2)
    return pi - abs(2*dq - pi)    

# Similar to rot_dist, but will brute-force check all of the provided
# symmetries, and return the minimum.
def rot_dist_w_syms(q1,q2,syms):
    #import pdb; pdb.set_trace()
    q1_w_syms = q1.outer_prod(syms)
    if q2 is not None: q2 = q2[...,None]
    dists = rot_dist(q1_w_syms,q2)
    if _is_np(dists): dist_min = dists.min(-1)
    else: dist_min = dists.min(-1)[0]
    return dist_min



# Main class for quaternions
# Defaults to using numpy unless one of the following is true:
#     use_torch = True
#     X is a already a torch tensor
# You can convert between the two with .use_numpy or .use_torch()
# Should act like a numpy array or torch tensor in most ways, supporting
# slicing, addition, subtraction, multiplication, transpose, etc. Also
# includes a method for rotation of points in 3D. Rotation of another
# rotation should be done using multiplication.
class Quat:
    def __init__(self,X,use_torch=False,scalar_first=True, use_gpu= False):
        #import pdb; pdb.set_trace()
        if not use_torch and not isinstance(X,torch.Tensor):
            self.X = np.asarray(X)
        else: self.X = torch.as_tensor(X).cuda() if use_gpu else torch.as_tensor(X).cuda()
        assert self.X.shape[-1] == 4, 'Last dimension must be of size 4'
        self.shape = self.X.shape[:-1]
        if not scalar_first:
            if _is_np(self.X): self.X = np.roll(self.X,1,axis=-1)
            else: self.X = torch.roll(self.X,1,dims=-1)

    def __add__(self,q2):
        return Quat(self.X+q2.X)

    def __sub__(self,q2):
        return Quat(self.X-q2.X)

    def __mul__(self,q2):
        return hadamard_prod(self,q2)

    def outer_prod(self,q2):
        return outer_prod(self,q2)

    def __str__(self):
        return str(self.X)

    def __getitem__(self,index):
        if isinstance(index,tuple): index = index + (slice(None),)
        else: index = (index,slice(None))
        return Quat(self.X[index])

    def to_numpy(self): 
        #import pdb; pdb.set_trace()
        return Quat(self.X.numpy())

    def to_torch(self): return Quat(torch.as_tensor(self.X).float())

    # Equivalent to the inverse of a rotation
    def conjugate(self):
        X_out = _array_copy(self.X)
        X_out[...,1:] *= -1
        return Quat(X_out)

    def reshape(self,axes):
        if isinstance(axes,tuple): return Quat(self.X.reshape(axes + (4,)))
        else: return Quat(self.X.reshape((axes,4)))

    def transpose(self,axes):
        assert min(axes) >= 0
        #import pdb; pdb.set_trace()
        # does not work with torch tensor
        return Quat(self.X.transpose(axes+(-1,)))

    def rotate(self,points,element_wise=False):
        if _is_np(self.X):
            points = np.asarray(points)
            P = np.zeros(points.shape[:-1] + (4,))
        else:
            points = torch.as_tensor(points)
            P = torch.zeros(points.shape[:-1] + (4,))
        assert points.shape[-1] == 3, 'Last dimension must be of size 3'
        P[...,1:] = points
        qp = Quat(P)
        if element_wise:
            X_out = (self * qp * self.conjugate()).X
        else:
            X_int = self.outer_prod(qp)
            inds = (slice(None),)*(len(self.X.shape)-1) + \
                    (None,)*(len(qp.X.shape)) + (slice(None),)
            X_out = (_vec2mat(X_int.X) * self.conjugate().X[inds]).sum(-1)
        return X_out[...,1:]


# A simple script to test the quats class for numpy and torch
if __name__ == '__main__':

    import pdb; pdb.set_trace()
    np.random.seed(1)
    N = 7
    M = 11
    K = 13

    for i in range(2):

        use_torch = bool(i)
        if i == 0:
            print('using numpy (default float64)')
        else: 
            print('using torch (default float32)')

        q1 = rand_quats(M,use_torch)
        q2 = rand_quats(N,use_torch)
        p1 = rand_points(K,use_torch)

        p2 = q2.rotate(q1.rotate(p1))
        p3 = q2.outer_prod(q1).rotate(p1)
        p4 = q1.conjugate()[:,None].rotate(q1.rotate(p1),element_wise=True)

        print('Composition of rotation error:')
        err = abs(p2-p3).sum()/np.prod(p2.shape)
        print('\t',err,'\n')

        print('Rotate then apply inverse rotation error:')
        err = abs(p4-p1).sum()/np.prod(p4.shape)
        print('\t',err,'\n')

