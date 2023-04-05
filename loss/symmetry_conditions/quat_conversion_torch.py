####################################################################################################
# Copyright (c) 2017-2020, Martin Diehl/Max-Planck-Institut für Eisenforschung GmbH
# Copyright (c) 2013-2014, Marc De Graef/Carnegie Mellon University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
#     - Redistributions of source code must retain the above copyright notice, this list
#        of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright notice, this
#        list of conditions and the following disclaimer in the documentation and/or
#        other materials provided with the distribution.
#     - Neither the names of Marc De Graef, Carnegie Mellon University nor the names
#        of its contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
####################################################################################################

# Note: An object oriented approach to use this conversions is available in DAMASK, see
# https://damask.mpie.de and https://github.com/eisenforschung/DAMASK

# Done: Quat to ho, ho to cu, quat to ro


import torch
import math
from torch.autograd import Variable

P = -1

# parameters for conversion from/to cubochoric
sc   = math.pi**(1./6.)/6.**(1./6.)
beta = math.pi**(5./6.)/6.**(1./6.)/2.
R1   = (3.*math.pi/4.)**(1./3.)


#---------- Quaternion Operations ----------

def quat2cubo(qu, scalar_first=False ):
    """Quaternion to cubochoric vector."""

    """ Step 1: Quaternion to homochoric vector."""

    # Converting Quaternion Order convention
    # DREAM3D convention is <xyz> s (<xyz> = imaginary vector component)
    # Convention used here is s <xyz>
    # This conversion is <xyz>,s to s,<xyz>
    if scalar_first is not True:
        # if qu.ndim == 1:
        #     qu = torch.unsqueeze(qu, dim=0)
        qu = torch.index_select(qu, -1, _device_check(torch.LongTensor([3, 0, 1, 2]), qu.device))
    #-------

    # with np.errstate(invalid='ignore'):
    omega = 2.0 * torch.arccos(torch.clip(qu[..., 0:1], -1.0, 1.0))

    ho = torch.where(torch.lt(torch.abs(omega), 1.0e-12),
                  _precision_check([0, 0, 0], qu.dtype, qu.device),
                  qu[..., 1:4] / torch.linalg.norm(qu[..., 1:4], dim=-1, keepdim=True) \
                  * _cbrt(0.75 * (omega - torch.sin(omega))))

    # return ho # inserted here gives back the homochoric coordinates

    """
        Step 2: Homochoric vector to cubochoric vector.
        References
        ----------
        D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
        https://doi.org/10.1088/0965-0393/22/7/075013
        """
    rs = torch.linalg.norm(ho, dim=-1, keepdim=True)

    # xyz3 = np.take_along_axis(ho, _get_pyramid_order(ho, 'forward'), -1)
    xyz3 = torch.gather(ho, -1, _get_tensor_pyramid_order(ho, 'forward'))

    # with np.errstate(invalid='ignore', divide='ignore'):
    # inverse M_3

    xyz2 = xyz3[..., 0:2] * torch.sqrt(2.0 * rs / (rs + torch.abs(xyz3[..., 2:3])))
    qxy = torch.sum(xyz2 ** 2, -1, keepdim=True)

    q2 = qxy + torch.amax(torch.abs(xyz2), -1, keepdim=True) ** 2
    sq2 = torch.sqrt(q2)
    q = (beta / math.sqrt(2.0) / R1) * torch.sqrt(q2 * qxy / (q2 - torch.amax(torch.abs(xyz2), -1, keepdim=True) * sq2))
    tt = torch.clip((torch.amin(torch.abs(xyz2), -1, keepdim=True) ** 2 \
                  + torch.amax(torch.abs(xyz2), -1, keepdim=True) * sq2) / math.sqrt(2.0) / qxy, -1.0, 1.0)
    T_inv = torch.where(torch.le(torch.abs(xyz2[..., 1:2]), torch.abs(xyz2[..., 0:1])),
                     torch.cat((torch.ones_like(tt), torch.arccos(tt)/math.pi*12), dim=-1),
                     torch.cat((torch.arccos(tt)/math.pi*12, torch.ones_like(tt)), dim=-1)) * q
    T_inv[xyz2 < 0.0] *= -1.0  # warning

    T_inv[(torch.isclose(qxy, _precision_check(0.0, qxy.dtype), rtol=0.0, atol=1.0e-12)).expand(T_inv.shape)] = 0.0
    cu = torch.cat((T_inv, torch.where(torch.lt(xyz3[..., 2:3],0.0), -torch.ones_like(xyz3[..., 2:3]), torch.ones_like(xyz3[..., 2:3])) \
                    * rs / math.sqrt(6.0 / math.pi)), dim=-1) / sc

    cu[torch.isclose(torch.sum(torch.abs(ho), -1), _precision_check(0.0, ho.dtype), rtol=0.0, atol=1.0e-16)] = 0.0
    cu = torch.gather(cu, -1, _get_tensor_pyramid_order(ho, 'backward'))

    return cu

def quat2rod(qu, scalar_first=False):
    """Step 1: Quaternion to Rodrigues-Frank vector."""

    # Converting Quaternion Order convention
    # DREAM3D convention is <xyz> s (<xyz> = imaginary vector component)
    # Convention used here is s <xyz>
    # This conversion is <xyz>,s to s,<xyz>
    if scalar_first is not True:
        # if qu.ndim == 1:
        #     qu = torch.unsqueeze(qu, dim=0)
        qu = torch.index_select(qu, -1, _device_check(torch.LongTensor([3, 0, 1, 2]), qu.device))
    # -------

    # with np.errstate(invalid='ignore', divide='ignore'):
    s = torch.linalg.norm(qu[..., 1:4], dim=-1, keepdim=True)
    ro = torch.where(torch.lt(torch.abs(qu[..., 0:1]), 1.0e-12).expand(qu.shape),
                     torch.cat((qu[..., 1:2], qu[..., 2:3], qu[..., 3:4],
                                _precision_check(float('inf'), qu.dtype, qu.device).expand(qu.shape[:-1] + (1,))), dim=-1),
                     torch.cat((qu[..., 1:2] / s, qu[..., 2:3] / s, qu[..., 3:4] / s,
                                torch.tan(torch.acos(torch.clip(qu[..., 0:1], -1.0, 1.0)))), dim=-1)
                  )
    ro[torch.lt(torch.squeeze(torch.abs(s), dim=-1), 1.0e-12)] = _precision_check([0.0, 0.0, P, 0.0], s.dtype, s.device)
    return ro

#---------- Cubochoric Operations ----------

def cubo2quat(cu, scalar_first=False):
    """Cubochoric vector to quaternion."""

    """
        Step 1: Cubochoric vector to homochoric vector.
        References
        ----------
        D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
        https://doi.org/10.1088/0965-0393/22/7/075013
        """
    # with np.errstate(invalid='ignore', divide='ignore'):
    # get pyramide and scale by grid parameter ratio

    XYZ = torch.gather(cu, -1, _get_tensor_pyramid_order(cu, 'forward')) * sc
    order = torch.le(torch.abs(XYZ[..., 1:2]), torch.abs(XYZ[..., 0:1]))
    q = math.pi / 12.0 * torch.where(order, XYZ[..., 1:2], XYZ[..., 0:1]) \
        / torch.where(order, XYZ[..., 0:1], XYZ[..., 1:2])
    c = torch.cos(q)
    s = torch.sin(q)
    q = R1 * 2.0 ** 0.25 / beta / torch.sqrt(math.sqrt(2.0) - c) \
        * torch.where(order, XYZ[..., 0:1], XYZ[..., 1:2])


    T = torch.cat(((math.sqrt(2.0) * c - 1.0), math.sqrt(2.0) * s), dim=-1)*q


    # transform to sphere grid (inverse Lambert)
    c = torch.sum(T ** 2, -1, keepdim=True)
    s = c * math.pi / 24.0 / XYZ[..., 2:3] ** 2
    c = c * math.sqrt(math.pi / 24.0) / XYZ[..., 2:3]
    q = torch.sqrt(1.0 - s)

    ho = torch.where(torch.isclose(torch.sum(torch.abs(XYZ[..., 0:2]), -1, keepdim=True), _precision_check(0.0, cu.dtype), rtol=0.0, atol=1.0e-16),
                  torch.cat((torch.zeros_like(XYZ[..., 0:2]), math.sqrt(6.0 / math.pi) * XYZ[..., 2:3]), dim=-1),
                  torch.cat((torch.where(order, T[..., 0:1], T[..., 1:2]) * q,
                            torch.where(order, T[..., 1:2], T[..., 0:1]) * q,
                            math.sqrt(6.0 / math.pi) * XYZ[..., 2:3] - c), dim=-1)
                  )

    ho[torch.isclose(torch.sum(torch.abs(cu), -1), _precision_check(0.0, cu.dtype, cu.device), rtol=0.0, atol=1.0e-16)] = _precision_check(0.0, ho.dtype, ho.device)


    ho = torch.gather(ho, -1, _get_tensor_pyramid_order(cu, 'backward'))


    # return ho # here for homochoric

    """Step 2: Homochoric vector to axis angle pair."""
    tfit = [+1.0000000000018852, -0.5000000002194847,
                     -0.024999992127593126, -0.003928701544781374,
                     -0.0008152701535450438, -0.0002009500426119712,
                     -0.00002397986776071756, -0.00008202868926605841,
                     +0.00012448715042090092, -0.0001749114214822577,
                     +0.0001703481934140054, -0.00012062065004116828,
                     +0.000059719705868660826, -0.00001980756723965647,
                     +0.000003953714684212874, -0.00000036555001439719544]
    hmag_squared = torch.sum(ho ** 2., -1, keepdim=True)

    hm = torch.clone(hmag_squared) # use detach() for decoupled autograd relationship

    s = tfit[0] + tfit[1] * hmag_squared
    for i in range(2, 16):
        hm *= hmag_squared
        s += tfit[i] * hm

    # with np.errstate(invalid='ignore'):
    ax = torch.where(torch.lt(torch.abs(hmag_squared), torch.tensor(1.e-8)).expand(ho.shape[:-1] + (4,)),
                  _precision_check([0.0, 0.0, 1.0, 0.0], ho.dtype, ho.device),
                  torch.cat((ho / torch.sqrt(hmag_squared), 2.0 * torch.arccos(torch.clip(s, -1.0, 1.0))), dim=-1))

    # return ax # here for axis angle pair

    """Step 3: Axis angle pair to quaternion."""
    c = torch.cos(ax[..., 3:4] * .5)
    s = torch.sin(ax[..., 3:4] * .5)
    if cu.requires_grad:
        qu = Variable(torch.where(torch.lt(torch.abs(ax[..., 3:4]), 1.e-6), _precision_check([1.0, 0.0, 0.0, 0.0], ax.dtype, ax.device), torch.cat((c, ax[..., :3] * s), dim=-1)), requires_grad=True)

    else:
        qu = torch.where(torch.lt(torch.abs(ax[..., 3:4]), 1.e-6), _precision_check([1.0, 0.0, 0.0, 0.0], ax.dtype, ax.device), torch.cat((c, ax[..., :3] * s), dim=-1))


    # Converting Quaternion Order convention
    # DREAM3D convention is <xyz> s (<xyz> = imaginary vector component)
    # Convention used here is s <xyz>
    # This conversion is s,<xyz> to <xyz>,s
    if scalar_first is not True:
        # if qu.ndim == 1:
        #     qu = torch.unsqueeze(qu, dim=0)
        qu = torch.index_select(qu, -1, _device_check(torch.LongTensor([1, 2, 3, 0]), qu.device))
    #-------

    return qu

def cubo2rod(cu):
    """Cubochoric vector to quaternion."""

    """
        Step 1: Cubochoric vector to homochoric vector.
        References
        ----------
        D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
        https://doi.org/10.1088/0965-0393/22/7/075013
        """
    # with np.errstate(invalid='ignore', divide='ignore'):
    # get pyramide and scale by grid parameter ratio
    XYZ = torch.gather(cu, -1, _get_tensor_pyramid_order(cu, 'forward')) * sc
    order = torch.le(torch.abs(XYZ[..., 1:2]), torch.abs(XYZ[..., 0:1]))
    q = math.pi / 12.0 * torch.where(order, XYZ[..., 1:2], XYZ[..., 0:1]) \
        / torch.where(order, XYZ[..., 0:1], XYZ[..., 1:2])
    c = torch.cos(q)
    s = torch.sin(q)
    q = R1 * 2.0 ** 0.25 / beta / torch.sqrt(math.sqrt(2.0) - c) \
        * torch.where(order, XYZ[..., 0:1], XYZ[..., 1:2])

    T = torch.cat(((math.sqrt(2.0) * c - 1.0), math.sqrt(2.0) * s), dim=-1)*q

    # transform to sphere grid (inverse Lambert)
    c = torch.sum(T ** 2, -1, keepdim=True)
    s = c * math.pi / 24.0 / XYZ[..., 2:3] ** 2
    c = c * math.sqrt(math.pi / 24.0) / XYZ[..., 2:3]
    q = torch.sqrt(1.0 - s)

    ho = torch.where(torch.isclose(torch.sum(torch.abs(XYZ[..., 0:2]), -1, keepdim=True), _precision_check(0.0, XYZ.dtype), rtol=0.0, atol=1.0e-16),
                  torch.cat((torch.zeros_like(XYZ[..., 0:2]), math.sqrt(6.0 / math.pi) * XYZ[..., 2:3]), dim=-1),
                  torch.cat((torch.where(order, T[..., 0:1], T[..., 1:2]) * q,
                            torch.where(order, T[..., 1:2], T[..., 0:1]) * q,
                            math.sqrt(6.0 / math.pi) * XYZ[..., 2:3] - c), dim=-1)
                  )

    ho[torch.isclose(torch.sum(torch.abs(cu), -1), _precision_check(0.0, cu.dtype), rtol=0.0, atol=1.0e-16)] = 0.0 # warning
    ho = torch.gather(ho, -1, _get_tensor_pyramid_order(cu, 'backward'))

    # return ho # here for homochoric

    """Step 2: Homochoric vector to axis angle pair."""
    tfit = [+1.0000000000018852, -0.5000000002194847,
                     -0.024999992127593126, -0.003928701544781374,
                     -0.0008152701535450438, -0.0002009500426119712,
                     -0.00002397986776071756, -0.00008202868926605841,
                     +0.00012448715042090092, -0.0001749114214822577,
                     +0.0001703481934140054, -0.00012062065004116828,
                     +0.000059719705868660826, -0.00001980756723965647,
                     +0.000003953714684212874, -0.00000036555001439719544]
    hmag_squared = torch.sum(ho ** 2., -1, keepdim=True)

    hm = torch.clone(hmag_squared) # use detach() for decoupled autograd relationship

    s = tfit[0] + tfit[1] * hmag_squared
    for i in range(2, 16):
        hm *= hmag_squared
        s += tfit[i] * hm

    # with np.errstate(invalid='ignore'):
    ax = torch.where(torch.lt(torch.abs(hmag_squared), torch.tensor(1.e-8)).expand(ho.shape[:-1] + (4,)),
                  _precision_check([0.0, 0.0, 1.0, 0.0], ho.dtype, ho.device),
                  torch.cat((ho / torch.sqrt(hmag_squared), 2.0 * torch.arccos(torch.clip(s, -1.0, 1.0))), dim=-1))

    # return ax # here for axis angle pair

    """Step 3: Axis angle pair to Rodrigues-Frank vector."""
    ro = torch.cat((ax[...,:3],
                   torch.where(torch.isclose(ax[...,3:4], _precision_check(math.pi, ax.dtype),atol=1.e-15,rtol=.0),
                            _precision_check(float('inf'), ax.dtype, ax.device),
                            torch.tan(ax[...,3:4]*0.5))
                    ), dim = -1)
    ro[torch.lt(torch.abs(ax[...,3]), 1.e-6)] = _precision_check([.0,.0,P,.0], ax.dtype, ax.device)

    return ro

#---------- Rodrigues Operations ----------

def rod2cubo(ro):
    """ Step 1: Rodrigues-Frank vector to homochoric vector."""

    f = torch.where(torch.isfinite(ro[...,3:4]),2.0*torch.arctan(ro[...,3:4]) -torch.sin(2.0*torch.arctan(ro[...,3:4])), _precision_check(math.pi, ro.dtype, ro.device))

    ho = torch.where(torch.lt(torch.sum(ro[...,0:3]**2.0, -1, keepdim=True), _precision_check(1.e-8, ro.dtype)).expand(ro[...,0:3].shape),
                     _precision_check([0, 0, 0], ro.dtype, ro.device), ro[...,0:3]* _cbrt(0.75*f))

    # return ho # here for homochoric vector

    """
        Step 2: Homochoric vector to cubochoric vector.
        References
        ----------
        D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
        https://doi.org/10.1088/0965-0393/22/7/075013
        """
    rs = torch.linalg.norm(ho, dim=-1, keepdim=True)

    # xyz3 = np.take_along_axis(ho, _get_pyramid_order(ho, 'forward'), -1)
    xyz3 = torch.gather(ho, -1, _get_tensor_pyramid_order(ho, 'forward'))

    # with np.errstate(invalid='ignore', divide='ignore'):
    # inverse M_3

    xyz2 = xyz3[..., 0:2] * torch.sqrt(2.0 * rs / (rs + torch.abs(xyz3[..., 2:3])))
    qxy = torch.sum(xyz2 ** 2, -1, keepdim=True)

    q2 = qxy + torch.amax(torch.abs(xyz2), -1, keepdim=True) ** 2
    sq2 = torch.sqrt(q2)
    q = (beta / math.sqrt(2.0) / R1) * torch.sqrt(q2 * qxy / (q2 - torch.amax(torch.abs(xyz2), -1, keepdim=True) * sq2))
    tt = torch.clip((torch.amin(torch.abs(xyz2), -1, keepdim=True) ** 2 \
                  + torch.amax(torch.abs(xyz2), -1, keepdim=True) * sq2) / math.sqrt(2.0) / qxy, -1.0, 1.0)
    T_inv = torch.where(torch.le(torch.abs(xyz2[..., 1:2]), torch.abs(xyz2[..., 0:1])),
                     torch.cat((torch.ones_like(tt), torch.arccos(tt)/math.pi*12), dim=-1),
                     torch.cat((torch.arccos(tt)/math.pi*12, torch.ones_like(tt)), dim=-1)) * q
    T_inv[xyz2 < 0.0] *= -1.0  # warning

    T_inv[(torch.isclose(qxy, _precision_check(0.0, qxy.dtype), rtol=0.0, atol=1.0e-12)).expand(T_inv.shape)] = 0.0
    cu = torch.cat((T_inv, torch.where(torch.lt(xyz3[..., 2:3],0.0), -torch.ones_like(xyz3[..., 2:3]), torch.ones_like(xyz3[..., 2:3])) \
                    * rs / math.sqrt(6.0 / math.pi)), dim=-1) / sc

    cu[torch.isclose(torch.sum(torch.abs(ho), -1), _precision_check(0.0, ho.dtype), rtol=0.0, atol=1.0e-16)] = 0.0
    cu = torch.gather(cu, -1, _get_tensor_pyramid_order(ho, 'backward'))

    return cu

def rod2quat(ro, scalar_first=False):
    """Step 1:  Rodrigues-Frank vector to axis angle pair."""
    # with np.errstate(invalid='ignore',divide='ignore'):
    ax = torch.where(torch.isfinite(ro[...,3:4]),
         torch.cat((ro[...,0:3]*torch.linalg.norm(ro[...,0:3],dim=-1,keepdim=True), 2.*torch.arctan(ro[...,3:4])), dim=-1),
         torch.cat((ro[...,0:3], _precision_check(math.pi, ro.dtype, ro.device).expand(ro[...,3:4].shape)), dim=-1))
    ax[torch.lt(torch.abs(ro[...,3]), 1.e-8)]  = _precision_check([ 0.0, 0.0, 1.0, 0.0 ], ro.dtype, ro.device)
    # return ax # here for axis angle pair

    """Step 3: Axis angle pair to quaternion."""
    c = torch.cos(ax[..., 3:4] * .5)
    s = torch.sin(ax[..., 3:4] * .5)
    qu = torch.where(torch.lt(torch.abs(ax[..., 3:4]), 1.e-6), _precision_check([1.0, 0.0, 0.0, 0.0], ax.dtype, ax.device), torch.cat((c, ax[..., :3] * s), dim=-1))

    # Converting Quaternion Order convention
    # DREAM3D convention is <xyz> s (<xyz> = imaginary vector component)
    # Convention used here is s <xyz>
    # This conversion is s,<xyz> to <xyz>,s
    if scalar_first is not True:
        # if qu.ndim == 1:
        #     qu = torch.unsqueeze(qu, dim=0)
        qu = torch.index_select(qu, -1, _device_check(torch.LongTensor([1, 2, 3, 0]), qu.device))
    # -------

    return qu

#---------- Core/Convention Functions ----------
def _cbrt(x):
    return torch.where(torch.lt(x, 0), -(torch.abs(x)**(1/3)), x**(1/3))

def _precision_check(value, datatype, devicetype=None):
    if torch.is_tensor(value) is False:
        value = torch.as_tensor(value)

    if datatype is torch.float32:
        if devicetype is None:
            return value.float()
        else:
            return _device_check(value.float(), devicetype)

    elif datatype is torch.float64:
        if devicetype is None:
            return value.double()
        else:
            return _device_check(value.double(), devicetype)
    else:
        return torch.tensor(value)

def _device_check(value, devicetype):
    if torch.is_tensor(value) is False:
        value = torch.as_tensor(value)

    if devicetype.type == 'cuda':
        return value.cuda()
    elif devicetype.type == 'cpu':
        return value.cpu()
    else:
        return value

def _get_tensor_pyramid_order(xyz,direction=None):
    """
    Get order of the coordinates, adapted to Pytorch Tensor form
    Depending on the pyramid in which the point is located, the order need to be adjusted.
    Parameters
    ----------
    xyz : torch.tensor
       coordinates of a point on a uniform refinable grid on a ball or
       in a uniform refinable cubical grid.
    References
    ----------
    D. Roşca et al., Modelling and Simulation in Materials Science and Engineering 22:075013, 2014
    https://doi.org/10.1088/0965-0393/22/7/075013
    """

    order = {'forward': _device_check(torch.LongTensor([[0,1,2],[1,2,0],[2,0,1]]), xyz.device),
             'backward': _device_check(torch.LongTensor([[0,1,2],[2,0,1],[1,2,0]]), xyz.device)}

    p = torch.where(torch.le(torch.maximum(torch.abs(xyz[...,0]),torch.abs(xyz[...,1])), torch.abs(xyz[...,2])), 0,
                 torch.where(torch.le(torch.maximum(torch.abs(xyz[...,1]),torch.abs(xyz[...,2])), torch.abs(xyz[...,0])), 1, 2))

    return order[direction][p]
