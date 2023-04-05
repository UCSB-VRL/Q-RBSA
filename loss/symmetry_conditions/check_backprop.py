from quats import Quat, rot_dist_w_syms, rand_quats
from symmetries import hcp_syms, fcc_syms


N = 7

use_torch = False

q_est = rand_quats(N,use_torch=True)
q_gt = rand_quats(N,use_torch=True)

q_est.X.requires_grad = True

print(q_gt)

dists_hcp = rot_dist_w_syms(q_est,q_gt,hcp_syms.to_torch())
dists_fcc = rot_dist_w_syms(q_est,q_gt,fcc_syms.to_torch())

loss_hcp = (dists_hcp**2).sum()
loss_fcc = (dists_fcc**2).sum()


loss_hcp.backward()

print(q_est.X.grad)



