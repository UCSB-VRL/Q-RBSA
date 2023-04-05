import numpy as np
from loss.symmetry_conditions.quats import Quat


#import pdb; pdb.set_trace()
#rotate by 0 or 180 degrees about x axis
hcp_r1 = Quat(np.eye(4)[:2])

# rotate about 0, 60, ... 300 degrees about z axis
hcp_r2 = np.zeros((6,4))
hcp_r2[:,0] = np.cos(np.arange(6)/6*np.pi)
hcp_r2[:,3] = np.sin(np.arange(6)/6*np.pi)
hcp_r2 = Quat(hcp_r2)
hcp_syms = hcp_r2.outer_prod(hcp_r1).transpose((1,0)).reshape(-1)


# rotate about diagonal on cube
fcc_r1 = np.zeros((3,4))
fcc_r1[:,0] = np.cos(2/3*np.pi*np.arange(3))
fcc_r1[:,1:] = (np.sin(2/3*np.pi*np.arange(3))/np.sqrt(3))[:,None]
fcc_r1 = Quat(fcc_r1)

# rotate by 0 or 180 degrees around x-axis
fcc_r2 = Quat(np.array(np.eye(4)[:2]))

fcc_r3 = np.zeros((4,4))
fcc_r3[:,0] = np.cos(np.pi/4*np.arange(4))
fcc_r3[:,3] = np.sin(np.pi/4*np.arange(4))
fcc_r3 = Quat(fcc_r3)

fcc_r12 = fcc_r2.outer_prod(fcc_r1)
fcc_syms = fcc_r3.outer_prod(fcc_r12).reshape(-1)




if __name__ == '__main__':

    from plotting_utils import *

    np.random.seed(1)
    q = np.random.normal(0,1,4)
    q /= np.linalg.norm(q)
    q1 = Quat(q)

    rhomb_wire = path2prism(rhomb_path)
    all_rots = q1.outer_prod(hcp_syms)
    all_wires = all_rots.rotate(rhomb_wire)
    all_axes = all_rots.rotate(rhomb_axes)

    def setup_axes(m,n):
        r = np.sqrt(2)
        fig = plt.figure()
        axes = [fig.add_subplot(m,n,i+1,projection='3d') for i in range(m*n)]
        for a in axes:
            a.set_xlim(-r,r)
            a.set_ylim(-r,r)
            a.set_zlim(-r,r)    
        return fig, axes

    fig, axes = setup_axes(3,4)

    for i, ax in enumerate(axes):

        ax.plot(*all_wires[0].T,color='#888')
        ax.plot(*all_wires[i].T,color='#000')
        plot_axes(ax,all_axes[i])


    square_wire = path2prism(square_path)
    all_rots = q1.outer_prod(fcc_syms)
    all_wires = all_rots.rotate(square_wire)
    all_axes = all_rots.rotate(square_axes)

    fig, axes = setup_axes(4,6)

    for i, ax in enumerate(axes):
        ax.plot(*all_wires[0].T,color='#888')
        ax.plot(*all_wires[i].T,color='#000')
        plot_axes(ax,all_axes[i])


    plt.show()



