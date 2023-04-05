import numpy as np
import matplotlib.pyplot as plt


square_path = np.array([[0,0],[0,1],[1,1],[1,0],[0,0]])

l = np.sqrt(3)/2
rhomb_path = np.array([[0,0],[1,0],[1/2,l],[-1/2,l],[0,0]])

square_axes = np.eye(3)
rhomb_axes = np.eye(3)
rhomb_axes[1] = [-1/2,l,0]



def plot_axes(ax,R):
    colors = ['r','g','b']
    for c,x in zip(colors,R):
        xv = np.hstack((np.zeros((3,1)),x[:,None]))
        ax.plot(*xv,color=c)

def plot_axes2(ax,R):
    colors = ['r','g','b']
    print(R)
    for c,x in zip(colors,R):
        print(x)
        ax.plot(*x,color=c,marker='o')

def path2prism(path):
    bot = np.hstack((path,np.zeros((path.shape[0],1))))
    top = bot.copy()
    top[:,2] = 1
    P = np.hstack((bot,top,bot)).reshape((-1,3))
    P = np.vstack((P,top))
    return P


def cube_path():
    sq_bot = np.hstack((square_path,np.zeros((5,1))))
    sq_top = sq_bot.copy()
    sq_top[:,2] = 1
    P = np.hstack((sq_bot,sq_top,sq_bot)).reshape((-1,3))
    P = np.vstack((P,sq_top)) 
    return P




if __name__=='__main__':


    r = np.sqrt(2)
    fig = plt.figure()

    axes = [fig.add_subplot(1,2,i,projection='3d') for i in [1,2]]

    for ax in axes:
        ax.set_xlim(-r,r)
        ax.set_ylim(-r,r)
        ax.set_zlim(-r,r)

    axes[0].plot(*path2prism(square_path).T,'k')
    axes[1].plot(*path2prism(rhomb_path).T,'k')


    plot_axes(axes[0],square_axes)
    plot_axes(axes[1],rhomb_axes)


    plt.show()


    #fig,axes = plt.subplots(1,2)
    

