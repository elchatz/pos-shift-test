import numpy as np
import pythtb

def model_ferroaxial_pythtb(mu=0.0, t=1.0, tp=1.0, Delta=1.0, t2=0.5, t3=1.0, a=1.0, c=1.0, 
                            spin=False, t_spin=0.0,
                            rot_deg=0):

    rot_alpha = rot_deg * np.pi/180  # Rotation angle in radians
    rot_mat = np.array([[np.cos(rot_alpha), -np.sin(rot_alpha), 0],
                        [np.sin(rot_alpha),  np.cos(rot_alpha), 0],
                        [0,                  0,                 1]])

    a_latt = a*np.sqrt(3)
    lat_0 = np.array([[a_latt,-0,0], [-a_latt/2, a_latt*np.sqrt(3)/2, 0], [0, 0, c]])
    lat = np.dot(lat_0, rot_mat.T)  # Rotate lattice vectors

    orb = np.array([[0, 0, 0] , [1/3, 2/3, 0]])  
    
    #lattice = pythtb.Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1, 2])
    #my_model = pythtb.TBModel(lattice)

    my_model = pythtb.tb_model(3, 3, lat, orb, nspin=2 if spin else 1)
    my_model.set_onsite([Delta - mu, -Delta - mu])

    
    for shift in [[0,0,0], [0, -1,0], [-1,-1,0]]:
        # nearest neighbor hoppings in-plane (t)
        my_model.set_hop(t, 0, 1, shift)
        # Vertical Hoppings (tpz)
        if tp != 0:
            for z in [1, -1]:
                my_model.set_hop( tp, 0, 1, [shift[0], shift[1],  z])


    if t3 != 0:
        # Third Neighbor Hoppings (tp)
        my_model.set_hop( t3, 1, 0, [ 2,  2,  0]) 
        my_model.set_hop( t3, 1, 0, [ 0, -1,  0])
        my_model.set_hop( t3, 1, 0, [-1,  1,  0])
        my_model.set_hop(-t3, 0, 1, [ 1,  1,  0]) 
        my_model.set_hop(-t3, 0, 1, [-2, -1,  0])
        my_model.set_hop(-t3, 0, 1, [ 0, -2,  0])


    if spin:
        sigma_0 = np.array([1., 0., 0., 0])
        sigma_z = np.array([0., 0., 0., 1])
    else:
        sigma_0 = 1
        sigma_z = 0

    for shift in [[0,1,0], [1,0,0], [-1,-1,0]]: # Skipping [0,0,0] for diagonal
        for i, sign in enumerate([1, -1]):
            # In-plane second neighbor hoppings (t2)
            hop = t2*sigma_0 + 1j*t_spin*sign*sigma_z
            my_model.set_hop(hop, i, i, shift)

    return my_model

def visualise(model, filename, title=None):
    (fig,ax)=pythtb.visualization.tbmodel.plot_tbmodel(model, proj_plane=(0,1))
    ax.set_title(title)
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    fig.tight_layout()
    fig.savefig(f"visualize_model_{filename}.pdf")
    