from pyscf import gto, scf, adc, lib
import numpy as np


def get_dyson(atm='Ne'):

    mol = gto.M(atom=atm, basis='aug-cc-pvdz')
    mol.spin = 0
    mf = scf.RHF(mol).run()

    myadc = adc.ADC(mf)
    myadc.method = "adc(3)"
    myadc.method_type = "ip"
    myadc.kernel(nroots=1)

    dyson_orb = myadc.compute_dyson_mo()

    return dyson_orb.flatten(), mol


def get_orb_grid(rf, rn, tn, pn, atm='Ne'):

    coeff, mol = get_dyson(atm=atm)

    dim = rn * tn * pn
    if dim > 8000:
        raise ValueError("Number of grid points too large")

    # get spherical grid
    R = np.linspace(0, rf, rn)
    T = np.linspace(0, 2*np.pi, tn)
    P = np.linspace(0, np.pi, pn)
    coords_sphe = lib.cartesian_prod([R, T, P])  # dim x 3 array corresponding to coordinates

    # transform into cart grid
    coords_cart = np.zeros_like(coords_sphe)  #
    coords_cart[:, 0] = coords_sphe[:, 0] * np.sin(coords_sphe[:, 2]) * np.cos(coords_sphe[:, 1])  # x
    coords_cart[:, 1] = coords_sphe[:, 0] * np.sin(coords_sphe[:, 2]) * np.sin(coords_sphe[:, 1])  # y
    coords_cart[:, 2] = coords_sphe[:, 0] * np.cos(coords_sphe[:, 2])                              # z

    # Compute density
    ao = mol.eval_gto("GTOval", coords_cart)
    orb_on_grid = np.dot(ao, coeff)

    return orb_on_grid, coords_sphe


orb_density, coord = get_orb_grid(3., 10, 5, 4)
print(orb_density.shape)
print(coord.shape)

