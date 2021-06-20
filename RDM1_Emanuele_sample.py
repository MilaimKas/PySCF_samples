'''
Example of one-electron propetries calculation with PySCF

p,q =  indices in MOs basis
r,s = indices in AOs basis

math: rdm1 = <Psi|a_p^{dag}a_q|Psi>

The make_rdm1 PySCF's function constructs the rdm1 matrix
- for Psi=HF, rdm1 is given in AOS basis by default
- for Psi=CC, rdm1 is given in MOs basis by default

The .cube file contains the electronic density on a grid
I use the program VMD to display densities but there are numerous free
programms that can do that, pick up the one you want ;)
'''

# import modules
# --------------------------
import numpy as np
from pyscf.tools import cubegen
from pyscf import scf,gto,ao2mo,cc

# molecule definition (here H2O)
# -------------------------------
mol = gto.M(atom=[
    [8, (0., 0., 0.)],
    [1, (0., -0.757, 0.587)],
    [1, (0., 0.757, 0.587)]])
mol.basis = '6-31+g*'

# store kinetic energy integrals
# math: Ek_int = <r|Ek|s>
Ek_int = mol.intor_symmetric('int1e_kin')

# Hartree Fock calculation
# --------------------------
mf = scf.RHF(mol).run()

# store MO coefficients and its inverse
mo = mf.mo_coeff
mo_inv = np.linalg.inv(mo)

# construct HF-rdm1 in AOs basis
rdm1_hf_ao = mf.make_rdm1()

# transform into MO basis 
# --> it should be a diagonal matrix with occupation number 
rdm1_hf_mo = np.einsum('pr,rs,qs->pq',mo_inv,rdm1_hf_ao,mo_inv.conj())

# CC calculation
# ------------------------
mycc = cc.RCCSD(mf)
mycc.kernel()

# construct CC-rdm1 in MOs basis
rdm1_cc_mo = mycc.make_rdm1()

# transform to AOs basis
# math: gamma_rs = mo_rp gamma_pq mo_sq*
rdm1_cc_ao = np.einsum('rp,pq,sq->rs', mo, rdm1_cc_mo, mo.conj())

# One-electron properties calculation
# --------------------------------------

# Ekin
# math: Ek = sum_{rs} Ek_int_{rs} gamma_{sr} 
Ek_cc = np.einsum('rs,sr',Ek_int,rdm1_cc_ao)
Ek_hf = np.einsum('rs,sr',Ek_int,rdm1_hf_ao)

# Electronic density
# Generate cube file using PySCF's cubegen module
# math: rho(grid) = sum_{rs} AO_{r}(grid) AO_{s}*(grid) gamma_{sr}
cubegen.density(mol, 'C2H2_hf.cube',rdm1_hf_ao) 
cubegen.density(mol, 'C2H2_CC.cube', rdm1_cc_ao)   

print('Kinetic energy')
print('HF   --> ', Ek_hf)
print('CCSD --> ', Ek_cc)


