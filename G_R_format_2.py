#!/usr/bin/python
# -*- coding: utf-8 -*-

############################################
# Testing the generalized CCSD
# RDM1 and t1 amp -> comparaison 
############################################

import numpy as np
import random
import pyscf
from pyscf import scf,gto,ao2mo,cc
from pyscf.cc import gccsd
from pyscf.tools import cubegen

mol=gto.Mole()

#mol.atom = [
#    [8, (0., 0., 0.)],
#    [1, (0., -0.757, 0.587)],
#    [1, (0., 0.757, 0.587)]]
mol.atom="""
H 0 0 0
H 0 0 1.
"""
mol.basis = '6-31g'

mol.build()                           # build mol object

############################
# Restricted calculation
############################

rmf=scf.RHF(mol)
rmf.kernel()                
mo_coeff = rmf.mo_coeff

myrcc = pyscf.cc.RCCSD(rmf).run()
rdm1_rcc = myrcc.make_rdm1()

################################
# Generalized version
################################

mfg=scf.GHF(mol)
mfg.kernel()                           # do scf calculation

mo_coeff_g = mfg.mo_coeff              

mygcc = pyscf.cc.GCCSD(mfg).run()
rdm1_gcc = mygcc.make_rdm1()

new_mfg = scf.addons.convert_to_ghf(rmf)
new_mo_coeff_g = new_mfg.mo_coeff 
new_mygcc = pyscf.cc.GCCSD(mfg).run()
new_rdm1_gcc = mygcc.make_rdm1()

def convert_r_to_g_coeff(mo_coeff):
    
    dim = mo_coeff.shape[0]*2
    new_coeff = np.zeros((dim, dim))
    new_coeff[0:dim//2,0::2] = mo_coeff
    new_coeff[dim//2:,1::2] = mo_coeff

    return new_coeff

def convert_g_to_r_coeff(mo_coeff):
    
    dim = mo_coeff.shape[0] // 2
    new_coeff = mo_coeff[:dim,0::2] 

    return new_coeff

print('################################')
print('# Compare RHF, GHF and RHF->GHF ')
print('################################')
print()

print('---------')
print('mo_coeff')
print('---------')

print('R')
print(mo_coeff)
print('G')
print(mo_coeff_g)
print('R->G')
print(new_mo_coeff_g)
print('transform R -> G')
mo_coeff_trans = convert_r_to_g_coeff(mo_coeff)
print(np.subtract(new_mo_coeff_g, mo_coeff_trans))
print()
print('transform G -> R')
mo_coeff_trans = convert_g_to_r_coeff(new_mo_coeff_g)
print(np.subtract(mo_coeff, mo_coeff_trans))
print()

print('---------')
print('orbspin')
print('---------')

print('R->G')
print(new_mo_coeff_g.orbspin)

print()

print('---------')
print('t1 coeff')
print('---------')
print('R')
print(myrcc.t1)
print('G')
print(mygcc.t1)
print('R->G')
print(new_mygcc.t1)

print()
print(mo_coeff_g.shape, mol.nbas)
