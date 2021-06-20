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

#mol.atom="""
#H 0.0 0.0 0.0
#H 0.0 0.0 1.0
#"""
mol.atom = [
    [8, (0., 0., 0.)],
    [1, (0., -0.757, 0.587)],
    [1, (0., 0.757, 0.587)]]
#mol.atom="""
#H 0 0 0
#H 0 0 1.
#"""
mol.basis = '6-31g'

mol.build()                           # build mol object

############################
# Restricted calculation
############################

rmf=scf.RHF(mol)
rmf.kernel()                           # do scf calculation

mo_coeff = rmf.mo_coeff           # Molecular orbital (MO) coefficients (matrix where rows are atomic orbitals (AO) and columns are MOs)
mo_coeff_inv = np.linalg.inv(mo_coeff)
mo_ene   = rmf.mo_energy          # MO energies (vector with length equal to number of MOs)
mo_occ   = rmf.mo_occ             # MO occupancy (vector with length equal to number of MOs)
mocc     = mo_coeff[:,mo_occ>0]  # Only take the mo_coeff of occupied orb
mvir     = mo_coeff[:,mo_occ==0] # Only take the mo_coeff of virtual orb
nocc     = mocc.shape[1]         # Number of occ MOs in HF
nvir     = mvir.shape[1]         # Number of virtual MOS in HF
dim      = mol.nao_nr()          # number of AO --> size of the basis

dm1_hf_r      = rmf.make_rdm1()

myrcc = pyscf.cc.CCSD(rmf).run()
rdm1_rcc = myrcc.make_rdm1()

################################
# Generalized version
################################

mfg=scf.GHF(mol)
mfg.kernel()                           # do scf calculation

mo_coeff_g = mfg.mo_coeff              
mo_ene_g   = mfg.mo_energy             
mo_occ_g   = mfg.mo_occ                
mocc_g     = mo_coeff_g[:,mo_occ_g>0]  
mvir_g     = mo_coeff_g[:,mo_occ_g==0] 
nocc_g     = mocc_g.shape[1]        
nvir_g     = mvir_g.shape[1]       
dim_g      = mol.nao_nr()          

dm1_hf_g   = mfg.make_rdm1()

mygcc = pyscf.cc.GCCSD(mfg).run()
rdm1_gcc = mygcc.make_rdm1()

new_mfg = scf.addons.convert_to_ghf(rmf)
new_mo_coeff_g = new_mfg.mo_coeff
new_mygcc = pyscf.cc.GCCSD(new_mfg).run()
new_rdm1_gcc = new_mygcc.make_rdm1()

################################
# Unrestricted version
################################

mfu=scf.UHF(mol)
mfu.kernel()

mo_coeff_u = mfu.mo_coeff
mo_ene_u   = mfu.mo_energy
mo_occ_u   = mfu.mo_occ
#mocc_u     = mo_coeff_u[:,mo_occ_u>0]  
#mvir_u     = mo_coeff_u[:,mo_occ_u==0] 
#nocc_u     = mocc_u.shape[1]
#nvir_u     = mvir_u.shape[1]

dm1_hf_u   = mfu.make_rdm1()

myucc = pyscf.cc.UCCSD(mfu).run()
rdm1_ucc = mygcc.make_rdm1()


# ------------------------------------------
# Compare 
# -------------------------------------------

# CC rdm1 in AOs
rdm1_gcc = np.einsum('pi,ij,qj->pq', mo_coeff_g, rdm1_gcc, mo_coeff_g.conj()) 
rdm1_rcc = np.einsum('pi,ij,qj->pq', mo_coeff, rdm1_rcc, mo_coeff.conj())
new_rdm1_gcc = np.einsum('pi,ij,qj->pq', new_mo_coeff_g, new_rdm1_gcc,new_mo_coeff_g.conj())
#rdm1_ucc = np.einsum('pi,ij,qj->pq', mo_coeff_u, rdm1_ucc, mo_coeff_u.conj())

#######################
# Conversion functions
#######################

def convert_g_to_r_amp(amp):
    '''
    Converts G format single amplitudes into R format

    :param amp:
    :return:
    '''

    #r_amp = np.zeros((amp.shape[0]//2,amp.shape[1]//2))
    #for i in range(0,amp.shape[0],2):
    #    for j in range(0,amp.shape[1],2):
    #        r_amp[i//2,j//2] = amp[i,j]
    
    if amp.ndim == 2:
       tmp = np.delete(amp, np.s_[1::2], 0)
       r_amp = np.delete(tmp, np.s_[1::2], 1)
    else:
        dim = amp.shape[0]+amp.shape[2]
        orbspin = np.zeros(dim, dtype=int)
        orbspin[1::2] = 1
        r_amp = cc.addons.spin2spatial(amp, orbspin)[1] # t2ab
    return r_amp

def convert_r_to_g_amp(amp):
    '''
    Converts single amplitudes in restricted format into generalized, spin-orbital format
    amp must be given in nocc x nvir shape
    NOTE: PySCF as several functions to perform the conversion within CC, CI or EOM classes: spatial2spin()

    :param amp: single amplitudes (t in restricted format
    :return: amp in generalized spin-orb format
    '''
    
    if amp.ndim == 2:
        g_amp = np.zeros((amp.shape[0]*2,amp.shape[1]*2))
        for i in range(amp.shape[0]):
            for j in range(amp.shape[1]):
                a = amp[i,j]
                g_amp[i*2:i*2+2,j*2:j*2+2] = np.diag(np.asarray([a,a]))
    elif amp.ndim == 4:
        g_amp = cc.addons.spatial2spin(amp)

    return g_amp

def convert_g_to_ru_rdm1(rdm1_g):
    '''
    Transform generalised rdm1 to R and U rdm1

    :param rdm1_g: one-electron reduced density matrix in AOs basis
    :return: rdm1 in R and U format
    '''

    nao = rdm1_g.shape[0]//2

    rdm_a = rdm1_g[:nao,:nao]
    rdm_b = rdm1_g[nao:,nao:]

    rdm_u = (rdm_a,rdm_b)

    rdm_r = rdm_a + rdm_b

    return rdm_r,rdm_u

def convert_u_to_g_rdm1(rdm_u):
    """
    convert U rdm1 to G rdm1

    :param rdm_u: urestricted format rdm1 in AOs basis
    :return:
    """

    nao, nao = rdm_u[0].shape

    rdm_g = np.zeros((nao * 2, nao * 2))
    rdm_g[::2, ::2] = rdm_u[0]
    rdm_g[1::2, 1::2] = rdm_u[1]

    return rdm_g

def convert_r_to_g_rdm1(rdm_r):
    """
    convert R rdm1 to G rdm1

    :param rdm_r: restricted format rdm1 in AOs basis
    :return:
    """

    nao, nao = rdm_r.shape

    rdm_g = np.zeros((nao * 2, nao * 2))
    rdm_g[:nao, :nao] = 0.5*rdm_r
    rdm_g[nao:, nao:] = 0.5*rdm_r

    return rdm_g

print()
print("############")
print("CCSD RDM1")
print("############")
print()
print("GCCSD")
#print(rdm1_gcc)
print("RCCSD")
#print(rdm1_rcc)
print('R->G CCSD')
print()
print("transformed RCC-->GCC")
print('compare to GCC')
ans = np.subtract(convert_r_to_g_rdm1(rdm1_rcc),rdm1_gcc)
#print(ans)
print('norm= ',np.sum(ans))
print('compare to R->G CC')
ans = np.subtract(convert_r_to_g_rdm1(rdm1_rcc),new_rdm1_gcc)
#print(ans)
print('norm= ',np.sum(ans))

print()
print("transformed GCC-->RCC")
ans = np.subtract(convert_g_to_ru_rdm1(rdm1_gcc)[0],rdm1_rcc)
#print(ans)
print('norm=', np.sum(ans))
print("transformed R->G-->RCC")
ans = np.subtract(convert_g_to_ru_rdm1(new_rdm1_gcc)[0],rdm1_rcc)
#print(ans)
print('norm=', np.sum(ans))
print()

print("############")
print("CCSD t1 amp")
print("############")
print()
print('GCCSD')
#print(mygcc.t1)
print()
print('RCCSD')
#print(myrcc.t1)
print()
print('R->GCCSD')
#print(new_mygcc.t1)
print()
print("transformed RCC-->GCC")
ans = convert_r_to_g_amp(myrcc.t1)
#print(ans)
#print(ans.shape)
print('Compare to GCC')
print('diff= ', np.sum(np.subtract(ans, mygcc.t1)))
print('Compare to R->GCC')
print('diff= ', np.sum(np.subtract(ans, new_mygcc.t1)))

print()
print("PySCF")
ans = cc.addons.spatial2spin(myrcc.t1)
print('diff= ', np.sum(np.subtract(ans, mygcc.t1)))
print()
print("transformed GCC-->RCC")
ans = convert_g_to_r_amp(mygcc.t1)
#print(ans.shape)
print('diff= ', np.sum(np.subtract(ans,myrcc.t1)))
print("transformed R->GCC-->RCC")
ans = convert_g_to_r_amp(new_mygcc.t1)
#print(ans.shape)
print('diff= ', np.sum(np.subtract(ans,myrcc.t1)))

print()

print("############")
print("CCSD t2 amp")
print("############")
print()
print('GCCSD')
print(mygcc.t2.shape)
#print(mygcc.t2)
print()
print('RCCSD')
print(myrcc.t2.shape)
#print(myrcc.t2)
print()
print("transformed RCC-->GCC")
ans = convert_r_to_g_amp(myrcc.t2)
print(ans.shape)
print('Compare to GCC')
print('diff= ', np.sum(np.subtract(ans,mygcc.t2)))
print('Compare to R->GCC')
print('diff= ', np.sum(np.subtract(ans,new_mygcc.t2)))
print()
print("transformed GCC-->RCC")
ans = convert_g_to_r_amp(mygcc.t2)
print(ans.shape)
print('diff= ', np.sum(np.subtract(ans,myrcc.t2)))
print("transformed R->GCC-->RCC")
ans = convert_g_to_r_amp(new_mygcc.t2)
print(ans.shape)
print('diff= ', np.sum(np.subtract(ans,myrcc.t2)))

#print(ans)
print()

print()
print("###########################")
print(" G-EOM                     ")
print("###########################")
print()

eom = cc.eom_gccsd.EOMEE(mygcc)
e,r = eom.kernel()
r = eom.vector_to_amplitudes(r)

print('initial guess')
#print(eom.get_init_guess())
print()

print('G-EOM')
#print(r[0])
print()

print('R-EOM')
eom = cc.eom_rccsd.EOMEESinglet(myrcc)
e,r = eom.kernel()
r = eom.vector_to_amplitudes(r)

#print(r[0])

print()
print("###########################")
print(" MO coef                   ")
print("###########################")
print()
print('G')
#print(mo_coeff_g)
print('U')
#print(mo_coeff_u)
print('R')
#print(mo_coeff)
print('Orbspin')

new_mfg = scf.addons.convert_to_ghf(rmf)
#print(new_mfg.mo_coeff)
#orbspin = scf.addons.get_ghf_orbspin(new_mfg.mo_energy, new_mfg.mo_occ, False)
print(new_mfg.mo_coeff.orbspin)

print()
print("###############################")
print(" t1 coef from GHF and RHF->GHF ")
print("###############################")
print()

new_myccg = cc.GCCSD(new_mfg)
new_myccg.kernel()
#print(np.subtract(new_myccg.t1,mygcc.t1))

