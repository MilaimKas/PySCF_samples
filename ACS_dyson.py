from pyscf import gto, scf, adc
from pyscf.tools import cubegen

mol = gto.M(atom='Ne', basis='aug-cc-pvdz')
mol.spin  = 0
mf = scf.RHF(mol).run()

myadc = adc.ADC(mf)
myadc.method = "adc(3)"
myadc.method_type = "ip"
eip, vip, pip, xip = myadc.kernel(nroots = 1)

print(mol.nao)
print(eip)
# alternative
# eip,vip,pip,xip,ip_es = myadc.ip_adc()
# eea,vea,pea,xea,ea_es = myadc.ea_adc()

# compute dyson orb
dyson_orb = myadc.compute_dyson_mo()

cubegen.orbital(mol, 'Ne_dyson.cube', dyson_orb.flatten())
