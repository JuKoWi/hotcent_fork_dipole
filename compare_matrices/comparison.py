import numpy as np

dirpath = "./compare_matrices/7_benz"

hotcent_o = np.loadtxt(fname=dirpath+'/oversqr_hotcent.dat', skiprows=5)
hotcent_h = np.loadtxt(fname=dirpath+'/hamsqr1_hotcent.dat', skiprows=5)
dftb_h = np.loadtxt(fname=dirpath+'/hamsqr1.dat', skiprows=5)
dftb_o = np.loadtxt(fname=dirpath+'/oversqr.dat', skiprows=5)

same1 = np.allclose(hotcent_h, dftb_h,
                    rtol=1e-6
                    )
same2 = np.allclose(hotcent_o, dftb_o, 
                    rtol=1e-6
                    )

print(same1)
print(same2)
# print(hotcent_h-dftb_h)
# print(hotcent_o-dftb_o)