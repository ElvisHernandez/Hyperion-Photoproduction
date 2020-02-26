# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:54:36 2019

@author: elvis
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

# Mass of particles in GeV
omegaMass = 0.78265
KaonMass = 0.493677
ProtonMass = 0.938272
SigmaMass = 1.192642
LambdaMass = 1.115683
Lambda1405Mass = 1.4051
d1405 = 0.0505

# The following applies for all data sets 
# Use `loc[]` to select column `'A'`
# W is the center if mass energy, W = sqrt(s) measured in GeV
# ds_dcos = d(sigma)/dcos(theta) measured in microbarns (i think)
# ds_dcos_ran is the statistical error
# ds_dcos_sys is the systematic error

##### This is Trevor's data from the gamma p -> p omega
omega = "omega"
FILE_PATH = r"C:\Users\elvis\Desktop\Physics_math_stuff\Photoproduction_Reaction_Original\gamma_p_reaction_data"
W_o = []
ds_dcos_o = [] 
ds_dcos_ran_o = []
for j in range(1,110):
    data_ = pd.read_csv(FILE_PATH + '\Table%s.csv' % j, skiprows = 12 )
    ACE = np.array(data_.loc[:, 'COS(THETA(P=3,RF=CM))'])
    ds_dcos_vec = np.array(data_.loc[:,'D(SIG)/DCOS(THETA) [MUB]'])
    ds_dcos_ran_vec = np.array(data_.loc[:,'error +'])
    #this part retrieves the W values 
    data_W = pd.read_csv(FILE_PATH + '\Table%s.csv' % j, skiprows = 9, skipfooter = len(ACE) + 3, engine='python')  # this selects only row 11 in the data files, this is the row that contains the W value
    W_un = np.array(data_W.loc[:,])
    W_o.append(float(W_un[0][3][:5]))
    i = 0
    while True:
        if ACE[i] == 0.05:
            row_number = i
            break
        else:
            i += 1   
    ds_dcos_o.append(ds_dcos_vec[row_number])
    ds_dcos_ran_o.append(ds_dcos_ran_vec[row_number]) 
ds_dcos_o = np.array(ds_dcos_o)
ds_dcos_ran_o = np.array(ds_dcos_ran_o)
ds_dcos_sys_o = 0.07*np.array(ds_dcos_o)
# Best_s_range(omega,ProtonMass,0,0,W_o,ds_dcos_o,ds_dcos_ran_o,ds_dcos_sys_o)
# counting_rule_fit(omega,ProtonMass,0,5.45,0,W_o,ds_dcos_o,ds_dcos_ran_o,ds_dcos_sys_o)
#####

##### Sigma groundstate data 
Sigma = "Sigma^0"
data1 = pd.read_csv("Sigma groundstate.csv") # coverts the sigma csv file into a pandas dataframe object
W_sig = np.array(data1.loc[:,'"W"'])
ds_dcos_sig = np.array(data1.loc[:,'"dsig/dcos"'])
ds_dcos_ran_sig = np.array(data1.loc[:,'"stat err"'])
ds_dcos_sys_sig = np.array(data1.loc[:,'"sys err"'])
# Best_s_range(Sigma,SigmaMass,0,0,W_sig,ds_dcos_sig,ds_dcos_ran_sig,ds_dcos_sys_sig)

#####

##### Lambda groundstate data
Lambda =  "Lambda"
path2 = r"C:\Users\elvis\Desktop\Physics_math_stuff\Photoproduction_Reaction_Original\Lambda_groundstate"
W_l = []
ds_dcos_l =[]
ds_dcos_ran_l = []
ds_dcos_sys_l =[]
for j in range(1,120):
    data2 = pd.read_csv(path2 + '\Table%s.csv' % j, skiprows = 13 )
    ACE = np.array(data2.loc[:, 'COS(THETA(P=3,RF=CM)'])
    ds_dcos_vec = np.array(data2.loc[:,'D(SIG)/DCOS(THETA(P=3,RF=CM) [MUB]'])
    ds_dcos_ran_vec = np.array(data2.loc[:,'stat +'])
    ds_dcos_sys_vec = np.array(data2.loc[:,'sys +'])
    #this part retrieves the W values 

    data_W = pd.read_csv(path2 + '\Table%s.csv' % j, skiprows = 10, skipfooter = len(ACE) + 3, engine='python')  # this selects only row 11 in the data files, this is the row that contains the W value
    W_vec = np.array(data_W.loc[:,])
    W_l.append(float(W_vec[0][3][:5]))
    i = 0
    while True:
        if ACE[i] == 0:
            row_number = i
            break
        else:
            i += 1 
    ds_dcos_l.append(ds_dcos_vec[row_number])
    ds_dcos_ran_l.append(ds_dcos_ran_vec[row_number]) 
    ds_dcos_sys_l.append(ds_dcos_sys_vec[row_number])
ds_dcos_l = np.array(ds_dcos_l)
ds_dcos_ran_l = np.array(ds_dcos_ran_l)
ds_dcos_sys_l = np.array(ds_dcos_sys_l)

##### Lambda(1405) 
# the prefix P means cos(theta) = 0.15,
# N means cos(theta) = -0.15

# for lambda(1405) for cosTheta = 0.15 for SIGMA+ PI-
Lambda1405PN = "Lambda(1405) < \Sigma^+\pi^-"
W_1405PPN = np.array([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
Pds_dcos_1405PPN = np.array([0.1214,0.1637,0.0620,0.0424,0.0296,0.0102,0.0055,0.0038,0.0032])
Pds_dcos_ran_1405PPN = np.array([0.0056,0.0056,0.0039,0.0024,0.0018,0.0010,0.0007,0.0006,0.0004])
Pds_dcos_sys_1405PPN = 0.116*Pds_dcos_1405PPN

# for lambda(1405) for cosTheta = 0.05 for SIGMA+ PI-
W_1405PPN = np.array([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
ds_dcos_1405PPN = np.array([0.1074,0.1510,0.0456,0.0273,0.0148,0.0082,0.0036,0.0043,0.0019])
ds_dcos_ran_1405PPN = np.array([0.0048,0.0058,0.0042,0.0021,0.0015,0.0010,0.0007,0.0006,0.0004])
ds_dcos_sys_1405PPN = 0.116*ds_dcos_1405PPN

# for lambda(1405) for cosTheta = -0.05 for SIGMA+ PI-
W_1405NPN = np.array([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
ds_dcos_1405NPN = np.array([0.0893,0.1420,0.0349,0.0129,0.0118,0.0061,0.0044,0.0030,0.0026])
ds_dcos_ran_1405NPN = np.array([0.0058,0.0074,0.0046,0.0021,0.0018,0.0010,0.0006,0.0005,0.0006])
ds_dcos_sys_1405NPN = .116*ds_dcos_1405NPN

# for lambda(1405) for cosTheta = -0.15 for SIGMA+ PI-
W_1405NPN = np.array([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
Nds_dcos_1405NPN = np.array([0.1027,0.1249,0.0220,0.0264,0.0117,0.0059,0.0046,0.0030,0.0023])
Nds_dcos_ran_1405NPN = np.array([0.0067,0.0068,0.0036,0.0026,0.0019,0.0012,0.0008,0.0006,0.0005])
Nds_dcos_sys_1405NPN = .116*Nds_dcos_1405NPN

# for lambda(1405) for cosTheta = 0.15 for SIGMA0 PI0
Lambda140500 = "Lambda(1405) < \Sigma^0\pi^0"
W_1405P00 = np.array([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
Pds_dcos_1405P00 = np.array([0.1439,0.1567,0.0710,0.0351,0.0221,0.0151,0.0058,0.0036,0.0035])
Pds_dcos_ran_1405P00 = np.array([0.0131,0.0106,0.0077,0.0047,0.0040,0.0026,0.0016,0.0012,0.0012])
Pds_dcos_sys_1405P00 = .116*Pds_dcos_1405P00

# for lambda(1405) for cosTheta = 0.05 for SIGMA0 PI0
W_1405P00 = np.array([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
ds_dcos_1405P00 = np.array([0.1481,0.1558,0.0842,0.0215,0.0153,0.0085,0.0041,0.0040,0.0033])
ds_dcos_ran_1405P00 = np.array([0.0146,0.0104,0.0079,0.0044,0.0032,0.0021,0.0014,0.0015,0.0013])
ds_dcos_sys_1405P00 = .116*ds_dcos_1405P00

# for lambda(1405) for cosTheta = -0.05 for SIGMA0 PI0
W_1405N00 = np.array([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
ds_dcos_1405N00 = np.array([0.0369,0.1679,0.0622,0.0229,0.0178,0.0100,0.0114,0.0015,0.0040])
ds_dcos_ran_1405N00 = np.array([0.0086,0.0181,0.0103,0.0060,0.0042,0.0028,0.0026,0.0013,0.0014])
ds_dcos_sys_1405N00 = .116*ds_dcos_1405N00

# for lambda(1405) for cosTheta = -0.15 for SIGMA0 PI0
W_1405N00 = np.array([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
Nds_dcos_1405N00 = np.array([0.0374,0.1017,0.0488,0.0245,0.0168,0.0082,0.0046,0.0032,0.0034])
Nds_dcos_ran_1405N00 = np.array([0.0081,0.0107,0.0117,0.0065,0.0063,0.0035,0.0025,0.0018,0.0018])
Nds_dcos_sys_1405N00 = .116*Nds_dcos_1405N00

# for lambda(1405) for cosTheta = 0.15 for SIGMA- PI+
Lambda1405NP = "Lambda(1405) < \Sigma^-\pi^+"
W_1405PNP = np.array([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
Pds_dcos_1405PNP = np.array([0.0712,0.0606,0.0334,0.0274,0.0176,0.0087,0.0022,0.0013,0.0019])
Pds_dcos_ran_1405PNP = np.array([0.0030,0.0025,0.0020,0.0017,0.0015,0.0010,0.0006,0.0004,0.0004])
Pds_dcos_sys_1405PNP = .116*Pds_dcos_1405PNP

# for lambda(1405) for cosTheta = 0.05 for SIGMA- PI+
W_1405PNP = np.array([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
ds_dcos_1405PNP = np.array([0.0668,0.0597,0.0346,0.0214,0.0134,0.0050,0.0027,0.0018,0.0014])
ds_dcos_ran_1405PNP = np.array([0.0031,0.0027,0.0020,0.0016,0.0014,0.0009,0.0006,0.0005,0.0004])
ds_dcos_sys_1405PNP = .116*ds_dcos_1405PNP

# for lambda(1405) for cosTheta = -0.05 for SIGMA- PI+
W_1405NNP = np.array([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
ds_dcos_1405NNP = np.array([0.0642,0.0561,0.0294,0.0201,0.0115,0.0054,0.0013,0.0023,0.0021])
ds_dcos_ran_1405NNP = np.array([0.0039,0.0035,0.0027,0.0019,0.0013,0.0009,0.0006,0.0005,0.0004])
ds_dcos_sys_1405NNP = .116*ds_dcos_1405NNP

# for lambda(1405) for cosTheta = -0.15 for SIGMA- PI+
W_1405NNP = np.array([2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
Nds_dcos_1405NNP = np.array([0.0596,0.0466,0.0238,0.0206,0.0055,0.0097,0.0079,0.0032,0.0029])
Nds_dcos_ran_1405NNP = np.array([0.0041,0.0030,0.0024,0.0020,0.0016,0.0014,0.0010,0.0006,0.0005])
Nds_dcos_sys_1405NNP = .116*Nds_dcos_1405NNP

# BaryonMass is the mass of the Y* in gamma + P --> K+ + Y*
# dBM is the uncertainty in the baryon mass, only applies to resonance states with wide mass peaks

def Theta_to_t(s,BaryonMass,dBM,costheta,dsig_dcos,dsig_dcos_ran_err,dsig_dcos_sys_err):
    if BaryonMass == ProtonMass:
        KaonMass = omegaMass 
    else: 
        KaonMass = 0.493677
    ##  All energies in this routine are center-of-mass energies!
    W = np.sqrt(s)
    Egamma = (s - ProtonMass**2)/W/2
    EK = (s + KaonMass**2 - BaryonMass**2)/W/2
    pK = np.sqrt(EK**2 - KaonMass**2)
    t = KaonMass**2 - 2*Egamma*EK + 2*Egamma*pK*costheta
    dt_dcos = 2*Egamma*pK
    dsig_dt = dsig_dcos/dt_dcos 
    if BaryonMass == SigmaMass:
        scale = 0.0828
    elif BaryonMass == LambdaMass:
        scale = 0.0808
    else:
        scale = 0
    #dsig_dt_ran_err = dsig_dcos_ran_err/dt_dcos
    dsig_dt_ran_err = np.sqrt((dsig_dcos_ran_err/dt_dcos)**2 + (0.5*dsig_dcos*EK*BaryonMass*dBM/(W*Egamma*pK**3))**2)
    dsig_dt_sys_err = dsig_dcos_sys_err/dt_dcos
    ScaleError = scale*dsig_dt
    PointSysError = np.sqrt(dsig_dt_sys_err**2-ScaleError**2)
    dsig_dt_err = np.sqrt(dsig_dt_ran_err**2+PointSysError**2)
    #dsig_dt_err = np.sqrt(dsig_dt_ran_err**2 + dsig_dt_sys_err**2)
    return t,dsig_dt,dsig_dt_err

# z_alpha = stats.norm.ppf(alpha,loc = 0, scale = 1) for standard normal distribution
def BCa(Method,n,alpha,N_est,s_fit,dfc_fit,adj_res,ordered_list):
    i = 0
    while ordered_list[i] < N_est:
        i += 1
        if i == n-1:
            break
    p = i 
    b = stats.norm.ppf(p/n,loc=0,scale=1)
    N_i = []
    def f(s,A,N):
        return(A*s**(-N))
    
    if Method == 'sp': # np stands for semi-parametric   
        for i in range(len(adj_res)):
            adj_res_i = np.delete(adj_res,i)
            s_fit_i = np.delete(s_fit,i)
            dfc_fit_i = np.delete(dfc_fit,i)
            popt, pcov = curve_fit(f, s_fit_i, dfc_fit_i, sigma = adj_res_i, maxfev = 1000000)
            A,N = popt
            N_i.append(N)
    else:  
        print("Sorry m8, no luck yet.")
        return()
    N_i = np.array(N_i)
    N_i_mean = np.mean(N_i)
    acc = (sum((N_i_mean - N_i)**3) )/(6* (sum((N_i_mean - N_i)**2))**(3/2) )
    
    z_alpha = stats.norm.ppf(alpha,loc = 0, scale = 1) 
    z_up = stats.norm.ppf(1-alpha,loc = 0, scale = 1) 
    alpha1 = b + (b + z_alpha)/(1 - acc*(b + z_alpha))
    alpha_1 = stats.norm.cdf(alpha1, loc = 0, scale = 1)
    alpha2 = b + (b + z_up)/(1 - acc*(b + z_up))
    alpha_2 = stats.norm.cdf(alpha2, loc = 0, scale = 1)
    lo = int(alpha_1*n)
    up = int(alpha_2*n)
    print("The %s percent CI is [%s,%s]" % (1-alpha,ordered_list[lo],ordered_list[up]) )
    return(ordered_list[lo],ordered_list[up])