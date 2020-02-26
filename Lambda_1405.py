# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:03:33 2019

@author: elvis
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from sklearn.utils import resample 
from scipy import stats
from scipy.interpolate import splev, splrep
import data as D
import MAIN as M

##  Fix font and stuff
#mpl.rc('font',size=15,family='cmr10',weight='normal')
#mpl.rc('text',usetex=True)

def fit(s,dfc,dfc_err):
    def f(s,A,N):
        return(A*s**(-N))
    popt, pcov = curve_fit(f,s[2:],dfc[2:], sigma = dfc_err[2:], maxfev = 1000000)
    A,N = popt
    A_err = np.sqrt(pcov[0][0])
    N_err = np.sqrt(pcov[1][1])
    print(popt)
    smooth = np.linspace(min(s[2:]),max(s[2:]),10000)
    #plt.plot(smooth,f(smooth,*popt), label = "A = %s $\pm$ %s,"
    #         "\n"
    #         r"N = %s $\pm$ %s" % (int(A),int(A_err),round(N,3),round(N_err,3)),color = 'red')
    #plt.xlabel(r'$s$')
    #plt.ylabel(r'$\frac{d\sigma}{dt}$')
    #plt.legend()
    #plt.errorbar(s, dfc, yerr = dfc_err, fmt = 'go', ecolor = 'r' )
    return(A,N)
# s_range(D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405PPN,D.ds_dcos_ran_1405PPN,D.ds_dcos_sys_1405PPN)
def s_range(Baryon_name,BaryonMass,dBM,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err):
    # BaryonMass is to indicate the particle mass
    # s_fit_range indicates that you only want to consider  s_fit_range <= s <= max {s} for the fit
    # mt = momentum_transfer(t), dfc = differential cross section (dsig/dt), dfc_err is the error in dfc    
    s = np.power(W,2)
    cosThetaMatrix = []
    for i in range(len(s)):
        cosThetaMatrix.append(0)
        cosThetaMatrix[i] = costheta
    cosThetaMatrix = np.array(cosThetaMatrix)
    mt1,dfc1,dfc_err1 = D.Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err)
    mt2,dfc2,dfc_err2 = D.Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,D.ds_dcos_1405NPN,D.ds_dcos_ran_1405NPN,D.ds_dcos_sys_1405NPN)
    mt3,dfc3,dfc_err3 = D.Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,D.ds_dcos_1405P00,D.ds_dcos_ran_1405P00,D.ds_dcos_sys_1405P00)
    mt4,dfc4,dfc_err4 = D.Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,D.ds_dcos_1405N00,D.ds_dcos_ran_1405N00,D.ds_dcos_sys_1405N00)
    mt5,dfc5,dfc_err5 = D.Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,D.ds_dcos_1405PNP,D.ds_dcos_ran_1405PNP,D.ds_dcos_sys_1405PNP)
    mt6,dfc6,dfc_err6 = D.Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,D.ds_dcos_1405NNP,D.ds_dcos_ran_1405NNP,D.ds_dcos_sys_1405NNP)
    
    print(dfc5/dfc6)
    print
    plt.subplot(2,3,1)
    fit(s,dfc1,dfc_err1)
    plt.title(r'$cos \theta = 0.05$, $\Lambda(1405) < \Sigma^+ \pi^-$ ')
    plt.ylabel(r'$\frac{d\sigma}{dt}$')
    
    plt.subplot(2,3,2)
    fit(s,dfc3,dfc_err3)
    plt.title(r'$cos \theta = 0.05$, $\Lambda(1405) < \Sigma^0 \pi^0$ ')
    
    plt.subplot(2,3,3)
    fit(s,dfc5,dfc_err5)
    plt.title(r'$cos \theta = 0.05$, $\Lambda(1405) < \Sigma^- \pi^+$ ')
    
    plt.subplot(2,3,4)
    fit(s,dfc2,dfc_err2)
    plt.title(r'$cos \theta = -0.05$') #, $\Lambda(1405) < \Sigma^+ \pi^-$ ')
    plt.ylabel(r'$\frac{d\sigma}{dt}$')
    plt.xlabel(r'$s$')
    
    plt.subplot(2,3,5)
    fit(s,dfc4,dfc_err4)
    plt.title(r'$cos \theta = -0.05$') #, $\Lambda(1405) < \Sigma^0 \pi^0$ ')
    plt.xlabel(r'$s$')
    
    plt.subplot(2,3,6)
    fit(s,dfc6,dfc_err6)
    plt.title(r'$cos \theta = -0.05$')# , $\Lambda(1405) < \Sigma^- \pi^+$ ')
    plt.xlabel(r'$s$')
    plt.show()

# interp(D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405PPN,D.ds_dcos_ran_1405PPN,D.ds_dcos_sys_1405PPN)
def interp(Baryon_name,BaryonMass,dBM,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err):
    s = np.power(W,2)
    cosThetaMatrix = []
    for i in range(len(s)):
        cosThetaMatrix.append(0)
        cosThetaMatrix[i] = costheta
    cosThetaMatrix = np.array(cosThetaMatrix)
    mt1,dfc1,dfc_err1 = D.Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err)
    
    # interpolating the cross section values
    spl1 = splrep(s,dfc1, w = 1/dfc_err1, k = 3)#, full_output = 1) #, task = -1, t = s[1:-1])
    s1 = np.linspace(min(s),max(s),100)
    y1 = splev(s1,spl1)
    
    #interpolating the cross section errors
    spl2 = splrep(s,dfc_err1, k = 3)
    s2 = s1
    y2 = splev(s2,spl2)
    
    # normal distribution from which to choose cross section errors
    #err = np.random.normal(y1,y2)
    err = y2
    
    #fig1 = plt.figure()
    #####plt.errorbar(s, dfc1, yerr = dfc_err1, fmt = 'go', ecolor = 'r', label = r'Experimental values')######
    #plt.plot(s1,y1,'o')
    ######plt.errorbar(s1,y1, yerr = err, fmt = 'b.', ecolor = 'black', label = r'Interpolated values')######
    #plt.plot(s,dfc_err1, 'bo')
    #plt.show()
    
    #CCR fit 
    s_beg = s[2]
    i = 0
    while s1[i] < s_beg:
        i += 1
    s_fit = s1[i:]
    dfc_fit = y1[i:]
    dfc_err_fit = err[i:]
    
    #fig2 = plt.figure()
    fit(s_fit,dfc_fit,dfc_err_fit)
    ######plt.legend()######
    #plt.show()
    return(s_fit,dfc_fit,dfc_err_fit)

# inter_graph(D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405PPN,D.ds_dcos_ran_1405PPN,D.ds_dcos_sys_1405PPN)
def inter_graph(Baryon_name,BaryonMass,dBM,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err):
    
    plt.subplot(2,3,1)
    interp(D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405PPN,D.ds_dcos_ran_1405PPN,D.ds_dcos_sys_1405PPN)
    plt.title(r'$cos \theta = 0.05$, $\Lambda(1405) < \Sigma^+ \pi^-$ ')
    plt.ylabel(r'$\frac{d\sigma}{dt}$')
    
    plt.subplot(2,3,2)
    interp(D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405P00,D.ds_dcos_ran_1405P00,D.ds_dcos_sys_1405P00)
    plt.title(r'$cos \theta = 0.05$, $\Lambda(1405) < \Sigma^0 \pi^0$ ')
    
    plt.subplot(2,3,3)
    interp(D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405PNP,D.ds_dcos_ran_1405PNP,D.ds_dcos_sys_1405PNP)
    plt.title(r'$cos \theta = 0.05$, $\Lambda(1405) < \Sigma^- \pi^+$ ')
    
    plt.subplot(2,3,4)
    interp(D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405NPN,D.ds_dcos_ran_1405NPN,D.ds_dcos_sys_1405NPN)
    plt.title(r'$cos \theta = -0.05$') #, $\Lambda(1405) < \Sigma^+ \pi^-$ ')
    plt.ylabel(r'$\frac{d\sigma}{dt}$')
    plt.xlabel(r'$s$')
    
    plt.subplot(2,3,5)
    interp(D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405N00,D.ds_dcos_ran_1405N00,D.ds_dcos_sys_1405N00)
    plt.title(r'$cos \theta = -0.05$') #, $\Lambda(1405) < \Sigma^0 \pi^0$ ')
    plt.xlabel(r'$s$')
    
    plt.subplot(2,3,6)
    interp(D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405NNP,D.ds_dcos_ran_1405NNP,D.ds_dcos_sys_1405NNP)
    plt.title(r'$cos \theta = -0.05$')# , $\Lambda(1405) < \Sigma^- \pi^+$ ')
    plt.xlabel(r'$s$')
    plt.show()

# SP_boot(10000,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405PPN,D.ds_dcos_ran_1405PPN,D.ds_dcos_sys_1405PPN)
def SP_boot(n,Baryon_name,BaryonMass,dBM,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err):
    N_list = []
    def f(s,A,N):
        return(A*s**(-N))
    s_fit,dfc_fit,dfc_err_fit = interp(Baryon_name,BaryonMass,dBM,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err)
    A_est,N_est = fit(s_fit,dfc_fit,dfc_err_fit)
    
    res = dfc_fit - f(s_fit,A_est,N_est)
    mean_res = np.mean(res)
    adj_res = res - mean_res
    
    for i in range(n):
        adj_res_b = resample(adj_res)
        popt, pcov = curve_fit(f, s_fit, dfc_fit, sigma = adj_res_b, maxfev = 1000000)
        A,N = popt
        N_list.append(N)
        print(i)
    
    hist = plt.hist(N_list, bins = 'auto')
    binscenters = np.array([0.5 * (hist[1][k] + hist[1][k+1]) for k in range(len(hist[1])-1)])
    def pdf(x,a,b,c):  # gaussian pdf
        return(a*np.exp(-((x-b)**2)/(2*c**2)) )
    param, param_cov = curve_fit(pdf,binscenters,hist[0], maxfev=1000000)
    a,b,c = param
    smooth = np.linspace(min(N_list),max(N_list),10000)
    ordered_list = sorted(N_list)
    #BEG,END = D.BCa('sp',n,0.16,N_est,s_fit,dfc_fit,adj_res,ordered_list)
    
    #plt.plot(smooth,pdf(smooth,*param) )#, label = "mean = %s, SD = %s" 
    #         "\n"
    #         r"68 percent CI: [%s,%s]" % (round(b,3),round(c,3),round(BEG,3),round(END,3)   ))
    plt.xlabel(r'N')
    plt.ylabel(r'Frequency')
    plt.legend()
    plt.show()
    print("The mean is: %s, and the standard deviation is: %s" % (b,c))

# SP_N_hist(3000)    
def SP_N_hist(n):
    
    plt.subplot(2,3,1)
    SP_boot(n,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405PPN,D.ds_dcos_ran_1405PPN,D.ds_dcos_sys_1405PPN)
    plt.title(r'$cos \theta = 0.05$, $\Lambda(1405) < \Sigma^+ \pi^-$ ')
    plt.ylabel(r'Frequency')
    
    plt.subplot(2,3,2)
    SP_boot(n,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405P00,D.ds_dcos_ran_1405P00,D.ds_dcos_sys_1405P00)
    plt.title(r'$cos \theta = 0.05$, $\Lambda(1405) < \Sigma^0 \pi^0$ ')

    plt.subplot(2,3,3)
    SP_boot(n,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405PNP,D.ds_dcos_ran_1405PNP,D.ds_dcos_sys_1405PNP)
    plt.title(r'$cos \theta = 0.05$, $\Lambda(1405) < \Sigma^- \pi^+$ ')
    
    plt.subplot(2,3,4)
    SP_boot(n,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405NPN,D.ds_dcos_ran_1405NPN,D.ds_dcos_sys_1405NPN)
    plt.title(r'$cos \theta = -0.05$') #, $\Lambda(1405) < \Sigma^+ \pi^-$ ')
    plt.ylabel(r'Frequency')
    plt.xlabel(r'$N$')
    
    plt.subplot(2,3,5)
    SP_boot(n,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405N00,D.ds_dcos_ran_1405N00,D.ds_dcos_sys_1405N00)
    plt.title(r'$cos \theta = -0.05$') #, $\Lambda(1405) < \Sigma^0 \pi^0$ ')
    plt.xlabel(r'$N$')
    
    plt.subplot(2,3,6)
    SP_boot(n,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405NNP,D.ds_dcos_ran_1405NNP,D.ds_dcos_sys_1405NNP)
    plt.title(r'$cos \theta = -0.05$')# , $\Lambda(1405) < \Sigma^- \pi^+$ ')
    plt.xlabel(r'$N$')
    plt.show()    
# NP_boot(5000,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405PPN,D.ds_dcos_ran_1405PPN,D.ds_dcos_sys_1405PPN)    
def NP_boot(n,Baryon_name,BaryonMass,dBM,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err):
    N_list = []
    s_fit,dfc_fit,dfc_err_fit = interp(Baryon_name,BaryonMass,dBM,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err)
    
    for i in range(n):
        dfc_err_b = resample(dfc_err_fit)
        
        def f(s,A,N):
            return(A*s**(-N))
        popt, pcov = curve_fit(f, s_fit, dfc_fit, sigma = dfc_err_b, maxfev=10000) 
        A,N = popt
        N_list.append(N)
        print(i)

    hist = plt.hist(N_list, bins = 'auto')
    binscenters = np.array([0.5 * (hist[1][k] + hist[1][k+1]) for k in range(len(hist[1])-1)])
    
    #def gaussian(x,a,b,c):
    #    return(a*np.exp(-((x-b)**2)/(2*c**2)) )
    #param, param_cov = curve_fit(gaussian,binscenters,hist[0], maxfev=10000)
    #print("Height = %s , location = %s , standard deviation = %s " % (param[0],param[1],param[2]))
    #smooth = np.linspace(min(N_list),max(N_list),10000)
    #loc_ord = sorted(N_list)
    #print("The expected 95% confidence interval is [%f,%f]" % (loc_ord[249],loc_ord[9749]))
    #print(loc_ord[249])
    #print(loc_ord[9749])
    #plt.plot(smooth, gaussian(smooth,*param), label = 'mean = %s, SD = %s, 95 percent CI = [%s,%s]' % ( round(param[1],3),round(param[2],3),round(loc_ord[249],3),round(loc_ord[9749],3) )) 
    #plt.plot(location_mean,gaussian(location_mean,*param))
    #plt.xlabel(r'$N$')
    #plt.ylabel(r'Frequency')
    plt.show()

# NP_N_hist(3000)
def NP_N_hist(n):
    
    plt.subplot(2,3,1)
    NP_boot(n,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405PPN,D.ds_dcos_ran_1405PPN,D.ds_dcos_sys_1405PPN)
    plt.title(r'$cos \theta = 0.05$, $\Lambda(1405) < \Sigma^+ \pi^-$ ')
    plt.ylabel(r'Frequency')
    
    plt.subplot(2,3,2)
    SP_boot(n,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405P00,D.ds_dcos_ran_1405P00,D.ds_dcos_sys_1405P00)
    plt.title(r'$cos \theta = 0.05$, $\Lambda(1405) < \Sigma^0 \pi^0$ ')

    plt.subplot(2,3,3)
    SP_boot(n,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405PNP,D.ds_dcos_ran_1405PNP,D.ds_dcos_sys_1405PNP)
    plt.title(r'$cos \theta = 0.05$, $\Lambda(1405) < \Sigma^- \pi^+$ ')
    
    plt.subplot(2,3,4)
    SP_boot(n,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405NPN,D.ds_dcos_ran_1405NPN,D.ds_dcos_sys_1405NPN)
    plt.title(r'$cos \theta = -0.05$') #, $\Lambda(1405) < \Sigma^+ \pi^-$ ')
    plt.ylabel(r'Frequency')
    plt.xlabel(r'$N$')
    
    plt.subplot(2,3,5)
    SP_boot(n,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405N00,D.ds_dcos_ran_1405N00,D.ds_dcos_sys_1405N00)
    plt.title(r'$cos \theta = -0.05$') #, $\Lambda(1405) < \Sigma^0 \pi^0$ ')
    plt.xlabel(r'$N$')
    
    plt.subplot(2,3,6)
    SP_boot(n,D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405NNP,D.ds_dcos_ran_1405NNP,D.ds_dcos_sys_1405NNP)
    plt.title(r'$cos \theta = -0.05$')# , $\Lambda(1405) < \Sigma^- \pi^+$ ')
    plt.xlabel(r'$N$')
    plt.show()

##################################################### FINAL RESULTS #####################################################    
# cos(theta) expansion and 2d fit
_s = np.power(D.W_1405PPN,2)
costheta = np.array([-0.15,-0.05,0.05,0.15])

# for SIGMA+ PI-
mt1,dfc1,dfc_err1 = D.Theta_to_t(_s,D.Lambda1405Mass,D.d1405,costheta[3],D.Pds_dcos_1405PPN,D.Pds_dcos_ran_1405PPN,D.Pds_dcos_sys_1405PPN)
mt2,dfc2,dfc_err2 = D.Theta_to_t(_s,D.Lambda1405Mass,D.d1405,costheta[2],D.ds_dcos_1405PPN,D.ds_dcos_ran_1405PPN,D.ds_dcos_sys_1405PPN)
mt3,dfc3,dfc_err3 = D.Theta_to_t(_s,D.Lambda1405Mass,D.d1405,costheta[1],D.ds_dcos_1405NPN,D.ds_dcos_ran_1405NPN,D.ds_dcos_sys_1405NPN)
mt4,dfc4,dfc_err4 = D.Theta_to_t(_s,D.Lambda1405Mass,D.d1405,costheta[0],D.Nds_dcos_1405NPN,D.Nds_dcos_ran_1405NPN,D.Nds_dcos_sys_1405NPN)

# for SIGMA0 PI0
mt5,dfc5,dfc_err5 = D.Theta_to_t(_s,D.Lambda1405Mass,D.d1405,costheta[3],D.Pds_dcos_1405P00,D.Pds_dcos_ran_1405P00,D.Pds_dcos_sys_1405P00)
mt6,dfc6,dfc_err6 = D.Theta_to_t(_s,D.Lambda1405Mass,D.d1405,costheta[2],D.ds_dcos_1405P00,D.ds_dcos_ran_1405P00,D.ds_dcos_sys_1405P00)
mt7,dfc7,dfc_err7 = D.Theta_to_t(_s,D.Lambda1405Mass,D.d1405,costheta[1],D.ds_dcos_1405N00,D.ds_dcos_ran_1405N00,D.ds_dcos_sys_1405N00)
mt8,dfc8,dfc_err8 = D.Theta_to_t(_s,D.Lambda1405Mass,D.d1405,costheta[0],D.Nds_dcos_1405N00,D.Nds_dcos_ran_1405N00,D.Nds_dcos_sys_1405N00)

# for SIGMA- PI+
mt9,dfc9,dfc_err9 = D.Theta_to_t(_s,D.Lambda1405Mass,D.d1405,costheta[3],D.Pds_dcos_1405PNP,D.Pds_dcos_ran_1405PNP,D.Pds_dcos_sys_1405PNP)
mt10,dfc10,dfc_err10 = D.Theta_to_t(_s,D.Lambda1405Mass,D.d1405,costheta[2],D.ds_dcos_1405PNP,D.ds_dcos_ran_1405PNP,D.ds_dcos_sys_1405PNP)
mt11,dfc11,dfc_err11 = D.Theta_to_t(_s,D.Lambda1405Mass,D.d1405,costheta[1],D.ds_dcos_1405NNP,D.ds_dcos_ran_1405NNP,D.ds_dcos_sys_1405NNP)
mt12,dfc12,dfc_err12 = D.Theta_to_t(_s,D.Lambda1405Mass,D.d1405,costheta[0],D.Nds_dcos_1405NNP,D.Nds_dcos_ran_1405NNP,D.Nds_dcos_sys_1405NNP)

# these are the total (summed) cross sections

# for costheta = 0.15
cosP15 = np.array([0.15,0.15,0.15,0.15,0.15,0.15,0.15])
dfc_P15 = dfc1[2:] + dfc5[2:] + dfc9[2:]
dfc_errP15 = np.sqrt(  dfc_err1[2:]**2 + dfc_err5[2:]**2 + dfc_err9[2:]**2  )

# for costheta = 0.05
cosP05 = np.array([0.05,0.05,0.05,0.05,0.05,0.05,0.05])
dfc_P05 = dfc2[2:] + dfc6[2:] + dfc10[2:]
dfc_errP05 = np.sqrt(  dfc_err2[2:]**2 + dfc_err6[2:]**2 + dfc_err10[2:]**2  )

# for costheta = -0.05
cosN05 = np.array([-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05])
dfc_N05 = dfc3[2:] + dfc7[2:] + dfc11[2:]
dfc_errN05 = np.sqrt(  dfc_err3[2:]**2 + dfc_err7[2:]**2 + dfc_err11[2:]**2  )

# for costheta = -0.15
cosN15 = np.array([-0.15,-0.15,-0.15,-0.15,-0.15,-0.15,-0.15])
dfc_N15 = dfc4[2:] + dfc8[2:] + dfc12[2:]
dfc_errN15 = np.sqrt(  dfc_err4[2:]**2 + dfc_err8[2:]**2 + dfc_err12[2:]**2  )

# 2D fit part

cos1 = np.append(cosP15,cosP05)
cos2 = np.append(cosN05,cosN15)
cosArray = np.append(cos1,cos2)
s_array = np.array([ 4.84, 5.29, 5.76, 6.25, 6.76, 7.29, 7.84,4.84, 5.29, 5.76, 6.25, 6.76, 7.29, 7.84,4.84, 5.29, 5.76, 6.25, 6.76, 7.29, 7.84,4.84, 5.29, 5.76, 6.25, 6.76, 7.29, 7.84])
dfc_1 = np.append(dfc_P15,dfc_P05)
dfc_2 = np.append(dfc_N05,dfc_N15)
dfc_array = np.append(dfc_1,dfc_2) #####

dfc_err_1 = np.append(dfc_errP15,dfc_errP05)
dfc_err_2 = np.append(dfc_errN05,dfc_errN15)
dfc_err_array = np.append(dfc_err_1,dfc_err_2) #####

x1 = cosArray
x2 = s_array

# 2D counting rule

def CCR_2d(X,A,B,N):
    # unpack 1D list into 2D x and y coordinates
    x,y = X 
    # 2D counting rule
    return( (A + B*x)*y**(-N) )
    
# fit
fit_params, cov_mat = curve_fit(CCR_2d,(x1,x2),dfc_array,sigma = dfc_err_array)
A,B,N = fit_params
A_err = np.floor(np.sqrt(cov_mat[0][0])) 
B_err = np.floor(np.sqrt(cov_mat[1][1])) 
N_err = np.round(np.sqrt(cov_mat[2][2]),3)

A = np.floor(A)
B = np.floor(B)
N = np.round(N,3) 

s_lim = np.linspace(_s[2],_s[-1],10000)

plt.errorbar(_s, (dfc1 + dfc5 + dfc9), yerr = np.sqrt(  dfc_err1**2 + dfc_err5**2 + dfc_err9**2  ), fmt = 'go', ecolor = 'r',label = r'$\cos\theta$ = 0.15' )
plt.plot(s_lim, CCR_2d((0.15,s_lim),*fit_params), 'b-')

plt.errorbar(_s, (dfc2 + dfc6 + dfc10), yerr = np.sqrt(  dfc_err2**2 + dfc_err6**2 + dfc_err10**2  ), fmt = 'co', ecolor = 'r',label = r'$\cos\theta$ = 0.05' )
plt.plot(s_lim, CCR_2d((0.05,s_lim),*fit_params), 'b-')

plt.errorbar(_s, (dfc3 + dfc7 + dfc11), yerr = np.sqrt(  dfc_err3**2 + dfc_err7**2 + dfc_err11**2  ), fmt = 'ko', ecolor = 'r',label = r'$\cos\theta$ = -0.05' )
plt.plot(s_lim, CCR_2d((-0.05,s_lim),*fit_params), 'b-')

plt.errorbar(_s, (dfc4 + dfc8 + dfc12), yerr = np.sqrt(  dfc_err4**2 + dfc_err8**2 + dfc_err12**2  ), fmt = 'mo', ecolor = 'r',label = r'$\cos\theta$ = -0.15' )
plt.plot(s_lim, CCR_2d((-0.15,s_lim),*fit_params), 'b-')

plt.text(6.4,0.55,"A = %s $\pm$ %s"
           "\n"
           r"B = %s $\pm$ %s"
           "\n"
           r"N = %s $\pm$ %s" % (A,A_err,B,B_err,N,N_err)) 
plt.title(r'CCR fit to $\frac{d\sigma}{dt} = (A+Bcos\theta)s^{-N}$ for $\Lambda(1405)$')
plt.legend()
plt.show()




