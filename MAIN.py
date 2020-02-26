# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:14:51 2019

@author: elvis
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from sklearn.utils import resample 
from scipy import stats
import data as D

##  Fix font and stuff
#mpl.rc('font',size=22,family='cmr10',weight='normal')
#mpl.rc('text',usetex=True)

##### Bootstrapping + additional statistical analysis for omega 
# bootstrap_unc(10000,50,D.ProtonMass,0,D.W_o,0,D.ds_dcos_o,D.ds_dcos_ran_o,D.ds_dcos_sys_o)
# bootstrap_unc(10000,50,D.SigmaMass,0,D.W_sig,0,D.ds_dcos_sig,D.ds_dcos_ran_sig,D.ds_dcos_sys_sig)
# bootstrap_unc(10000,50,D.LambdaMass,0,D.W_l,0,D.ds_dcos_l,D.ds_dcos_ran_l,D.ds_dcos_sys_l)
def bootstrap_unc(n,num_bins,BaryonMass,dBM,W,costheta,ds_dcos,ds_dcos_ran,ds_dcos_sys): # n stands for number of bootstrap samples
    location_mean = []
    arithmetic_mean = []
    median = []
    p0 = [1,1,1]
    s = np.power(W,2)
    cosThetaMatrix = []
    for i in range(len(s)):
        cosThetaMatrix.append(0)
        cosThetaMatrix[i] = costheta
    cosThetaMatrix = np.array(cosThetaMatrix)
    mt,dfc,dfc_err = D.Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,ds_dcos,ds_dcos_ran,ds_dcos_sys)
    
    def phi(x,A,B,C):
        return((1/A)*np.exp(-(x-B)/A) + C)
    for i in range(n):
        dfc_err_b = resample(dfc_err)   
        bins = np.linspace(min(dfc_err_b),max(dfc_err_b),num_bins)
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        hist = np.histogram(dfc_err_b, bins= bins)    

        popt, pcov = curve_fit(phi,binscenters,hist[0], p0 = p0, maxfev=10000)
        p0 = [popt[0],popt[1],popt[2]]
        location_mean.append(popt[0])
        arithmetic_mean.append(np.mean(dfc_err_b))
        median.append(np.median(dfc_err_b))
        print(i)
        

    print("A = %f, B = %f, C = %f " % (popt[0],popt[1],popt[2]) )
    smooth_x = np.linspace(min(dfc_err_b),max(dfc_err_b),10000)
    plt.subplot(2,1,2)
    plt.hist(dfc_err_b, bins = bins)
    plt.plot(smooth_x,phi(smooth_x,*popt) , label = ("A = %f, B = %f, C = %f " % (popt[0],popt[1],popt[2])) )
    plt.xlabel(r'$\sigma_{d\sigma/dt}$')
    plt.ylabel(r'Frequency')
    plt.legend()
    plt.show()
    
    location_mean = arithmetic_mean
      
    binss = np.linspace(min(location_mean),max(location_mean), 60)
    bin_cent = np.array([0.5 * (binss[place_holder] + binss[place_holder+1]) for place_holder in range(len(binss)-1)])
    histo_info = np.histogram(location_mean,bins= binss)
    #return("The length of the bin center array is %s and the length of the frequency array is %s " % (len(histo_info[1]), len(histo_info[0])))
    def gaussian(x,a,b,c):
        return( a*np.exp(-((x-b)**2)/(2*c**2) ) )
    param, param_cov = curve_fit(gaussian,bin_cent,histo_info[0], p0 = [500,0.0565,0.0482], maxfev=10000)
    print("Height = %s , location = %s , standard deviation = %s " % (param[0],param[1],param[2]))
    print("The length of the location_mean array is %s " % len(location_mean))
    smooth = np.linspace(min(location_mean),max(location_mean),10000)
    loc_ord = sorted(location_mean)
    #print("The expected 95% confidence interval is [%f,%f]" % (loc_ord[249],loc_ord[9749]))
    print(loc_ord[249])
    print(loc_ord[9749])
    plt.subplot(2,1,1)
    plt.hist(location_mean,bins= binss)
    plt.plot(smooth, gaussian(smooth,*param), label =  "Mean = %s, SD = %s, 95 percent CI = [%s,%s] " % (round(param[1],3),round(param[2],3),round(loc_ord[249],3) ,round(loc_ord[9749],3))) 
    plt.xlabel(r'Mean $\sigma_{d\sigma/dt}$')
    plt.ylabel(r'Frequency')
    plt.legend()
    plt.title(r'$\sigma_{d\sigma/dt}$ Histograms fitted to $ae^{-({x-b})^2/{2c^2}}$ and $(1/A)e^{-({x-B})/{A}} + C$ for 10,000 samples')
    plt.show()
#####
#s_fit_range = 5.4
# bootstrap_N(10000,D.ProtonMass,0,D.W_o,0,5.4,D.ds_dcos_o,D.ds_dcos_ran_o,D.ds_dcos_sys_o)
# bootstrap_N(10000,SigmaMass,0,D.W_sig,0,5.45,D.ds_dcos_sig,D.ds_dcos_ran_sig,D.ds_dcos_sys_sig)
# bootstrap_N(10000,D.LambdaMass,0,D.W_l,0,5.00,D.ds_dcos_l,D.ds_dcos_ran_l,D.ds_dcos_sys_l)
# this function resamples the original uncertainty dataset non-parametrically and 
# then fits the new data N times to obtain the parameter distributions
def bootstrap_N(n,BaryonMass,dBM,W,costheta,s_fit_range,ds_dcos,ds_dcos_ran,ds_dcos_sys):
    s = np.power(W,2)
    cosThetaMatrix = []
    for i in range(len(s)):
        cosThetaMatrix.append(0)
        cosThetaMatrix[i] = costheta
    cosThetaMatrix = np.array(cosThetaMatrix)
    mt,dfc,dfc_err = D.Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,ds_dcos,ds_dcos_ran,ds_dcos_sys)
    N_list = []
    A_list = []
    j = 0    
    while s[j] < s_fit_range: 
        j += 1
    for i in range(n):
        dfc_err_b = resample(dfc_err)
        mt_fit = mt[j:]    
        s_fit = s[j:]
        dfc_fit = dfc[j:]
        dfc_err_fit = dfc_err_b[j:]
        print(i)

        def f(s,A,N):
            return(A*s**(-N))
    
        popt, pcov = curve_fit(f, s_fit, dfc_fit, sigma = dfc_err_fit, maxfev=10000) 
        A,N = popt
        N_list.append(N)
        A_list.append(A)
        B_err = np.sqrt(pcov[1][1])  
    print("N = %s" % N)
    print("N_err = %s" % B_err )
    bins = np.linspace(min(N_list),max(N_list), 150)
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    #fig1 = plt.figure()
    hist = plt.hist(N_list, bins = bins)
    #binscenters = np.array([0.5 * (hist[1][k] + hist[1][k+1]) for k in range(len(hist[1])-1)])
    
    def gaussian(x,a,b,c):
        return(a*np.exp(-((x-b)**2)/(2*c**2)) )
    param, param_cov = curve_fit(gaussian,binscenters,hist[0], maxfev=10000)
    print("Height = %s , location = %s , standard deviation = %s " % (param[0],param[1],param[2]))
    smooth = np.linspace(min(N_list),max(N_list),10000)
    loc_ord = sorted(N_list)
    #print("The expected 95% confidence interval is [%f,%f]" % (loc_ord[249],loc_ord[9749]))
    print(loc_ord[249])
    print(loc_ord[9749])
    plt.plot(smooth, gaussian(smooth,*param), label = 'mean = %s, SD = %s, 95 percent CI = [%s,%s]' % ( round(param[1],3),round(param[2],3),round(loc_ord[249],3),round(loc_ord[9749],3) )) 
    #plt.plot(location_mean,gaussian(location_mean,*param))
    plt.xlabel(r'$N$')
    plt.ylabel(r'Frequency')
    plt.legend()
    plt.title(r'N Histogram $[ As^{-N} ]$ for 10,000 samples fitted from %s $GeV^2$ ' % s_fit_range)
    '''
    fig2 = plt.figure()
    plt.hist2d(N_list, A_list, bins= 'auto')
    plt.xlabel('N')
    plt.ylabel('A')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.show()
'''
# bootstrap_N2(10000,D.LambdaMass,0,D.W_l,0,4.56,D.ds_dcos_l,D.ds_dcos_ran_l,D.ds_dcos_sys_l)
# bootstrap_N2(10000,SigmaMass,0,D.W_sig,0,5.45,D.ds_dcos_sig,D.ds_dcos_ran_sig,D.ds_dcos_sys_sig)
# this function models the uncertainty parametrically from a normal distribution
# with mean and standard deviation equal to that of the original uncertainty dataset
def bootstrap_N2(n,BaryonMass,dBM,W,costheta,s_fit_range,ds_dcos,ds_dcos_ran,ds_dcos_sys):
    s = np.power(W,2)
    cosThetaMatrix = []
    for i in range(len(s)):
        cosThetaMatrix.append(0)
        cosThetaMatrix[i] = costheta
    cosThetaMatrix = np.array(cosThetaMatrix)
    mt,dfc,dfc_err = D.Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,ds_dcos,ds_dcos_ran,ds_dcos_sys)
    
    s = np.delete(s,0)
    mt = np.delete(mt,0)
    dfc = np.delete(dfc,0)
    dfc_err = np.delete(dfc_err,0)
    neg_dfc_err = -dfc_err
    tot_dfc_err = np.append(dfc_err, neg_dfc_err)
    
    #neg_dfc_err = -dfc_err
    #tot_dfc_err = np.append(dfc_err, neg_dfc_err)
    N_list = []
    A_list = []
    
    #bins = np.linspace(min(tot_dfc_err),max(tot_dfc_err),30)
    #binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    fig1 = plt.figure()
    hist = plt.hist(tot_dfc_err, bins = 'auto')

    binscenters = np.array([0.5 * (hist[1][k] + hist[1][k+1]) for k in range(len(hist[1])-1)])
    #print(len(binscenters))
    
    def gaussian(x,a,b,c):
        return(a*np.exp(-((x-b)**2)/(2*c**2)) )
    param, param_cov = curve_fit(gaussian,binscenters,hist[0], maxfev=10000)
    mean = round(param[1],5)
    #SD = 3*round(abs(param[2]),5)
    SD = 1000
    print("Height = %s , location = %s , standard deviation = %s " % (param[0],param[1],param[2]))
    smooth = np.linspace(min(tot_dfc_err),max(tot_dfc_err),10000)
    plt.plot(smooth,gaussian(smooth,*param))
    plt.show()
    j = 0    
    while s[j] < s_fit_range: 
        j += 1
    for i in range(n):
        normal_boot_err = np.random.normal(mean,SD,len(dfc_err))
        s_fit = s[j:]
        dfc_fit = dfc[j:]
        dfc_err_fit = normal_boot_err[j:]
        def f(s,A,N):
            return(A*s**(-N))
        popt, pcov = curve_fit(f, s_fit, dfc_fit, sigma = dfc_err_fit, maxfev=1000000) 
        A,N = popt
        N_list.append(N)
        A_list.append(A)
        print(i)
    print(mean)
    print(SD)
    fig2 = plt.figure()
    hist = plt.hist(N_list, bins = 'auto')
    
    binscenters = np.array([0.5 * (hist[1][k] + hist[1][k+1]) for k in range(len(hist[1])-1)])
    print("Hello")
    param, param_cov = curve_fit(gaussian,binscenters,hist[0], maxfev=10000)
    print("Height = %s , location = %s , standard deviation = %s " % (param[0],param[1],param[2]))
    smooth = np.linspace(min(N_list),max(N_list),10000)
    loc_ord = sorted(N_list)
    #print("The expected 95% confidence interval is [%f,%f]" % (loc_ord[249],loc_ord[9749]))
    print(loc_ord[249])
    print(loc_ord[9749])
    plt.plot(smooth, gaussian(smooth,*param), label = ' ')  #label = 'mean = %s, SD = %s, 95 percent CI = [%s,%s]' % ( round(param[1],3),round(param[2],3),round(loc_ord[249],3),round(loc_ord[9749],3) )) 
    plt.xlabel(r'$N$')
    plt.ylabel(r'Frequency')
    plt.legend()
    plt.title(r'N Histogram $[ As^{-N} ]$ for 10,000 samples fitted from %s $GeV^2$ ' % s_fit_range)
    plt.show()

# bootstrap_N3(10000,D.LambdaMass,0,D.W_l,0,5.00,D.ds_dcos_l,D.ds_dcos_ran_l,D.ds_dcos_sys_l)
# bootstrap_N3(10000,D.SigmaMass,0,D.W_sig,0,5.45,D.ds_dcos_sig,D.ds_dcos_ran_sig,D.ds_dcos_sys_sig) 
# this function will use semi-parametric bootstrapping (look up for more details)
def bootstrap_N3(n,BaryonMass,dBM,W,costheta,s_fit_range,ds_dcos,ds_dcos_ran,ds_dcos_sys):
    s = np.power(W,2)
    cosThetaMatrix = []
    for i in range(len(s)):
        cosThetaMatrix.append(0)
        cosThetaMatrix[i] = costheta
    cosThetaMatrix = np.array(cosThetaMatrix)
    mt,dfc,dfc_err = D.Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,ds_dcos,ds_dcos_ran,ds_dcos_sys)
    N_list = []
    
    j = 0
    while s[j] < s_fit_range:
        j += 1
    mt_fit = mt[j:]    
    s_fit = s[j:]
    dfc_fit = dfc[j:]
    dfc_err_fit = dfc_err[j:]
    def f(s,A,N):
        return(A*s**(-N))
    popt, pcov = curve_fit(f, s_fit, dfc_fit, sigma = dfc_err_fit) 
    A_est,N_est = popt
    res = dfc_fit - f(s_fit,*popt)
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
    BEG,END = D.BCa('sp',n,0.16,N_est,s_fit,dfc_fit,adj_res,ordered_list)
    
    plt.plot(smooth,pdf(smooth,*param), label = "mean = %s, SD = %s" 
             "\n"
             r"68 percent CI: [%s,%s]" % (round(b,3),round(c,3),round(BEG,3),round(END,3)   ))
    plt.xlabel(r'N')
    plt.ylabel(r'Frequency')
    plt.legend()
    plt.show()
    print("The mean is: %s, and the standard deviation is: %s" % (b,c))
    
    
# Best_s_range(D.Lambda,D.LambdaMass,0,0,D.W_l,D.ds_dcos_l,D.ds_dcos_ran_l,D.ds_dcos_sys_l)
# Best_s_range(D.Sigma,D.SigmaMass,0,0,D.W_sig,D.ds_dcos_sig,D.ds_dcos_ran_sig,D.ds_dcos_sys_sig)
# Best_s_range(D.Lambda1405PN,D.Lambda1405Mass,D.d1405,0,D.W_1405PPN,D.ds_dcos_1405PPN,D.ds_dcos_ran_1405PPN,D.ds_dcos_sys_1405PPN)
def Best_s_range(Baryon_name,BaryonMass,dBM,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err):
    # BaryonMass is to indicate the particle mass
    # s_fit_range indicates that you only want to consider  s_fit_range <= s <= max {s} for the fit
    # mt = momentum_transfer(t), dfc = differential cross section (dsig/dt), dfc_err is the error in dfc    
    s = np.power(W,2)
    cosThetaMatrix = []
    for i in range(len(s)):
        cosThetaMatrix.append(0)
        cosThetaMatrix[i] = costheta
    cosThetaMatrix = np.array(cosThetaMatrix)
    mt,dfc,dfc_err = D.Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err)
    
    
    red_chisq = []
    for i in range(len(s)-1): # the last 4 data points are giving trouble for some reason
        def f(s,A,N):
            return(A*s**(-N))
        popt, pcov = curve_fit(f, s[i:], dfc[i:], sigma = dfc_err[i:], maxfev = 1000000)
        A,N = popt
        res = dfc[i:] - f(s[i:],A,N)
        #return(res)
        chisq = sum((res/dfc_err[i:])**2)
        red_chisq.append(chisq/(len(dfc[i:])-len(popt)))

    ideal_chisq = []
    for i in range(len(red_chisq)):
        ideal_chisq.append(1)
             
    plt.plot(s[:-1], red_chisq, 'go')#,label = r'Optimal $s \approx %s$' % round(5.1,2) + ' \n ' + r'$\frac{d\sigma}{dt} = A*s^{2-n}$' )
    plt.plot(s[:-1],ideal_chisq, 'b-')#,label = '')
    plt.xlabel(r'$\ s (GeV^2)$')
    plt.ylabel(r'$\chi^2/\nu $')
    #plt.legend()
    #plt.title(r'$\gamma + P \rightarrow K^+ + \%s$' % Baryon_name + " Statistical Analysis  " + r'for $\cos \theta _{c.m.} = %s $' % costheta )
    plt.show()


    # shifted the graph down by one to find roots, i.e. points that would normally be at red_chisq = 1    
    red_chisq2 = np.array(red_chisq) - 1
        
    # Method (Bisection)
    delta = 0.1
    a = 0
    b = len(red_chisq2)
    q = []
    q.append((a+b)/2)
    COUNTER = 0
    while abs(red_chisq2[int(q[COUNTER])]) > delta:
        if red_chisq2[int(q[COUNTER])]*red_chisq2[int(a)] > 0:
            a = q[COUNTER]
        else:
            b = q[COUNTER]
        q.append((a+b)/2)
        #print(COUNTER)
        COUNTER += 1
        if COUNTER > 100:
            break
    if s[int(q[-2])] > s[int(q[-1])]:
        s1 = s[int(q[-1])]
        s2 = s[int(q[-2])]
    else:
        s1 = s[int(q[-2])]
        s2 = s[int(q[-1])]
    s_bi = s[int(q[-1])]
    #print(len(s))
    #print(len(red_chisq))
    plt.plot(s[:-1], red_chisq, 'go' , label = 'green = Lambda')#,label = r'Optimal $s \approx %s$' % round(s_bi,2) + ' \n ' + r'$\frac{d\sigma}{dt} = A*s^{2-n}$' )
    plt.plot(s[:-1],ideal_chisq, 'b-')#,label = '')
    plt.xlabel(r'$\ s (GeV^2)$')
    plt.ylabel(r'$\chi^2/\nu $')
    plt.legend()
    #plt.title(r'$\gamma + P \rightarrow K^+ + \%s$' % Baryon_name + " Statistical Analysis  " + r'for $\cos \theta _{c.m.} = %s $' % costheta )
    '''
    plt.annotate("", xy=(4.3, 1.17), xytext=(5.29,14.48), arrowprops=dict(arrowstyle="->"))
    a = plt.axes([0.45, 0.6, .2, .2])
    plt.plot(s[8:],red_chisq[:], 'go')
    plt.plot(s[8:],ideal_chisq[:],'b-')
    #plt.title('Impulse response')
    plt.xlim(3.2, 5.7)
    plt.ylim(0.5,1.5)
    plt.xticks([])
    plt.yticks([])
    '''
    plt.show()
    print("Bisection: Best range %s < s < %s " % (s1,s2))
    print("Bisection: Reduced Chi Square = %s" % (red_chisq2[int(q[-1])]+1))
    return(s_bi)

#(Sigma) counting_rule_fit(Sigma,SigmaMass,0,5.45,0,W_sig,ds_dcos_sig,ds_dcos_ran_sig,ds_dcos_sys_sig)
#(Lambda) counting_rule_fit(Lambda,LambdaMass,0,5.0,0,W_l,ds_dcos_l,ds_dcos_ran_l,ds_dcos_sys_l)
def counting_rule_fit(Baryon_name,BaryonMass,dBM,s_fit_range,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err):
    # BaryonMass is to indicate the particle mass
    # s_fit_range indicates that you only want to consider  s_fit_range <= s <= max {s} for the fit
    # mt = momentum_transfer(t), dfc = differential cross section (dsig/dt), dfc_err is the error in dfc    
    s = np.power(W,2)
    cosThetaMatrix = []
    for i in range(len(s)):
        cosThetaMatrix.append(0)
        cosThetaMatrix[i] = costheta
    cosThetaMatrix = np.array(cosThetaMatrix)
    mt,dfc,dfc_err = Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err)
    
    #s_fit_range = Best_s_range(Baryon_name,BaryonMass,dBM,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err)

    j = 0    
    while s[j] < s_fit_range: # the way the cutoff is picked here is different than in the S_fit function 
                              # and could be leading to the discrepancy
        j += 1
    
    mt_fit = mt[j:]    
    s_fit = s[j:]
    dfc_fit = dfc[j:]
    dfc_err_fit = dfc_err[j:]

    def f(s,A,N):
        return(A*s**(-N))
    
    popt, pcov = curve_fit(f, s_fit, dfc_fit, sigma = dfc_err_fit) #, absolute_sigma = True, bounds = ([-np.inf,7.999],[np.inf,8.001])) 
    A,N = popt
    A_err = np.sqrt(pcov[0][0])
    B_err = np.sqrt(pcov[1][1])  
    residual = dfc_fit - f(s_fit,A,N) 
    chisq = sum((residual/dfc_err_fit)**2)
    red_chisq = chisq/(len(s_fit)-len(popt))
    print("N = %s" % N)
    print("N_err = %s" % B_err )
    # this array is just to give the fitting function a smooth appearance when graphed
    x = np.linspace(s_fit[0], s_fit[-1],10000)
    
    plt.subplot(2,1,2)
    plt.errorbar(s, dfc, yerr = dfc_err, fmt = 'go', ecolor = 'r',label = 'Experimental' )
    #plt.plot(-mt, dfc)
    plt.plot(x, f(x,A,N), 'b-',label = 'fit: A = %s ' % int(A) + r'$\pm $ ' + '%s  ' % int(A_err) + '; N = %s ' % round(N,1) + r'$\pm$ ' + '%s ' % round(B_err,1) + '; ' + r'$\chi^2/\nu$ = %s' % round(red_chisq,2) + ' \n ' + r'$ %s < -t \: (GeV^2) < %s $' % (round(-mt_fit[0],2),round(-mt_fit[-1],2)) )
    plt.xlabel(r'$\ s (GeV^2)$')
    plt.ylabel(r'$\frac{d\sigma}{dt} (\mu b *GeV^{-2})$')
    plt.legend()
    plt.show()
    
    # this is for the linear fit part
    
    #### for the constant part ####
    scale_factor = np.power(s, N)
    constituent_constant = scale_factor*dfc

    # error in scale factor
    dSF = scale_factor*np.log(s)*B_err
    # error in constituent constant
    first_part = np.power(dfc*dSF,2)
    second_part = np.power(scale_factor*dfc_err,2)
    #dCC = np.sqrt(first_part+second_part)
    dCC = dfc*dSF + scale_factor*dfc_err
    #print(dCC)
      
    def g(s,C):
        return (C)
    
    constituent_constant_fit = constituent_constant[j:]
    dCC_fit = dCC[j:]
    
    popt2, pcov2 = curve_fit(g, s_fit, constituent_constant_fit, sigma = dCC_fit, absolute_sigma = True)
    C = popt2[0]
    # error in C
    C_err = np.sqrt(pcov2[0][0])
    residual2 = constituent_constant_fit - g(s_fit,C)
    chisq2 = sum((residual2/dCC_fit)**2)
    red_chisq2 = chisq2/(len(s_fit)-len(popt2))
    #print(chisq2)
    
    # C & C_err vectors
    C_vec = []
    C_err_vec = []
    for i in range(len(s_fit)):
        C_vec.append(C)
        C_err_vec.append(C_err)

    plt.subplot(2,1,1)
    plt.errorbar(s, constituent_constant, yerr = dCC, fmt = 'go', ecolor = 'r',label = 'Experimental')
    plt.plot(s_fit, C_vec, 'b-',label = 'fit: A = %s ' % int(C) + r'$ \pm $' + ' %s' % int(C_err) + '; ' + r'$ \chi^2/\nu $ = %s ' % round(red_chisq2,2))
    plt.ylabel(r'$s^{-N} \frac{d\sigma}{dt} (\mu b *GeV^{2n-2})$')
    plt.legend()
    plt.title(r'$\gamma + P \rightarrow K^+ + \%s$' % Baryon_name + " Constituent Counting Fit  " + ' (for ' + r'$\cos \theta _{c.m.} = %s$' % costheta + ' and ' + r'$ s > %s\ GeV^2)$' % round(float(s_fit_range),2))
    plt.show()
  

# Best_s_range2(0,Lambda,LambdaMass,0,0,W_l,ds_dcos_l,ds_dcos_ran_l,ds_dcos_sys_l)
# Best_s_range2(0,Sigma,SigmaMass,0,0,W_sig,ds_dcos_sig,ds_dcos_ran_sig,ds_dcos_sys_sig)
# Best_s_range2(0,Lambda1405PN,Lambda1405Mass,d1405,0.05,W_1405PPN,ds_dcos_1405PPN,ds_dcos_ran_1405PPN,ds_dcos_sys_1405PPN)
def Best_s_range2(s_fit_beg,Baryon_name,BaryonMass,dBM,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err):
    #if gamma > 0, the fit will start at threshold energy, otherwise the fit will be of the form s_fit -> s_max
    # BaryonMass is to indicate the particle mass
    # s_fit_range indicates that you only want to consider  s_fit_range <= s <= max {s} for the fit
    # mt = momentum_transfer(t), dfc = differential cross section (dsig/dt), dfc_err is the error in dfc    
    s = np.power(W,2)
    cosThetaMatrix = []
    for i in range(len(s)):
        cosThetaMatrix.append(0)
        cosThetaMatrix[i] = costheta
    cosThetaMatrix = np.array(cosThetaMatrix)
    mt,dfc,dfc_err = Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err)

    y = np.power(s,6)*dfc
    y_err = np.power(s,6)*dfc_err    
    #s_fit_range = np.linspace(s[0],s[-1],len(s))
    red_chisq = []
    
    # last few points in s_fit_range were giving some trouble for the fit 
    g = 0
    if s_fit_beg < 0:
        s_beg = max(y)
        #return(s_beg)
        while y[g] < s_beg :
            g += 1

    #return(g,s_fit_beg)          # test line to determine start point of stat analysis
    i = 0  
    k = g
    while True:   
         if s_fit_beg == 0:   
             if i >= (len(s)-6):
                 break
             s_fit = s[i:]
             dfc_fit = dfc[i:]
             dfc_err_fit = dfc_err[i:]
             i += 1
         else:
             if k >= (len(s)-6):
                 break
             k += 1
             s_fit = s[g:k]
             dfc_fit = dfc[g:k]
             dfc_err_fit = dfc_err[g:k]
             #print(k)
            
         def f(s,A,N):
             return(A*s**(-N))
        
         y_fit = np.power(s_fit,6)*dfc_fit
         y_fit_err = np.power(s_fit,6)*dfc_err_fit
    
         popt, pcov = curve_fit(f, s_fit, y_fit, sigma = y_fit_err)#, absolute_sigma = True, bounds = ([-np.inf,1.999],[np.inf,2.001]))
         A,N = popt
         A_err = np.sqrt(pcov[0][0])
         B_err = np.sqrt(pcov[1][1])  
         residual = y_fit - f(s_fit,A,N) 
         #print(residual)
         chisq = sum((residual/y_fit_err)**2)
         red_chisq.append(chisq/(len(y_fit)-len(popt)))
    #return(red_chisq)
    
    ideal_chisq = []
    for i in range(len(red_chisq)):
        ideal_chisq.append(1)
        
    plt.plot(s[g:-6], red_chisq, 'go')#,label = r'Optimal $s \approx %s$' % round(5.10,2) + ' \n ' + r'$s^6\frac{d\sigma}{dt} = A*s^{-N_{\gamma}}$')
    plt.plot(s[g:-6],ideal_chisq, 'b-',label = '')
    plt.xlabel(r'$\ s (GeV^2)$')
    plt.ylabel(r'$\chi^2/\nu $')
    plt.legend()
    plt.title(r'$\gamma + P \rightarrow K^+ + \%s$' % Baryon_name + " Statistical Analysis  " + r'for $\cos \theta _{c.m.} = %s $' % costheta )
    plt.show()

    # shifted the graph down by one to find roots, i.e. points that would normally be at red_chisq = 1    
    red_chisq2 = np.array(red_chisq) - 1
    
    # Method (Bisection)
    delta = 0.001
    a = 0
    b = len(red_chisq2)
    q = []
    q.append((a+b)/2)
    COUNTER = 0
    while abs(red_chisq2[int(q[COUNTER])]) > delta:
        if red_chisq2[int(q[COUNTER])]*red_chisq2[int(a)] > 0:
            a = q[COUNTER]
        else:
            b = q[COUNTER]
        q.append((a+b)/2)
        #print(COUNTER)
        COUNTER += 1
        if COUNTER > 100:
            break
    if s[int(q[-2])] > s[int(q[-1])]:
        s1 = s[int(q[-1])]
        s2 = s[int(q[-2])]
    else:
        s1 = s[int(q[-2])]
        s2 = s[int(q[-1])]
    s_bi = s[int(q[-1])]
    # best estimate for s rang
    plt.plot(s[g:-6], red_chisq, 'go',label = r'Optimal $s \approx %s$' % round(5.10,2) + ' \n ' + r'$s^6\frac{d\sigma}{dt} = A*s^{-N_{\gamma}}$')
    plt.plot(s[g:-6],ideal_chisq, 'b-',label = '')
    plt.xlabel(r'$\ s (GeV^2)$')
    plt.ylabel(r'$\chi^2/\nu $')
    plt.legend()
    plt.title(r'$\gamma + P \rightarrow K^+ + \%s$' % Baryon_name + " Statistical Analysis  " + r'for $\cos \theta _{c.m.} = %s $' % costheta )
    plt.show()
    print("Bisection: Best range %s < s < %s " % (s1,s2))
    print("Bisection: Reduced Chi Square = %s" % (red_chisq2[int(q[-1])]+1))
    return(s_bi)  

#(Lambda) gamma_ext(Lambda,LambdaMass,0,0,0,0,W_l,ds_dcos_l,ds_dcos_ran_l,ds_dcos_sys_l)
#(Sigma) gamma_ext(Sigma,SigmaMass,0,0,0,0,W_sig,ds_dcos_sig,ds_dcos_ran_sig,ds_dcos_sys_sig)    
def gamma_ext(Baryon_name,BaryonMass,dBM,s_fit_beg,s_fit_end,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err):
    s = np.power(W,2)
    cosThetaMatrix = []
    for i in range(len(s)):
        cosThetaMatrix.append(0)
        cosThetaMatrix[i] = costheta
    cosThetaMatrix = np.array(cosThetaMatrix)
    mt,dfc,dfc_err = Theta_to_t(s,BaryonMass,dBM,cosThetaMatrix,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err)
    '''
    y = np.power(s,6)*dfc
    y_err = np.power(s,6)*dfc_err
    #s_fit_end = Best_s_range2(s_fit_beg,Baryon_name,BaryonMass,dBM,costheta,W,ds_dcos,ds_dcos_ran_err,ds_dcos_sys_err)
    '''
    y = dfc
    y_err = dfc_err
    g = 0
    if s_fit_beg < 0:
        s_beg = max(y)
        while y[g] < s_beg :
            g += 1
    
    k = g
    while s[k] < s_fit_beg:
        k += 1
    j = 0    
    while s[j] < s_fit_end:
        j += 1
        
    if s_fit_beg == 0:
        mt_fit = mt[j:]
        s_fit = s[j:]
        dfc_fit = dfc[j:]
        dfc_err_fit = dfc_err[j:]
    else:
        mt_fit = mt[k:j]
        s_fit = s[k:j]
        dfc_fit = dfc[k:j]
        dfc_err_fit = dfc_err[k:j]
        
    def f(s,A,N):
        return(A*s**(-N))
        
    '''
    y_fit = np.power(s_fit,6)*dfc_fit
    y_fit_err = np.power(s_fit,6)*dfc_err_fit
    '''
    y_fit = dfc_fit
    y_fit_err = dfc_err_fit
    
    popt, pcov = curve_fit(f, s_fit, y_fit, sigma = y_fit_err)
    A,N = popt
    A_err = np.sqrt(pcov[0][0])
    B_err = np.sqrt(pcov[1][1])  
    residual = y_fit - f(s_fit,A,N) 
    chisq = sum((residual/y_fit_err)**2)
    red_chisq = chisq/(len(y_fit)-len(popt))
    print("N = %s" % N)
    print("N_err = %s" % B_err )
    # this array is just to give the fitting function a smooth appearance when graphed
    x = np.linspace(s_fit[0],s_fit[-1],10000)
    
    plt.subplot(2,1,2)
    plt.errorbar(s, y, yerr = y_err, fmt = 'go', ecolor = 'r',label = 'Experimental')
    plt.plot(x, f(x,A,N), 'b-',label = 'fit: A = %s ' % int(A) + r'$\pm $ ' + '%s  ' % int(A_err) + r'; $N_{\gamma}$ = %s ' % round(N,1) + r'$\pm$ ' + '%s ' % round(B_err,1) + '; ' + r'$\chi^2/\nu$ = %s' % round(red_chisq,2) + ' \n ' + r'$ %s < -t \: (GeV^2) < %s $' % (round(-mt_fit[0],2),round(-mt_fit[-1],2)))
    plt.xlabel(r'$\ s (GeV^2)$')
    plt.ylabel(r'$s^6 \frac{d\sigma}{dt} (\mu b *GeV^{2n-2})$')
    plt.legend()
    #plt.title(r'$\gamma + P \rightarrow K^+ + \%s$' % Baryon_name + "  Experimental fit to  " +  r'$  s^6 \frac{d\sigma}{dt} = A*s^{-N_{\gamma}} $' + ' (for ' + r'$\cos \theta _{c.m.} = %s$' % costheta +')' )
    plt.show()
    
    # this is for the linear fit part
    
    #### for the constant part ####
    scale_factor = np.power(s, N)
    constituent_constant = scale_factor*dfc

    # error in scale factor
    dSF = scale_factor*np.log(s)*B_err
    # error in constituent constant
    first_part = np.power(dfc*dSF,2)
    second_part = np.power(scale_factor*dfc_err,2)
    #dCC = np.sqrt(first_part+second_part)
    dCC = dfc*dSF + scale_factor*dfc_err
    #print(dCC)
      
    def g(s,C):
        return (C)
    
    constituent_constant_fit = constituent_constant[j:]
    dCC_fit = dCC[j:]
    
    popt2, pcov2 = curve_fit(g, s_fit, constituent_constant_fit, sigma = dCC_fit)#, absolute_sigma = True)
    C = popt2[0]
    # error in C
    C_err = np.sqrt(pcov2[0][0])
    residual2 = constituent_constant_fit - g(s_fit,C)
    chisq2 = sum((residual2/dCC_fit)**2)
    red_chisq2 = chisq2/(len(s_fit)-len(popt2))
    #print(chisq2)
    
    # C & C_err vectors
    C_vec = []
    C_err_vec = []
    for i in range(len(s_fit)):
        C_vec.append(C-4000)
        C_err_vec.append(C_err)
        

    plt.subplot(2,1,1)
    plt.errorbar(s, constituent_constant, yerr = dCC, fmt = 'go', ecolor = 'r',label = 'Experimental')
    plt.plot(s_fit, C_vec, 'b-',label = 'fit: A = %s ' % int(C-4000) + r'$ \pm $' + ' %s' % int(C_err) + '; ' + r'$ \chi^2/\nu $ = %s ' % round(red_chisq2,2))
    plt.ylabel(r'$s^{-N} \frac{d\sigma}{dt} (\mu b *GeV^{2n-2})$')
    plt.legend()
    plt.title(r'$\gamma + P \rightarrow K^+ + \%s$' % Baryon_name + " Constituent Counting Fit  " + ' (for ' + r'$\cos \theta _{c.m.} = %s$' % costheta + ' and ' + r' 3.60 $GeV^2$ $\leq$ s $\leq$ 5.00 $GeV^2$)') # % round(float(s_fit_range),2))
    plt.show()
    