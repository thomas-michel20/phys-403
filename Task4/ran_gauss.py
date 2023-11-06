"""
Generates a normal distribution using two different methods 
First wrote by Gabriele Sclauzero (EPFL, Lausanne), Sept. 2010 in FORTRAN
Adapted to python 3 by Arnaud Lorin (EPFL, Lausanne), Nov 2021
Input
-Method of choice (1=BM,2=CLT)
    -if CLT Number of number in one sum
-Number of random numbers to generate
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import ran


def main(method=None, nsum=None, ntry=None):
    
    # Input paramters
    # Method (1=Box-Muller;2=C.L. theor.)
    #method=int(input("method (1=Box-Muller;2=C.L. theor.): "))
    # Only for method == 2
    if method == 2:
        nsum=nsum

    # Numner of try
    ntry=ntry
    
    # Binning parameters
    nbins=200 
    rmin=-4 # Min range for binning
    rmax=-rmin # Max range for binning

    if method == 2:
        # Number of R.N. to sum
        sgm=(1./(12*nsum))**0.5
    
    global invbinw
    global cnt

    # Random seed
    x=datetime.datetime.now()
    seed=int(x.microsecond/1000)+1000*(x.second+60*(x.minute+60*x.hour))
    #seed =  imsec              +1000*(isec    +60*(imin    +60*ihr   ))
    seed = seed | 1 # to ensure that seed is an odd number
    
    # Fixed seed (for debugging purposes)
    #seed=123456789
    
    cnt=np.zeros(nbins)
    invbinw = nbins / (rmax - rmin)

    rave = 0 # Average
    r2ave = 0 # Standard deviation 
    nran = 0 # Number of numbers selected (only for BM)
    
    for i in range(ntry):
    
        if method == 1:
            # Box-Muller
    
            res,seed=ran.ran2(seed) 
            v1 = 2*res - 1
            res,seed=ran.ran2(seed) 
            v2 = 2*res - 1
            r2 = v1**2 + v2**2
            
            if r2 >= 1: continue
            
            nran = nran + 1
            fac = ( -2*np.log(r2)/r2 )**0.5
            res = v2*fac
    
        elif method == 2:
            # Central limit theorem
            res = 0
            for j in range(nsum):
                res1,seed=ran.ran2(seed) 
                res += res1
    
            nran += 1
            res = (res/nsum - 0.5) / sgm
        else:
          raise Exception ("unknown method")
    
        binning(res,rmin,rmax)
    
        rave = rave + res
        r2ave = r2ave + res*res
    
    rave = rave / nran
    r2ave = r2ave / nran
    
    print('average  = {:14.9f}'.format(rave/nran))
    print('std. dev.= {:14.9f}'.format(np.sqrt(r2ave - rave*rave)))
    
    # write histogram to file
    if method == 1:
       methc = 'BM'
    elif method == 2:
       methc = 'CL'+str(nsum)
    filename = 'histo-gauss{}_{}'.format(methc,ntry)
    
    f=open(filename,'w')
    for i in range(nbins):
        f.write('{:14.6f}{:14.6f}\n'.format((i+0.5)/invbinw + rmin,cnt[i]/nran*invbinw ))
    f.close()
    
    if method == 1:
       name='BM '+str(ntry)+' try'
    elif method == 2:
       name='CL '+str(nsum)+' sum '+str(ntry)+' try'
    x=np.zeros(nbins)
    gauss=np.zeros(nbins)
    for i in range(nbins):
        x[i]=(i+0.5)/invbinw + rmin
    gauss=np.exp(-np.square(x)/2)/(np.sqrt(2*np.pi))
    plt.plot(x,cnt/nran*invbinw,label=name)
    plt.plot(x,gauss,label='Theo.')
    plt.legend()
    plt.show()

# Other functions

def binning(r,rmin,rmax):

    global invbinw
    global cnt

    s = r - rmin
    
    if ( s < rmax-rmin ):
       ibin = int(invbinw*s)
       cnt[ibin] += 1

if __name__=="__main__":
    main()
