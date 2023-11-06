"""
Testing random number generators provided by Numerical Recipes book (2nd ed.)
First wrote by Gabriele Sclauzero (EPFL, Lausanne), Sept. 2010 in FORTRAN
Adapted to python 3 by Arnaud Lorin (EPFL, Lausanne), Nov 2021
The code do two things:
- Create an histogram with the random number over ntry * nexp numbers.
- Compute the chi^2 and chi^2 2D over nexp.
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

import ran

def main(iran=None,ntry=None,nbins2=None,nexp=None):

    
    #Input parameters
    if iran==None:
        iran=int(input("iran:"))
    if ntry==None:
        ntry=int(input("number of tries:"))
    if nbins2==None:
        nbins=int(input("number of bins:"))
    else:
        nbins=nbins2
    if nexp==None:
        nexp=int(input("number of experiments:"))
    
    # Parameters X^2
    nbin2=50       # Number of bin for the calculation of X^2 and X^2 2d
    chisqx=40.0    # Maximum accepted value for X^2
    chisq2dx=200.0 # Maximum accepted value for X^2 2d
    nchisq=0       # Count the number of accepted X^2 (lower than chisqx)
    nchisq2d=0     # Same for X^2 2d
    
    nbinx=1000
    
    binw = 1.0 / nbins 
    invdchisq = float(nbin2) / chisqx
    invdchisq2d = float(nbin2) / chisq2dx
    
    # Tables
    cnt=np.zeros(nbins)
    cnt0=np.zeros(nbins)
    cnt2d=np.zeros((nbins,nbins))
    cnt2d0=np.zeros((nbins,nbins))
    distr=np.zeros(nbin2)
    distr2d=np.zeros(nbin2)
    intdistr=np.zeros(nbin2)
    
    if ( ntry <= 0 or nbins <= 0 or nexp <= 0 ):
       raise Exception("wrong input")
    if ( nbins > nbinx ):
       raise Exception("maximum number of bins= "+str(nbinx))
    
    # Random seed
    x=datetime.datetime.now()
    seed=int(x.microsecond/1000)+1000*(x.second+60*(x.minute+60*x.hour))
    #seed =  imsec              +1000*(isec    +60*(imin    +60*ihr   ))
    seed = seed | 1 # to ensure that seed is an odd number
    
    # Fixed seed (for debugging purposes)
    #seed=123456789
    
    # Select the RNG
    if iran==0:
        ran_f=ran.ran0
    elif iran==1:
        ran_f=ran.ran1
    elif iran==2:
        ran_f=ran.ran2
    elif iran==3:
        ran_f=ran.ran3
    else:
        raise Exception('unknown generator ran '+str(iran))

    # Loop the experience, for each experience do one chi^2 point
    for iexp in range(nexp):
        resl=[]
        # Generate ntry random number
        for itry in range(ntry):
            #res,seed=ran_f(seed)
            res1,seed=ran_f(seed)
            resl.append(res1)

        # Bin the generated number to an histogram with nbins bin
        binning_array_all(resl,cnt,cnt2d,nbins)

        # Compute X^2 for R_{i}
        chisq=0.0
        nideal=float(ntry)/nbins
        for i in range(nbins):
            chisq += (cnt[i] - cnt0[i] - nideal)**2
            cnt0[i]=cnt[i]
        chisq=chisq/nideal
    
        nchisq=chisq_distr(chisq,distr,chisqx,invdchisq,nchisq)
    
        # compute X^2 for (R_{i},R_{i-1})
        chisq = 0.0
        nideal = ntry/2.0/nbins**2
        for i in range(nbins):
            for j in range(nbins):
              chisq += (cnt2d[i,j] - cnt2d0[i,j] - nideal)**2 
              cnt2d0[i,j] = cnt2d[i,j]
        chisq = chisq / nideal
        
        nchisq2d=chisq_distr2d(chisq,distr2d,chisq2dx,invdchisq2d,nchisq2d)
        
    # print histogram to file
    name='ran{}_{}x{}-{}'.format(iran,nexp,ntry,nbins)
    
    # write the distribution to a file
    filename='histo-'+name
    f=open(filename,'w')
    for i in range(nbins):
        f.write('{:7.6f} {:9.6f}\n'.format(binw*(i+0.5),cnt[i]/(nexp*ntry)*nbins))
    f.close()
    
    if nchisq > nbin2:
      # write the X^2 distribution
      filename='chisq-'+name
      f=open(filename,'w')
      intdistr[0]=distr[0]/2.0
      # Compute the integral
      for i in range(1,nbin2):
          intdistr[i]=intdistr[i-1]+(distr[i-1]+distr[i])/2
      for i in range(nbin2):
          f.write('{:9.5f} {:9.5f} {:9.5f}\n'.format((i+0.5)/invdchisq,distr[i]*invdchisq/nexp, intdistr[i]/nexp ) )
      f.close()
      
      # write the 2d X^2 distribution
      filename='chisq2d-'+name
      f=open(filename,'w')
      intdistr[0]=distr2d[0]/2.0
      # Compute the integral
      for i in range(1,nbin2):
          intdistr[i]=intdistr[i-1]+(distr2d[i-1]+distr2d[i])/2
      for i in range(nbin2):
          f.write('{:9.5f} {:9.5f} {:9.5f}\n'.format((i+0.5)/invdchisq2d,distr2d[i]*invdchisq2d/nexp, intdistr[i]/nexp ) )
      f.close()
      
    else:
       print("too few X^2 values: try to increase nexp")
    
    # plot
    x=[]
    y=[]
    filename='histo-'+name
    for l in open(filename,'r'):
        x.append(float(l.split()[0]))
        y.append(float(l.split()[1]))
    x=np.array(x)
    y=np.array(y)
    width=x[1]-x[0]
    plt.bar(x,y,width,facecolor='none',edgecolor='black',label=name)
    plt.xlim([0,1])
    plt.show()


    if nchisq > nbin2:
        filename='chisq-'+name
        f=open(filename,'r')
        lx=[]
        ly=[]
        lyt=[]
        for l in f:
            x=l.split()
            x0=float(x[0])
            lx.append(x0)
            ly.append(float(x[2]))
            lyt.append(scipy.special.gammainc((nbins-1)/2,x0))  
        lx=np.array(lx)
        ly=np.array(ly)
        lyt=np.array(lyt)
        plt.plot(lx/2,ly,label=name)
        plt.plot(lx,lyt,label='Theo.')
        plt.legend()
        plt.ylabel('$\chi^2$')
        plt.show()

        filename='chisq2d-'+name
        f=open(filename,'r')
        lx=[]
        ly=[]
        lyt=[]
        for l in f:
            x=l.split()
            x0=float(x[0])
            lx.append(x0)
            ly.append(float(x[2]))
            lyt.append(scipy.special.gammainc(60*(nbins/11)**2,x0))  
        lx=np.array(lx)
        ly=np.array(ly)
        lyt=np.array(lyt)
        plt.plot(lx/2,ly,label=name)
        plt.plot(lx,lyt,label='Theo.')
        plt.legend()
        plt.ylabel('$\chi^2$ 2d')
        plt.show()


# Other functions

def binning_array_all(r,cnt,cnt2d,nbins):

    for i in range(len(r)//2):
        #print(i,len(r),2*i+1,2*i)
        ibin1=int(nbins*r[2*i])
        ibin2=int(nbins*r[2*i+1])
        cnt[ibin1]+=1
        cnt[ibin2]+=1
        cnt2d[ibin1,ibin2]+=1
    if len(r)%2==1:
        ibin1=int(nbins*r[-1])
        cnt[ibin1]+=1

def chisq_distr(x2,distr,chisqx,invdchisq,nchisq):

    if x2 < chisqx:
        ibin=int(invdchisq*x2)
        distr[ibin]+=1
        nchisq+=1
    return nchisq

def chisq_distr2d(x2,distr2d,chisq2dx,invdchisq2d,nchisq2d):

    if x2 < chisq2dx:
        ibin=int(invdchisq2d*x2)
        distr2d[ibin]+=1
        nchisq2d+=1
    return nchisq2d



if __name__=="__main__":
    main()
