"""
Testing correlations in random number generators 
(ran0, ran1, ran2, ... provided by Numerical Recipes book, 2nd ed.)
First wrote by Gabriele Sclauzero (EPFL, Lausanne), Sept. 2010 in FORTRAN
Adapted to python 3 by Arnaud Lorin (EPFL, Lausanne), Nov 2021
Input 
-the RNG (0=ran0,1=ran1,2=ran2,3=ran3,9=LCG
-Number of random number to generate
-Number of dimension for the correlation
"""

import datetime
import numpy as np 
import ran

def main(iran=None, ntry=None, ndim=None, **kwargs):

    global ia,ic,im

    # Number of bins for the binning
    nbins=100

    # Input parameters
    iran=iran
    
    # Parameters of LCG
    if iran==9:
        print("Parameters of the LGC (a,c,m,i0)")
        ia= kwargs['ia']
        ic= kwargs['ic']
        im= kwargs['im']
        i0= kwargs['i0']
        if (ia*im+ic > 2**31-1):
            raise ValueError("this generator does not avoid overflow, chose smaller parameters")


    ntry=ntry
    ndim=ndim

    if ntry <= 0 or ndim <= 0:
        raise Exception("wrong imput")
    
    # Random seed
    x=datetime.datetime.now()
    seed=int(x.microsecond/1000)+1000*(x.second+60*(x.minute+60*x.hour))
    #seed =  imsec              +1000*(isec    +60*(imin    +60*ihr   ))
    seed = seed | 1 # to ensure that seed is an odd number
    
    # Fixed seed (for debugging purposes)
    #seed=123456789
    
    if iran == 9: seed = i0
    
    # Tables
    cnt_prod = np.zeros(nbins)
    cnt_diff = np.zeros(nbins)
    buf = np.zeros(ndim)
    binw= 1.0/nbins
    j=0

    # Small R.N. count
    cntlow = 0
    rlowav = 0
    
    if iran==0:
        ran_f=ran.ran0
    elif iran==1:
        ran_f=ran.ran1
    elif iran==2:
        ran_f=ran.ran2
    elif iran==3:
        ran_f=ran.ran3
    elif iran==9:
        ran_f=lcg
    else:
        raise Exception('unknown generator ran '+str(iran))
    
    # Prepare the file, write the n-dim correlation
    name='-ran{}_{}dim-{}'.format(iran,ndim,ntry)
    filename='corr'+name
    f=open(filename,'w')

    for i in range(ntry):
        res,seed=ran_f(seed)
    
        if i > 0:

            # Test #1: distribution of R_{i-1}*R_{i} and R_{i-1}-R_{i}
            rprod = resold*res
            rdiff = resold-res
            binning(rprod,rdiff,nbins,cnt_prod,cnt_diff)
    
            # Test #2: average of numbers following a very small R.N. 
            if resold < 1e-5:
                cntlow+=1
                rlowav+=res
    
        resold=res
    
        # Test #3: n-dimensional correlation   
        ibuf= j % ndim
        j=j+1
        buf[ibuf]=res
        if ibuf == ndim-1:
            for k in range(ndim):
                f.write('{:14.6f}'.format(buf[k]))
            f.write('\n')

    f.close()

    # Write the distribution of R_{i-1}*R_{i} and R_{i-1}-R_{i}
    filename = 'distprod-ran{}_{}'.format(iran,ntry)
    f=open(filename,'w')
    for i in range(nbins):
        f.write('{:14.6f} {:14.6f} {:14.6f} {:14.6f}\n'.format(binw*(i+0.5),cnt_prod[i]/ntry*nbins,binw*(i+0.5)*2-1,cnt_diff[i]/ntry*nbins))
    f.close()

    # write on output the average 
    if cntlow > 0:
        print('average after R.N.<10^-5 ({} events): {}'.format(cntlow,rlowav/cntlow))
    else:
        print('no events with R.N.<10^-5: increase ntry')



def binning(rp,rd,nbins,cnt_prod,cnt_diff):

    ibin=int(nbins*rp)
    cnt_prod[ibin]+=1
    # binning of the diffrence
    # assumes -1 < r < 1
    ibin=int(nbins*(rd+1)/2)
    cnt_diff[ibin]+=1

def lcg(idum):

    global ia,ic,im
    idum= ( ia*idum + ic ) % im
    lcg= idum/im
    return (lcg,idum)

if __name__=="__main__":
    main()
