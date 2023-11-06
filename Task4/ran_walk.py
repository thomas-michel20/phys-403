"""
Generates a random walk in one dimension 
First wrote by Gabriele Sclauzero (EPFL, Lausanne), Sept. 2010 in FORTRAN
Adapted to python 3 by Arnaud Lorin (EPFL, Lausanne), Nov 2021
Input
-Number of walker
-Number of step for each walker
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import ran

# Number of walker
#nwalk=int(input("enter # of walkers: "))

# Number of steps
#nstep=int(input("enter # of steps: "))

def ran_walk(nwalk=None, nstep=None):
    # Random seed
    x=datetime.datetime.now()
    seed=int(x.microsecond/1000)+1000*(x.second+60*(x.minute+60*x.hour))
    #seed =  imsec              +1000*(isec    +60*(imin    +60*ihr   ))
    seed = seed | 1 # to ensure that seed is an odd number

    # Fixed seed (for debugging purposes)
    #seed=123456789

    x2ave=np.zeros(nstep)


    for i in range(nwalk):

        x=0 # Current location of the walker

        for j in range(nstep):

            res,seed =ran.ran2(seed)
            if res < 0.5:
                x += 1
            else:
                x -= 1

            x2ave[j] += x**2

    x2ave=x2ave/nwalk

    # Write mean square displacement to file
    filename='msd_{}walkers-{}steps'.format(nwalk,nstep)
    f=open(filename,'w')
    for i in range(nstep):
        f.write('{:14d} {:14d}\n'.format(i+1,int(x2ave[i]) ))
    f.close()

    plt.plot(x2ave,label= filename)
    plt.plot((0,nstep),(0,nstep),label='Theo.')
    plt.xlabel('Steps')
    plt.ylabel('$x^2$')
    plt.legend()
    plt.show()
