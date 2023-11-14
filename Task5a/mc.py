# Monte Carlo integration methods for Computational Physics I
# Wei Chen, EPFL, 11/2011
# 2023/11 modified metropolis.run to get correct normalisation in corr, G. Palermo

from random import random
import math

class montecarlo(object):
    def __init__(self, n=1000, verbose=1):
        self.n = n
        self.verbose = verbose
    
    def sampling(self):
        """ performs a simple monte carlo sampling 
            and counts the points inside the circle
            pi_0: calculated pi"""

        n = self.n
        method = "Throw and Count method"
        n_in = 0
        for i in range(n):
            x = random()
            y = random()  
            if x**2+y**2 <= 1:
                 n_in += 1
        self.pi_0 = float(n_in)/n*4
        if self.verbose == 1:
            self.showpi(method, self.pi_0, 0)

    def direct(self):
        """ calculate pi/4 by integrate 1/(1+x^2) from 0 to 1.
            error (ds): (<f**2>-<f>**2)/n 
            pi_1: calculated pi value
            ds_1: error estimate
            derr_1: absolute error"""

        method = "direct Monte Carlo integration"
        f = lambda x: 1.0/(1+x**2) 
        self.pi_1, self.ds_1, self.derr_1 = self.integrate(f)
        if self.verbose == 1:
             self.showpi(method, self.pi_1, self.ds_1)
        return self.pi_1

    def importance(self):
        """ Use a weighted function w(x) to find pi
            w(x) = (4-2x)/3
            y(x) = \int w(x) dx = (4x-x**2)/3
            x    = 2-sqrt(4-3y)
            pi/4 = \int f[x(y)]/w[x(y)] dy |(y=0-->1)
            outputs: pi_2, ds_2, derr_2"""
        
        method = "importance sampling"
        x = lambda y: 2-math.sqrt(4-3*y)
        f = lambda y: 1.0/(1+x(y)**2)
        w = lambda y: (4-2*x(y))/3
        g = lambda y: f(y)/w(y)
        self.pi_2, self.ds_2, self.derr_2 = self.integrate(g)      
        if self.verbose == 1:
            self.showpi(method, self.pi_2, self.ds_2)
 
    def integrate(self, f):
        """ the common block of a Monte Carlo integration """
        n = self.n
        s = 0; ds = 0
        for i in range(n):
            x = random()
            y = f(x)
            s += y
            ds += y**2
        pi = float(s)/n*4
        ds /= n
        ds = math.sqrt(math.fabs(ds-(pi/4)**2)/n)*4
        derr = math.fabs(math.pi-pi)
        return pi, ds, derr

    def showpi(self, method, pi, ds):
        """ print the results """
        print('pi by {0:31.30s}: {1:9.5f} +- {2:<9.5f}'.format(method, pi, ds))

class metropolis(montecarlo):
    """ A Metropolis scheme adpated from 
        "An Introduction to Computational Physics" by Tao Pang"""

    def __init__(self, nsize=100000, nskip=10, nburnin=100000, h=0.2, verbose=1):
        self.nsize = nsize
        self.nskip = nskip
        self.nburnin = nburnin 
        self.n = nsize*nskip
        self.h = h
        self.verbose = verbose

    def run(self):
        self.f = lambda x: 1.0/(1+x**2)
        self.w = lambda x: (4-2*x)/3
        self.g = lambda x: self.f(x)/self.w(x)


        method = "Metropolis algorithm"
        x = random()
        iaccept = 0

        # start burn-in
        xnew = x
        for i in range(self.nburnin):
            xnew, iaccept = self.loop(xnew, iaccept) 
        # burn-in finish

        z = 4.0
        s = 0; ds = 0; iaccept = 0
        c = 0             # <A_{n+l}A_n>
        g0 = self.g(xnew)        

        if self.nskip == 0:
            # if nskip = 0, then we only do burn-in
            # corr is = 1
            self.s = g0*z
            self.ds = self.s**2
            self.corr = 1
        else:
            for i in range(self.n):
                xnew, iaccept = self.loop(xnew, iaccept)
                if (i%self.nskip == 0):
                    u = self.g(xnew)
                    s += u
                    ds += u**2
                    c += u*g0
                    g0 = u

            s=s/self.nsize*z # <A>
            ds=ds/self.nsize*z**2 # <A^2>
            c = c/self.nsize*z**2 # <A_{n+l}A_n>
            corr = (c-s**2)/(ds-s**2)

            ds=math.sqrt(math.fabs(ds-s**2)/self.nsize) # error estimate
            
            naccept = 100.0*iaccept/self.n
            if self.verbose == 1:
                self.showpi(method, s, ds)
                print("accepting rate: {0:.2f}%".format(naccept))
            
            self.s = s
            self.ds = ds
            self.corr = corr
 
    def loop(self, x, iaccept):
        x0 = x                            # the old x
        w0 = self.w(x0)                   # the old probability w(old)  
        x1 = x0 + 2*self.h*(random()-0.5)  # trial step
        w1 = self.w(x1)                   # the new probability w(new)
        if x1 < 0 or x1 > 1:
            x1 = x0
            w1 = w0
        elif w1 > w0*random():          # if w(new)/w(old) > random()
            iaccept += 1             # we then use the new w
        else:
            x1 = x0                   # otherwise we keep the old w and x
            w1 = w0
        #print "new x: {0:.3f}, old x: {1:.3f}".format(x1, x0)
        return x1, iaccept
