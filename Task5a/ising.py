# Monte Carlo integration methods for Computational Physics I
# Wei Chen, EPFL, 11/2011"""

import numpy as np
from random import seed, random, randint
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML




class ising():
    """Simulate a Ising model by Metropolis algorithm""" 
    def __init__(self, ngrid=20, nstep=1000, nskip=10, neq=1000, init_status='warm', kT=1, scheme='scan'):
        self.ngrid = ngrid
        self.init_status = init_status
        self.kT = kT
        self.nstep = nstep
        self.nskip = nskip
        self.scheme = scheme
        self.neq = neq

    def setup(self):
        """set up the initial spin configuration"""
        ng = self.ngrid
        if self.init_status == 'warm':
            s = np.random.rand(self.ngrid, self.ngrid)
            s = 2*np.ceil(s-0.5)-1
        elif self.init_status == 'cold':
            s = np.ones((ng, ng))
        elif self.init_status == 'hot':
            s = np.ones((ng, ng))
            for i in range(0,ng-1+ng%2,2):
                for j in range(0,ng-1,2):
                    s[i,j+1] *= -1
            for j in range(0,ng-1+ng%2,2):
                for i in range(0,ng-1,2):
                    s[i+1,j] *= -1
        return s

    def energy(self, s, ng):
        """ calculate the initial energy with periodic boundary condition"""
        E = 0
        for i in range(ng-1):
            for j in range(ng-1):
                E -= s[i,j]*(s[i-1,j]+s[i+1,j]+s[i,j+1]+s[i,j-1])
            E -= s[i,ng-1]*(s[i-1,ng-1]+s[i+1,ng-1]+s[i,0]+s[i,ng-2])
        for j in range(ng-1):
            E -= s[ng-1,j]*(s[ng-1,j-1]+s[ng-1,j+1]+s[ng-2,j]+s[0,j]) 
        E -= s[ng-1,ng-1]*(s[ng-2,ng-1]+s[0,ng-1]+s[ng-1,ng-2]+s[ng-1,0])
        return 0.5*E

    def ediff(self, s, u, v):
        """calculate the energy difference when a single spin is flipped"""
        ngrid = self.ngrid
        if u < ngrid-1 and v < ngrid -1:
            snn = s[u,v+1] + s[u,v-1] + s[u-1,v] + s[u+1,v]
        else:
            snn = s[u,(v+1)%ngrid] + s[u,v-1] + s[u-1,v] + s[(u+1)%ngrid, v]
        de = -2*s[u,v]*snn
        #code = """
        #       double snn;
        #       if ((u<ngrid-1) && (v>ngrid-1))
        #           snn = s(u,v+1) + s(u,v-1) + s(u-1,v) + s(u+1,v);
        #       else
        #           snn = s(u,(v+1)%ngrid) + s(u,v-1) + s(u-1,v) + s((u+1)%ngrid, v);
        #       return_val = -2*s(u,v)*snn; 
        #       """  
        #de = weave.inline(code, ['u', 'v', 's', 'ngrid'], type_converters=converters.blitz, compiler='gcc')
        return de

    def magnetization(self, s):
        m = np.sum(s)/self.ngrid**2
        return m

    def walk(self, s, u, v):
        if self.scheme == 'random':
            u = randint(0, self.ngrid-1)
            v = randint(0, self.ngrid-1)
        elif self.scheme == 'scan':
            v += 1
            if v > self.ngrid-1:
                v = 0
                u += 1
                if u > self.ngrid-1:
                    u = 0
        s[u,v] *= -1
        return s, u, v

    def loop(self):
        self.M = self.magnetization(self.lat)
        latnew, self.u, self.v = self.walk(np.copy(self.lat), self.u, self.v)
        self.dE = self.ediff(latnew, self.u, self.v)
        prob = math.exp(-self.dE/self.kT)
        if self.dE < 0 or prob > random():
            self.lat = np.copy(latnew)
            self.E0 = self.E0 + self.dE
        else:
            self.dE = 0

    def write_header(self):
        if self.verbose >= 1:
            print("#      i      E/N       dE        M")

    def write(self):
        if self.verbose >= 1:
            print("{0:8} {1:8.5f} {2:8.5f} {3:8.3f}".format(self.ii, \
            self.E0/self.N, self.dE, self.M))
        if self.verbose > 1:
            print(self.lat)

    def calc_chi(self):
        self.chi = self.m2avg - self.mavg**2
        self.chi /= self.kT 
    
    def run(self, verbose):
        self.ii = 0               # sampling sequence number
        nsampl = self.nstep*self.nskip
        self.lat0 = self.setup()  # the initial spin configuration
        self.lat = self.lat0
        self.E0 = self.energy(self.lat, self.ngrid)  # old energy
        self.N = self.ngrid**2                       # number of lattice sites
        self.M = self.magnetization(self.lat)        # magnetization
        self.arrM= []                                # Array of magnetization
        self.arrM.append(self.M)
        self.dE = 0                                  # energy difference

        # we store the snapshot of spin configuration into a 'pool'
        self.pool = np.zeros((self.nstep+1, self.ngrid, self.ngrid)) 
        self.pool[0] = self.lat0
        self.verbose = verbose
        self.write_header()
        self.write()
        self.u = 0; self.v = 0
 
        if self.neq > 0:
            for i in range(self.neq*self.nskip):
                self.loop()

        self.mavg = 0                # <M>
        self.m2avg = 0               # <M2>
        self.E0avg = 0               # <E>
        self.E02avg = 0              # <E2>
        self.chi = 0                 # susceptiblitiy 

        for i in range(nsampl):
            self.loop()
            if i%self.nskip == 0:
                self.ii +=1              # sequence number   
                self.pool[self.ii] = self.lat
                self.arrM.append(self.M)
                self.mavg += self.M
                self.m2avg += self.M**2
                self.E0avg += self.E0
                self.E02avg += self.E0**2
                self.write()

        self.E0 = self.energy(self.lat, self.ngrid)  # old energy
        self.mavg /= self.ii
        self.m2avg /= self.ii
        self.E0avg /= self.ii
        self.E02avg /= self.ii
        self.calc_chi()
        self.arrM=np.array(self.arrM)

    def animation(self):
        # plot an animation 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.autoscale(tight=True)
        ax.set_axis_off()
        plt.autumn()
        plt.axis('equal')
        i = 0
        
        ims = []
        for i in range(self.nstep):
            im = plt.imshow(self.pool[i], vmin=-1, vmax=1)
            ims.append([im])    
        
        ani = animation.ArtistAnimation(fig, ims, interval=200, blit= False, repeat_delay =1000)
        plt.show()

    def animation_jupyter(self, plot_every=1):
        fig, [ax, ax1]  = plt.subplots(1,2)

        grid1=ax.imshow(self.pool[0],animated=True)
        ax.set_xticks(np.arange(0,self.ngrid+1,5))        
        ax.set_yticks(np.arange(0,self.ngrid+1,5))        

        line1, = ax1.plot([],[])
        ax1.axhline(1.0,lw=.7,color='r')
        ax1.set_xlim([0,self.nstep])
        ax1.set_ylim([0,1.05])
        ax1.set_ylabel('|M|')
        ax1.set_xlabel('step')
        ax1.set_aspect(self.nstep)
        plt.tight_layout()

        pool_reduced=self.pool[::plot_every]
        frames_reduced=np.arange(self.nstep)[::plot_every]
        n_reduced=len(pool_reduced)
        print(f'Plotting {n_reduced} frames:')

        def update(n):
            fig.suptitle(f'Step {frames_reduced[n]}')
            grid1.set_array(pool_reduced[n])

            line1.set_data(range(frames_reduced[n]),np.abs(self.arrM[:frames_reduced[n]]))
            return (grid1, line1)
        
        anim = animation.FuncAnimation(fig, update, 
                                       frames=n_reduced-1, interval=20, blit=True)        # create a figure and axes

        return anim
    


    def plot_step(self, n):
        fig, [ax, ax1]  = plt.subplots(1,2)

        grid1=ax.imshow(self.pool[0],animated=True)
        ax.set_xticks(np.arange(0,self.ngrid+1,5))        
        ax.set_yticks(np.arange(0,self.ngrid+1,5))        

        line1, = ax1.plot([],[])
        ax1.axhline(1.0,lw=.7,color='r')
        ax1.set_xlim([0,self.nstep])
        ax1.set_ylim([0,1.05])
        ax1.set_ylabel('|M|')
        ax1.set_xlabel('step')
        ax1.set_aspect(self.nstep)
        plt.tight_layout()

        def update(n):
            fig.suptitle(f'Step {n}')
            grid1.set_array(self.pool[n])

            line1.set_data(range(n),np.abs(self.arrM[:n]))
            return (grid1, line1)
        
        grid1, line1 = update(n)
       
        

        return fig
        # conda install -c conda-forge ffmpeg
