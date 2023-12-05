"""
Monte Carlo code
"""

from numba import jit, prange
import math
import random
import numpy as np
import time



# Initialize the position of the particles in a cubic shape
def ini_pos(npart, box):
    """_summary_

    Args:
        npart (_type_): _description_
        box (_type_): _description_

    Returns:
        _type_: _description_
    """
    nl=int((npart)**(1.0/3))+1
    if nl==0:nl=1
    delta=box*1.0/nl                # distance between two atoms for initialization
    ipart=0
    pos=np.zeros((npart,3))

    for i in range (nl) :
        for j in range (nl):
            for k in range (nl):
                if ipart < npart :
                  X=i*delta
                  Y=j*delta
                  Z=k*delta
                  pos[ipart]=np.array([X,Y,Z])
                  ipart += 1

    # add random numbers
    delta/=10000.0
    for ipart in range (npart):
        pos[ipart,0] += delta*(random.uniform(0,1)-0.5)
        pos[ipart,1] += delta*(random.uniform(0,1)-0.5)
        pos[ipart,2] += delta*(random.uniform(0,1)-0.5)

    return pos

# Initialize the position of the particles in a face centered cubic shape
def ini_pos_fcc(npart, box):

    nl=int((npart/4.0)**(1.0/3))+1
    if nl==0:nl=1
    delta=box*1.0/nl                # distance between two atoms for initialization
    ipart=0
    pos=np.zeros((npart,3))

    for i in range (nl) :
        for j in range (nl):
            for k in range (nl):
                if ipart < npart :
                  X=i*delta
                  Y=j*delta
                  Z=k*delta
                  pos[ipart]  =np.array([X          ,Y          ,Z])
                  pos[ipart+1]=np.array([X+0.5*delta,Y+0.5*delta,Z])
                  pos[ipart+2]=np.array([X+0.5*delta,Y          ,Z+0.5*delta])
                  pos[ipart+3]=np.array([X          ,Y+0.5*delta,Z+0.5*delta])
                  ipart += 4

    # add random numbers
    delta/=10000.0
    for ipart in range (npart):
        pos[ipart,0] += delta*(random.uniform(0,1)-0.5)
        pos[ipart,1] += delta*(random.uniform(0,1)-0.5)
        pos[ipart,2] += delta*(random.uniform(0,1)-0.5)

    return pos

# Compute the potential energy of the system
@jit(nopython=True)
def energy(pos, r2cut, eps, sig, npart, box, rho, vol, beta):
  etot=0
  vir=0
  for i in range (npart) :
    for j in range (npart) :
      if j > i :

        # Distance between two particle i,j, up to box
        dis2=   min ((pos[i,0]-pos[j,0])**2, (abs(pos[i,0]-pos[j,0])-box)**2 )
        dis2 += min ((pos[i,1]-pos[j,1])**2, (abs(pos[i,1]-pos[j,1])-box)**2 )
        dis2 += min ((pos[i,2]-pos[j,2])**2, (abs(pos[i,2]-pos[j,2])-box)**2 )

        if dis2 <=  r2cut :
          r6i=(sig**2/dis2)**3
          etot += 4*eps*(r6i**2-r6i)
          vir += 48*eps*(r6i**2-0.5*r6i)

  press = rho/beta + vir/(3*vol)
  return etot,press


# Compute the potential energy of one particle ipick at position new_pos
@jit (nopython=True)
def esingle(ipick, pos, new_pos, npart, box, r2cut, eps, sig):
  e1=0
  for j in range (npart) :
      if j != ipick :

        # Distance between two particle ipick,j, up to box
        dis2=   min ((new_pos[0]-pos[j,0])**2, (abs(new_pos[0]-pos[j,0])-box)**2 )
        dis2 += min ((new_pos[1]-pos[j,1])**2, (abs(new_pos[1]-pos[j,1])-box)**2 )
        dis2 += min ((new_pos[2]-pos[j,2])**2, (abs(new_pos[2]-pos[j,2])-box)**2 )

        if dis2 <=  r2cut :
          e1 += 4*eps*((sig/dis2)**6-(sig/dis2)**3)
  return e1


# Monte Carlo move: a "move" is made of ndispl tries of displacement
# return
#  -fraction of accepted displacement over ndispl
#  if toten=True
#  -total energy
#  - pressure
# Activating jit break the reproducibility but results seem still correct
#@jit#  (nopython=True)
def move(pos, new_pos, dmax, ndispl, npart, box, r2cut, eps, sig, beta, rho, vol, toten=False):
    naccpt=0
    for i in range (ndispl) :

        # select a particle at random
        ipick=int(random.uniform(0,npart))

        # Give the particle a random displacement
        new_pos[0]=pos[ipick,0]+dmax*(random.uniform(0,1)-0.5)
        new_pos[1]=pos[ipick,1]+dmax*(random.uniform(0,1)-0.5)
        new_pos[2]=pos[ipick,2]+dmax*(random.uniform(0,1)-0.5)

        # keep the particle in the box
        new_pos= new_pos % box


        # Calculate energy difference between new configuration
        # and old configuration.
        e1=esingle(ipick, pos, pos[ipick], npart, box, r2cut, eps, sig)
        e2=esingle(ipick, pos, new_pos, npart, box, r2cut, eps, sig)
        ediff=e2-e1

        # Test if new position is accepted
        # If ediff <= 0 always accepted ( exp will be > 1)
        # If ediff >  0 have to pass the test
        if random.uniform(0,1) < math.exp(-beta*ediff) :
          pos[ipick]=new_pos
          naccpt += 1

    # Compute the acceptance rate
    frac=naccpt/ndispl
    if toten:
        en,pr=energy(pos, r2cut, eps, sig, npart, box, rho, vol, beta)
    else :
        en=0
        pr=0
    return frac, en, pr

# -------------------g(r)-and-s(k)---------------------------

@jit (nopython=True)
def gofr(gr, pos, box, rho, nbins, npart):
    Lbin=box/nbins/2
    for i in range (npart) :
        for j in range (i+1, npart) :
            if j !=i :
                # periodic boundary condition
                dis2=   min ((pos[i,0]-pos[j,0])**2, (abs(pos[i,0]-pos[j,0])-box)**2 )
                dis2 += min ((pos[i,1]-pos[j,1])**2, (abs(pos[i,1]-pos[j,1])-box)**2 )
                dis2 += min ((pos[i,2]-pos[j,2])**2, (abs(pos[i,2]-pos[j,2])-box)**2 )
                rij=math.sqrt(dis2)
                if rij < box/2:
                  gr[int(rij/Lbin)]+=1

def ave_gofr(gr,L,N,nbins,nave):
    gofr_mean = np.zeros(nbins)
    r = np.zeros(nbins)
    Lbin = 0.5 * L / nbins
    V = L**3

    for ii in range(nbins):
        r[ii] = Lbin * (ii + 0.5)
        gofr_mean[ii] = 2*V/(N*(N-1)) * gr[ii]/nave / (4*np.pi*r[ii]**2*Lbin)
    return {'g': gofr_mean, 'r': r}

def sofk_FT(gofr, L, N, qmax=30, nqvec=300):
    v = np.zeros(len(gofr['r']))
    sofk = np.zeros(nqvec)
    k = np.zeros(nqvec)
    dq = qmax/float(nqvec-1)
    rho = N / L**3
    dr = gofr['r'][1]-gofr['r'][0]
    for ii in range(nqvec):
        k[ii] = ii*dq

    for ii in range(nqvec):
        for jj in range(len(gofr['r'])):
          phase = k[ii]*gofr['r'][jj]
          # Small sinus case
          if abs(phase) < 1e-8:
            v[jj] = gofr['r'][jj]**2 * \
                (gofr['g'][jj]-1)*(1-phase**2/6 * \
                 (1-phase**2/20*(1-phase**2/42)))
            # Integrand r[g(r)-1]sin(q*r)/q
          else:
            v[jj] = gofr['r'][jj]*(gofr['g'][jj]-1)*np.sin(phase)/k[ii]

        sofk[ii] = 1 + 4*np.pi*rho*np.trapz(v, dx=dr)

    return {'s': sofk, 'k': k}

# tail correction for energy
def cor_en(rcut,rho, sig, eps):
    sig3=sig**3
    ri3=sig3*1.0/(rcut**3)
    coru = 2*np.pi*eps*4*(rho*sig3)*(ri3*ri3*ri3/9-ri3/3)
    return coru

# tail correction for pressure
def cor_pr(rcut,rho, sig, eps):
    sig3=sig**3
    ri3=sig3*1.0/(rcut**3)
    corp = 4*np.pi*eps*4*(rho**2)*sig3*(2*ri3*ri3*ri3/9-ri3/3)
    return corp

# --------------------I/O functions--------------------

def dump_pos(fname, pos, N, L, dmax):
    with open(fname,'w') as f:
        f.write("% {} {:16.12f} {:16.12f} {:16.12f} {:16.12f} \n".format(N, dmax, L, L, L))
        np.savetxt(f, pos, fmt=['    %.25E','%.25E','%.25E'])

def read_pos(fname):
    parameters = open(fname).readline().strip().split()
    N = int(parameters[1])
    dmax = float(parameters[2])
    L = float(parameters[3])

    pos = np.loadtxt(fname, comments="%")
    return pos, N, L, dmax
