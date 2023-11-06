# All algorithms refer to the one proposed in Numerical Recipes book (2nd ed.) 


ia=16807
im=2147483647
am=1./im
iq=127773
ir=2836
MASK=123459876

# Schrage's algorithm
def ran0(idum):

    idum=idum ^ MASK
    k=idum//iq
    idum=ia*(idum-k*iq)-ir*k
    if idum < 0: idum=idum+im
    ran0=am*idum
    idum= idum ^ MASK
    return(ran0,idum)

# Additional Shuffle algorithm
def ran1(idum):
    ia=16807
    im=2147483647
    am=1./im
    iq=127773
    ir=2836
    ntab=32
    ndiv=1+(im-1)//ntab
    eps=1.2e-7
    rmnx=1-eps
    global iv
    global iy
    try: iy
    except NameError: iy=0

    if (idum <= 0 or iy == 0):  
        iv=[0]*ntab
        idum=max(-idum,1)
        for j in range(ntab+7,-1,-1):
            k=idum//iq
            idum=ia*(idum-k*iq)-ir*k
            if idum < 0: idum+=im
            if j < ntab: iv[j]=idum
        iy=iv[0]
    k=idum//iq
    idum=ia*(idum-k*iq)-ir*k
    if idum < 0: idum+=im
    j=iy//ndiv
    iy=iv[j]
    iv[j]=idum
    ran1=min(am*iy,rmnx)
    return(ran1,idum)



# Add l'Ecuyer technique to increase the period
def ran2(idum):

    ia1=40014
    im1=2147483563
    imm1=im1-1
    am1=1./im1
    iq1=53668
    ir1=12211

    ia2=40692
    im2=2147483399
    iq2=52774
    ir2=3791

    ntab2=32
    ndiv2=1+imm1//ntab2
    eps2=1.2e-7
    rmnx2=1-eps2

    global iv
    global iy
    global idum2
    try: iy
    except NameError: iy=0
    try: idum2
    except NameError: idum2=123456789
    try: iv
    except NameError: iv=[0]*ntab2

    if idum <= 0:
        idum=max(-idum,1)
        idum2=idum
        for j in range(ntab2+7,-1,-1):
            k=idum//iq1
            idum=ia1*(idum-k*iq1)-ir1*k
            if idum < 0: idum+=im1
            if j < ntab2: iv[j]=idum
        iy=iv[0]
    k=idum//iq1
    idum=ia1*(idum-k*iq1)-ir1*k
    if idum < 0: idum+=im1
    k=idum2//iq2
    idum2=ia2*(idum2-k*iq2)-ir2*k
    if idum2 < 0: idum2+=im2
    j=iy//ndiv2
    iy=iv[j]-idum2
    iv[j]=idum
    if iy < 1: iy+=imm1
    ran2=min(am1*iy,rmnx2)
    return(ran2,idum)
        
# Knuth algorithm
def ran3(idum):
    mbig=1000000000
    mseed=161803398
    mz=0
    fac=1./mbig

    global iff,inext,inextp
    global ma
    try: iff
    except NameError: iff=0
    try: ma
    except NameError: ma=[0]*56

    
    if(idum < 0 or iff == 0):
      iff=1
      mj=abs(mseed-abs(idum))
      mj=mj % mbig
      ma[55]=mj
      mk=1
      for i in range(1,55):
        ii= (21*i) % 55
        ma[ii]=mk
        mk=mj-mk
        if(mk < mz): mk=mk+mbig
        mj=ma[ii]
      for k in range(1,5):
          for i in range(1,56):
              ma[i]=ma[i]- ma[1+(i+30) % 55]
              if(ma[i] < mz): ma[i]=ma[i]+mbig
      inext=0
      inextp=31
      idum=1
    inext=inext+1
    if(inext == 56): inext=1
    inextp=inextp+1
    if(inextp == 56): inextp=1
    mj=ma[inext]-ma[inextp]
    if(mj < mz): mj=mj+mbig
    ma[inext]=mj
    ran3=mj*fac
    return (ran3,idum)

# Random generator using pseduo DES
# Not implemented in the fortran version ? TODO ?
def ran4(idum):
    jflmsk=0x007fffff
    jflone=0x3f800000
    if(idum < 0):
        idums=-idum
        idum=1
    irword=idum
    lword=idums
#    call psdes(lword,irword)
    itemp=jflone | (jflmsk & irword)
    ran4=ftemp-1.0
    idum=idum+1
    return(ran4,idum)   
