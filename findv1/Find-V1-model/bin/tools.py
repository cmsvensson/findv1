import numpy

def delta(f,f0):
    f1 = abs(f-f0) 
    
    out = numpy.zeros([f1.shape[0],f1.shape[3],1,1])
    ii = numpy.arange(0,f1.shape[0])
    jj = f1.argmin(axis=3)
    out[ii,jj[:,0,0],0,0] = 1.0;
    out.shape
    return out

def makeCrf(ori=0.0,    #in degrees
        cycles=1.0,
        phase=0.0,    #in degrees
        gratType="sin",
        diam=1.0,
        tau1 = 0.004,
        tau2 = 0.0053,
        surround = False):
    
    n1 = 9.0; n2 = 10.0; zeta = 0.9
    sigma = diam/2.0
    a=1/(sigma**2.0)
    freqs = numpy.fft.fftfreq(128,3.0/128.0)
    phase *= numpy.pi/180
    ori *= numpy.pi/180
    fx = cycles/5.0*numpy.cos(ori);
    fy = cycles/5.0*numpy.sin(ori);
    tf = numpy.fft.fftfreq(64,1.0/64.0)
    
    f1 = numpy.power((1.0j*2.0*numpy.pi*tf*tau1+1),-n1)*numpy.exp(-1.0j*2.0*numpy.pi*tf*0.0001)
    f2 = numpy.power((1.0j*2.0*numpy.pi*tf*tau2+1),-n2)*numpy.exp(-1.0j*2.0*numpy.pi*tf*0.0001)
    
    f=numpy.zeros([64,1,1],'complex')
    ht = numpy.zeros([64,1,1],'complex')
    f[:,0,0] = (f1 - zeta*f2)/(2.0*numpy.pi)**(1.0/2.0)
    ht[:,0,0] = 1.0j*numpy.sign(tf)/(2.0*numpy.pi)**(1.0/2.0)
    yy = numpy.linspace(-1.0,4.0,128.0)
    xx= yy.reshape(-1,1)
    
    masked = numpy.fft.fftn(numpy.sin(2*numpy.pi*(fx*xx+fy*yy)+phase)*makeGaussian(sigma)) 
    #masked = -5.0j*numpy.pi**(1.0/2.0)/(4.0*(2.0*numpy.pi)**(3.0/2.0)*a)*(numpy.exp(-8.0/a*((freqs.reshape(-1,1)-fx)**2+(freqs-fy)**2))*numpy.exp(-1.0j*phase*(numpy.sqrt(fx**2.0+fy**2.0)))-numpy.exp(-8.0/a*((freqs.reshape(-1,1)+fx)**2+(freqs+fy)**2))*numpy.exp(1.0j*phase*(numpy.sqrt(fx**2.0+fy**2.0))))*numpy.exp(-1.0j*1.5*2.0*numpy.pi*(freqs.reshape(-1,1)+freqs))
    
    quad_phase = phase-numpy.pi/2.0
    quad_masked = numpy.fft.fftn(numpy.sin(2*numpy.pi*(fx*xx+fy*yy)+quad_phase)*makeGaussian(sigma)) 
    #quad_masked = -5.0j*numpy.pi**(1.0/2.0)/(4.0*(2.0*numpy.pi)**(3.0/2.0)*a)*(numpy.exp(-8.0/a*((freqs.reshape(-1,1)-fx)**2+(freqs-fy)**2))*numpy.exp(-1.0j*quad_phase*(numpy.sqrt(fx**2.0+fy**2.0)))-numpy.exp(-8.0/a*((freqs.reshape(-1,1)+fx)**2+(freqs+fy)**2))*numpy.exp(1.0j*quad_phase*(numpy.sqrt(fx**2.0+fy**2.0))))*numpy.exp(-1.0j*1.5*2.0*numpy.pi*(freqs.reshape(-1,1)+freqs))
    
    M = f*masked+f*quad_masked*ht
    
    return M

def makeGaussian(diam=1.0):
    x = numpy.zeros([1,128])
    y = numpy.zeros([128,1])
    x[0,:] = numpy.linspace(0.0,3.0,128)
    y[:,0] = numpy.linspace(0.0,3.0,128)
    out = numpy.exp(-2.0*numpy.pi/(4.0*diam**2.0)*((x-1.5)**2.0+(y-1.5)**2.0))

    return out#8.0/a*numpy.exp(-8.0/a*((freqs.reshape(-1,1))**2+(freqs)**2))*numpy.exp(-1.0j*1.5*2.0*numpy.pi*(freqs.reshape(-1,1)+freqs))

def rectify(matrix, method='half'):
    """Half-wave or full-wave recitfication"""
    if method=='half':
        output = numpy.where(matrix<0, 0, matrix)
    else:
        output = numpy.asarray((matrix**2)**0.5)
        
    if type(matrix) in [float, int]:
        #return a single value
        try: #some versions of num return a rank-1 array
            return output[0]
        except: #others return a rank-0 array wihch needs this
            return float(output)
    else:
        #return a matrix
        return output
    
def getFourierPower(data, freq=None, ax=1, t=None):
    """Returns the Fourier power, either for the peak of the distribution (excluding the DC) or
    for the specified frequency if given
    
    **usage:**
      ``amp, phase, F = getFourierPower(data)``
      or
      ``amp, phase, F = getFourierPower(data, Freq)``
      
    *freq* is specified in cycles across the sample
    F is the frequency used in the calculation.    
    
    """

    if t==None:
        t = numpy.linspace(0.0,1.0,64)
        Fs = 1.0/t[1] 
        L = t.shape[0]
        ft = numpy.fft.fft(data,axis=ax)
        freqs = numpy.fft.fftfreq(64,1.0/64.0)
        ftAmp = abs(ft)/data.shape[1]*2
        ftPhase = numpy.arctan2(ft.real, ft.imag)
        if freq==None:
            ii = numpy.argmax(abs(ft[:,1:])**2.0,axis=1)
            ii += 1
        else:
            fr = numpy.asarray([freq])
            fr = fr.flatten()
            ii = numpy.ones(fr.shape)
            for kk in range(0,fr.shape[0]):
                ii[kk] = numpy.argmin(abs(freqs-fr[kk]))
        
        jj = numpy.arange(0,ftAmp.shape[0])
        amp = ftAmp[jj,ii.astype('int')]
        phase = ftPhase[jj,ii.astype('int')]
    else:
        Fs = 1.0/t[1]
        L = t.shape[0]
        ft = numpy.fft.fft(data,n=NFFT,axis=ax)[1:]/L#skip the DC
        ftAmp = abs(ft)/len(t)*2
        ftPhase = numpy.arctan2(ft.real, ft.imag)
        if freq==None:
            ii = argmax(abs(ft)**2.0,0)
        else: 
            ii = freq-1
            
        amp = ftAmp[:,ii]
        phase = ftPhase[:,ii]
    return amp, phase, ii

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)