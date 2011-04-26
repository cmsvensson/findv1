import numpy, sys, tools

class V1neuron:
    
    def __init__(self,
                 ori = 0.0,
                 sf = 0.0,
                 phase = 0.0,
                 crfGain = 1.0,
                 crfWidth = 0.5,
                 normPoolExp = 1.5,
                 surroundGain = 0.5,
                 surroundWidth = 2.5,
                 pixPerDeg = 128/3.0,
                 sizeDeg = 3.0, #Degrees of the matrix (Default 3.0)
                 sizePix = None,
                 surroundSuppression = 'RoG',
                 tau1 = 0.004,
                 tau2 = 0.0053):
        
        self.version = '0.0'
        self.crfGain = crfGain
        self.crfWidth = crfWidth
        self._crfMaxResp = 1.0
        self.c50 = 0.5
        self.exciteExp = 2.0 #always, to give appropriate behaviour for normPoolExpon
        self.normPoolExp = normPoolExp
        self.surroundGain = surroundGain  
        self.surroundWidth = surroundWidth
        self.surroundSuppression=surroundSuppression
        self._surroundMaxResp = 1.0
        self.ori = ori
        self.sf = sf
        if sf==0: self.sf=0.0000001
        self.phase = phase
        self.tau1 = tau1
        self.tau2 = tau2
        
        #setup the size and scale parameters
        self.sizeDeg = float(sizeDeg)
        if sizePix != None:
            self.sizePix = int(sizePix)
            self.pixPerDeg = self.sizePix/self.sizeDeg
        elif pixPerDeg != None:
            self.sizePix = int(self.sizeDeg*pixPerDeg)
            self.pixPerDeg = float(pixPerDeg)
        else:
            raise StandardError, 'Neuron needs either sizePix or pixPerDeg to determine scale'
        
        self._buildFilters()
        self.crfEnvelope = tools.makeGaussian(self.crfWidth)
        
    def _buildFilters(self):        
        crf = tools.makeCrf(ori = self.ori, cycles = self.sf*self.sizeDeg, diam=self.crfWidth,
                            phase = self.phase, tau1=self.tau1, tau2=self.tau2)
        
        self.crfPrimaryFilter = crf
        self.crfMaxResp = self.__getMaxResponse(crf)
        
        surr = tools.makeCrf(ori = self.ori, cycles = self.sf*self.sizeDeg/1.2, 
                             diam=self.surroundWidth,phase = self.phase, 
                             tau1 = self.tau1, tau2 = self.tau2)
        self.surroundPrimaryFilter = surr
        self.surrMaxResp = self.__getMaxResponse(surr)
        
    def getResponse(self, stim):
         #allow user to give a stimulus object or just a num array
        if isinstance(stim, Stimulus):
            stim = stim.data
        stim.shape
        
        crf = self.crfGain*numpy.real(numpy.fft.ifft(numpy.sum(numpy.sum(self.crfPrimaryFilter*stim,axis=3)/128,axis=2)/128,axis=1))*64/self.crfMaxResp
        
        #crf = self.crfGain*numpy.real(numpy.sum(numpy.sum(numpy.fft.ifftn(self.crfPrimaryFilter*stim,axes=[1,2,3]),axis=3),axis=2))/self.crfMaxResp

        surr = self.surroundGain*numpy.real(numpy.fft.ifft(numpy.sum(numpy.sum(self.surroundPrimaryFilter*stim,axis=3)/128,axis=2)/128,axis=1))*64/self.surrMaxResp
        normPool = self.__getNormPoolResponse(stim)
        outputResp = tools.rectify(crf)/(1+tools.rectify(surr))
        output = (outputResp**self.exciteExp)/((normPool*2)**self.normPoolExp)
        return output
       
    def __getMaxResponse(self,filt): 
        return numpy.sum(abs(numpy.real(numpy.fft.ifftn(filt,axes=[-3,-2,-1])*64)))
    
    def __getNormPoolResponse(self,stim):
        if isinstance(stim, Stimulus):
            stim = stim.data 
        rmsCont = numpy.std(numpy.reshape(numpy.real(numpy.fft.ifftn(stim,axes=[1,2,3])*64)*self.crfEnvelope,[stim.shape[0],stim.shape[1],stim.shape[2]**2]),2)
        return rmsCont
    
class Stimulus:   
    def __init__(self,
        ori=0.0,    #in degrees
        cycles=1.0,
        phase=0.0,    #in degrees
        gratType="sin",
        contr=1.0,
        driftrate=0.0,
        diameter=1.0):
        
        try:
            clen = len(contr)
        except:
            clen = 1
        try:
            olen = len(ori)
        except:
            olen = 1
        try:
            tlen = len(driftrate)
        except:
            tlen = 1
        try:
            slen = len(diameter)
        except:
            slen = 1
        try:
            sflen = len(cycles)
        except:
            sflen = 1
        contrast = numpy.zeros([clen,1,1,1])
        contrast[:,0,0,0] = contr
        orientation = numpy.zeros([olen,1,1,1])
        orientation[:,0,0,0] = ori
        drift = numpy.zeros([tlen,1,1,1])
        drift[:,0,0,0] = driftrate
        sigma = numpy.zeros([slen,1,1,1])
        sigma[:,0,0,0] = (diameter/2)
        cyc = numpy.zeros([sflen,1,1,1])
        cyc[:,0,0,0] = cycles
        freqs = numpy.fft.fftfreq(128,3.0/128.0)
        phase *= numpy.pi/180
        orientation *= numpy.pi/180
        fx = cyc/5.0*numpy.cos(orientation+0.0001);
        fy = cyc/5.0*numpy.sin(orientation+0.0001);
        tf = numpy.fft.fftfreq(64,2.0/64.0)
        t =  numpy.linspace(0.0,1.0,64).reshape(-1,1,1)
        yy = numpy.linspace(-1.0,4.0,128.0)
        xx = yy.reshape(-1,1)
        if(gratType == "sin"):
            grat = contrast*numpy.sin(2*numpy.pi*(fx*xx+fy*yy+drift*t))*(numpy.sqrt((xx-1.5)**2.0+(yy-1.5)**2.0)<=(sigma))
            #sa = numpy.sqrt((freqs.reshape(-1,1)-fx)**2+(freqs-fy)**2)
            #sam = numpy.sqrt((freqs.reshape(-1,1)+fx)**2+(freqs+fy)**2)
            #x0 = 1.5; y0 = 1.5
            #self.data = -5.0j*contrast*sigma*numpy.pi**(1.0/2.0)/(4.0*(2.0*numpy.pi)**3.0)*(1.0/sa*scipy.special.j1(sigma*sa)*numpy.exp(-1.0j*phase*(numpy.sqrt(fx**2.0+fy**2.0)))*tools.delta(tf,-drift)*(numpy.exp(-1.0j*2.0*numpy.pi*(x0*(freqs.reshape(-1,1)-fx)+y0*(freqs-fy))))-1.0/sam*scipy.special.j1(sigma*sam)*numpy.exp(1.0j*phase*(numpy.sqrt(fx**2.0+fy**2.0)))*tools.delta(tf,drift)*numpy.exp(-1.0j*2.0*numpy.pi*(x0*(freqs.reshape(-1,1)+fx)+y0*(freqs+fy))))
            self.data = numpy.fft.fftn(grat,axes=[1,2,3])
            #x0 = -1.5; y0 = 1.5
            #self.data -= -5.0j*contrast*sigma*numpy.pi**(1.0/2.0)/(4.0*(2.0*numpy.pi)**3.0)*(1.0/sa*scipy.special.j1(sigma*sa)*numpy.exp(-1.0j*phase*(numpy.sqrt(fx**2.0+fy**2.0)))*tools.delta(tf,-drift)*(numpy.exp(-1.0j*2.0*numpy.pi*(x0*(freqs.reshape(-1,1)-fx)+y0*(freqs-fy))))-1.0/sam*scipy.special.j1(sigma*sam)*numpy.exp(1.0j*phase*(numpy.sqrt(fx**2.0+fy**2.0)))*tools.delta(tf,drift)*numpy.exp(-1.0j*2.0*numpy.pi*(x0*(freqs.reshape(-1,1)+fx)+y0*(freqs+fy))))
       