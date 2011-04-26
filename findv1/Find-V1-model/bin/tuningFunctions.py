import Neuron, numpy, tools

def contrastTuning(neuron, conts = None, stimSize_in=2, sf_in=None, ori_in=None, drift=5.0):
    
    if conts==None or conts==[]:
        conts = numpy.array([0.0005, 0.03, 0.07, 0.125, 0.25, 0.5, 1.0])#numpy.linspace(0.0,1.0,10)
    assert isinstance(neuron, Neuron.V1neuron)#helps WingIDE know what the neuron is
    if ori_in == None:
        ori = neuron.ori
    else:
        ori = ori_in
        
    if sf_in == None:    
        sf=neuron.sf
    else:
        sf=sf_in    
    if stimSize_in == None:    
        stimSize=neuron.crfWidth/1.2
    else:
        stimSize=stimSize_in 
    
    allPSTH, allF0, allF1 =[], [], []
    
    stim = Neuron.Stimulus(gratType = "sin", cycles = neuron.sizeDeg*sf, ori = ori, driftrate = drift, diameter = stimSize_in, phase=0.0, contr = conts)
    
    response = neuron.getResponse(stim)
    
    allPSTH.append(response)
    allF0.append(numpy.mean(response,axis=1))
    f1anal = tools.getFourierPower(response,freq=drift)
    allF1.append(f1anal[0])
    #convert to num arrays for output
    arr=numpy.asarray
    contrasts, allPSTH, allF0, allF1 = arr(conts), arr(allPSTH), arr(allF0), arr(allF1)
        
    return contrasts, allPSTH, allF0, allF1

def oriTuning(neuron, oris = None, stimSize_in=2, sf_in=None, cont_in=None, drift=5.0):
    
    if oris==None:
        oris = numpy.linspace(0.0,360.0,10)
    assert isinstance(neuron, Neuron.V1neuron)#helps WingIDE know what the neuron is
    if cont_in == None:
        cont = 1.0
    else:
        cont = cont_in
        
    if sf_in == None:    
        sf=neuron.sf
    else:
        sf=sf_in    
    if stimSize_in == None:    
        stimSize=neuron.crfWidth/1.2
    else:
        stimSize=stimSize_in 
    
    allPSTH, allF0, allF1 =[], [], []
    
    stim = Neuron.Stimulus(gratType = "sin", cycles = neuron.sizeDeg*sf, ori = oris, driftrate = drift, diameter = stimSize_in, phase=0.0, contr = cont)
    
    response = neuron.getResponse(stim)
    
    allPSTH.append(response)
    allF0.append(numpy.mean(response,axis=1))
    f1anal = tools.getFourierPower(response,freq=drift)
    allF1.append(f1anal[0])
    #convert to num arrays for output
    arr=numpy.asarray
    oris, allPSTH, allF0, allF1 = arr(oris), arr(allPSTH), arr(allF0), arr(allF1)
        
    return oris, allPSTH, allF0, allF1

def tfTuning(neuron, ori_in = None, stimSize_in=2, sf_in=None, cont_in=None, drift=None):
    
    if drift==None:
        drift = numpy.linspace(0.01,25.0,9)
    assert isinstance(neuron, Neuron.V1neuron)#helps WingIDE know what the neuron is
    if cont_in == None:
        cont = 1.0
    else:
        cont = cont_in
        
    if sf_in == None:    
        sf=neuron.sf
    else:
        sf=sf_in    
    if stimSize_in == None:    
        stimSize=neuron.crfWidth/1.2
    else:
        stimSize=stimSize_in 
    if ori_in == None:    
        ori=neuron.ori
    else:
        ori=ori_in 
    
    allPSTH, allF0, allF1 =[], [], []
    
    stim = Neuron.Stimulus(gratType = "sin", cycles = neuron.sizeDeg*sf, ori = ori, driftrate = drift, diameter = stimSize_in, phase=0.0, contr = cont)
    
    response = neuron.getResponse(stim)
    
    allPSTH.append(response)
    allF0.append(numpy.mean(response,axis=1))
    f1anal = tools.getFourierPower(response,drift)
    allF1.append(f1anal[0])
    #convert to num arrays for output
    arr=numpy.asarray
    tfs, allPSTH, allF0, allF1 = arr(drift), arr(allPSTH), arr(allF0), arr(allF1)
        
    return tfs, allPSTH, allF0, allF1

def sizeTuning(neuron, ori_in = None, stimSize=None, sf_in=None, cont_in=None, drift=5.0):
    
    if stimSize==None:
        stimSize = numpy.linspace(0.1,3.0,9)
    assert isinstance(neuron, Neuron.V1neuron)#helps WingIDE know what the neuron is
    if cont_in == None:
        cont = 1.0
    else:
        cont = cont_in
        
    if sf_in == None:    
        sf=neuron.sf
    else:
        sf=sf_in    
    if ori_in == None:    
        ori=neuron.ori
    else:
        ori=ori_in 
    
    allPSTH, allF0, allF1 =[], [], []
    
    stim = Neuron.Stimulus(gratType = "sin", cycles = neuron.sizeDeg*sf, ori = ori, driftrate = drift, diameter = stimSize, phase=0.0, contr = cont)
    
    response = neuron.getResponse(stim)
    
    allPSTH.append(response)
    allF0.append(numpy.mean(response,axis=1))
    f1anal = tools.getFourierPower(response,drift)
    allF1.append(f1anal[0])
    #convert to num arrays for output
    arr=numpy.asarray
    sizes, allPSTH, allF0, allF1 = arr(stimSize), arr(allPSTH), arr(allF0), arr(allF1)
        
    return sizes, allPSTH, allF0, allF1

def sfTuning(neuron, ori_in = None, stimSize_in=None, sf=None, cont_in=None, drift=5.0):
    
    if sf==None:
        sf = numpy.linspace(0.1,5.0,10)
    assert isinstance(neuron, Neuron.V1neuron)#helps WingIDE know what the neuron is
    if cont_in == None:
        cont = 1.0
    else:
        cont = cont_in
        
    if stimSize_in == None:    
        stimSize=neuron.crfWidth/1.2
    else:
        stimSize = stimSize_in    
    if ori_in == None:    
        ori=neuron.ori
    else:
        ori=ori_in 
    
    allPSTH, allF0, allF1 =[], [], []
    
    stim = Neuron.Stimulus(gratType = "sin", cycles = neuron.sizeDeg*sf, ori = ori, driftrate = drift, diameter = stimSize, phase=0.0, contr = cont)
    
    response = neuron.getResponse(stim)
    
    allPSTH.append(response)
    allF0.append(numpy.mean(response,axis=1))
    f1anal = tools.getFourierPower(response,drift)
    allF1.append(f1anal[0])
    #convert to num arrays for output
    arr=numpy.asarray
    sfs, allPSTH, allF0, allF1 = arr(sf), arr(allPSTH), arr(allF0), arr(allF1)
        
    return sfs, allPSTH, allF0, allF1

def getResponses(modelNeur, expoData, ori_t = True, sf_t = True, size_t = True, 
                 cont_t =True, tf_t = True):
    size_factor = 1.0#/3.0
    sf_factor = 1.0#/size_factor
    if ori_t:
        if numpy.asarray(expoData.or8).any():
            oris = numpy.asarray(expoData.or8)
        else:
            oris = numpy.array([1])
        modOris, modPSTHOri, F0ori, F1ori = oriTuning(modelNeur,
                                                stimSize_in=expoData.orsize*size_factor, 
                                                oris=oris, 
                                                sf_in=expoData.orsf, 
                                                drift=expoData.ortf)
    if sf_t:
        if numpy.asarray(expoData.sf0).any():
            sfs = numpy.asarray(expoData.sfs)
        else:
            sfs = numpy.array([1])
        freqs = numpy.nonzero(sfs)
        modSFS, modPSTHSf, F0sf, F1sf = sfTuning(modelNeur,
                                                stimSize_in=expoData.sfsize*size_factor,
                                                sf=sfs[freqs], 
                                                #ori_in = expoData.sfori, 
                                                drift = expoData.sftf)
    else:
        modSFS, F0sf, F1sf = [], [], []
        
    if size_t:
        if numpy.asarray(expoData.size).any():    
            size = numpy.asarray(expoData.size)
        else:
            size = numpy.array([1])
        sizes = numpy.nonzero(size)  
        modSize, modPSTHSize, F0size, F1size = sizeTuning(modelNeur,
                                                stimSize=size[sizes]*size_factor, 
                                                sf_in=expoData.sizesf, 
                                                #ori_in=expoData.sizeori, 
                                                drift=expoData.sizetf)
    else:
        modSize, F0size, F1size = [], [], []
        
    if cont_t:
        if numpy.asarray(expoData.cont).any():
            contrasts = numpy.asarray(expoData.cont)
        else:
            contrasts = numpy.array([1])
        conts = numpy.nonzero(contrasts)  
        modCont, allPSTHCont, F0cont, F1cont = contrastTuning(modelNeur, 
                                                stimSize_in=expoData.contsize*size_factor,
                                                conts=contrasts[conts],
                                                #ori_in=expoData.contori, 
                                                sf_in=expoData.contsf, 
                                                drift=expoData.conttf)
    else:
        modCont, F0cont, F1cont = [], [], []
    if tf_t:
        if numpy.asarray(expoData.tfs).any():
            tfs = numpy.asarray(expoData.tfs)
        else:
            tfs = numpy.array([1])
        tfreqs = numpy.nonzero(tfs)
        modTf, allPSTHTf, F0Tf, F1Tf = tfTuning(modelNeur, 
                                                stimSize_in=expoData.tfsize*size_factor,
                                                drift=tfs[tfreqs],
                                                #ori_in=expoData.tfori, 
                                                sf_in=expoData.tfsf)
    else:
        modTf, F0Tf, F1Tf = [], [], []
    return modOris, F0ori, F1ori, modSFS, F0sf, F1sf, modSize, F0size, F1size,  modCont,  F0cont, F1cont, modTf, F0Tf, F1Tf






