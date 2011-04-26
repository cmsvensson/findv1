import sys, numpy, numpy.lib.io, tuningFunctions, pylab, Neuron, scipy

sys.path.append("../ExpoImportPython")

import one_cell
class manualFit:
    
    def __init__(self, file, args=None):
        self.filename = file
        
        #args_t = numpy.lib.io.load('../time_data_fits'+file[file.find('/'):]+'.npy')
        if args == None:
            self.args = numpy.array([0.0, 0.0, 250.0, 0.5, 1.3, 50.0, 1.5, 2.5, 0.004, 0.0052])
        else:
            self.args = numpy.asarray(args)
        
        self.expoData = one_cell.one_cell(file)  
        
    def __n_to_k(self, n):
        k = []
        if n == 1:
            return numpy.array([0])
        else:
            for i in numpy.asarray(n):
                if i== 1:
                    k.append(0.0)
                elif i == 2:
                    k.append(8.985)
                elif i == 3:
                    k.append(2.484)
                elif i == 4:
                    k.append(1.591)
                elif i == 5:
                    k.append(1.242)
                elif i == 6:
                    k.append(1.049)
                elif i == 7:
                    k.append(0.9248)
                elif i == 8:
                    k.append(0.8360)
                elif i == 9:
                    k.append(0.7687)
                else:
                    k.append(1.96/numpy.sqrt(i))
            return k #Gives 95% confidence intervals 
            #return (1.0/numpy.sqrt(i)) #Gives standard error bars
        
    def getStats(self):
        k_8 = self.__n_to_k(self.expoData.n_8)
        k_sf = self.__n_to_k(self.expoData.n_sf)
        k_size = self.__n_to_k(self.expoData.n_size)
        k_cont = self.__n_to_k(self.expoData.n_cont)
        k_tf = self.__n_to_k(self.expoData.n_tf)
        if numpy.asarray(self.expoData.sf0).any():
            sfs, sf0, sf1 = numpy.asarray(self.expoData.sfs), numpy.asarray(self.expoData.sf0), numpy.asarray(self.expoData.sf1)
            sf1tol = numpy.max([k_sf*numpy.asarray(self.expoData.sf1std), numpy.ones(numpy.asarray(self.expoData.sf1std).shape)],axis=0)
        else:
            sfs, sf0, sf1 = numpy.array([1]),numpy.array([1]),numpy.array([1])
            sf1tol = numpy.asarray([1.0])
            
        if numpy.asarray(self.expoData.tfs).any():
            tfs, tf0, tf1 = numpy.asarray(self.expoData.tfs), numpy.asarray(self.expoData.tf0), numpy.asarray(self.expoData.tf1)
            tf1tol = numpy.max([k_tf*numpy.asarray(self.expoData.tf1std), numpy.ones(numpy.asarray(self.expoData.tf1std).shape)],axis=0)
        else:
            tfs, tf0, tf1 = numpy.array([1]),numpy.array([1]),numpy.array([1]) 
            tf1tol = numpy.asarray([1.0])
            
        if numpy.asarray(self.expoData.or8).any():    
            or8, or80, or81 = numpy.asarray(self.expoData.or8), numpy.asarray(self.expoData.or80), numpy.asarray(self.expoData.or81)            
            or1tol = numpy.max([k_8*numpy.asarray(self.expoData.or81std), numpy.ones(numpy.asarray(self.expoData.or81std).shape)],axis=0)
        else:
            or8, or80, or81 = numpy.array([1]),numpy.array([1]),numpy.array([1])
            or1tol = numpy.asarray([1.0])
            
        if numpy.asarray(self.expoData.or_var).any():   
            or_var, or_var0, or_var1 = numpy.asarray(self.expoData.or_var), numpy.asarray(self.expoData.or_var0), numpy.asarray(self.expoData.or_var1)
        else:
            or_var, or_var0, or_var1 = numpy.array([1]),numpy.array([1]),numpy.array([1])
            
        if numpy.asarray(self.expoData.size).any():    
            size, sizeRes0, sizeRes1 = numpy.asarray(self.expoData.size), numpy.asarray(self.expoData.sizeRes0), numpy.asarray(self.expoData.sizeRes1)
            size1tol = numpy.max([k_size*numpy.asarray(self.expoData.size1std), numpy.ones(numpy.asarray(self.expoData.size1std).shape)],axis=0)
        else:
            size, sizeRes0, sizeRes1 = numpy.array([1]),numpy.array([1]),numpy.array([1])
            size1tol = numpy.asarray([1.0])
            
        if numpy.asarray(self.expoData.cont).any():
            contrasts, contRes0, contRes1 = numpy.asarray(self.expoData.cont), numpy.asarray(self.expoData.contRes0), numpy.asarray(self.expoData.contRes1)
            cont1tol = numpy.max([k_cont*numpy.asarray(self.expoData.cont1std), numpy.ones(numpy.asarray(self.expoData.cont1std).shape)],axis=0)
        else:
            contrasts, contRes0, contRes1 = numpy.array([1]),numpy.array([1]),numpy.array([1])
            cont1tol = numpy.asarray([1.0])
            

        modelNeur = Neuron.V1neuron(sizeDeg=3, sizePix=128, 
                                    crfWidth=self.args[3], ori = self.args[0],
                                    surroundWidth=self.args[6],
                                    surroundGain=self.args[5], normPoolExp=self.args[4],
                                    phase=self.args[1], sf = self.args[7], 
                                    crfGain=self.args[2], tau1 = self.args[8],
                                    tau2 = self.args[9])
        
        freqs = numpy.nonzero(sfs)
        sizes = numpy.nonzero(size)
        conts = numpy.nonzero(contrasts)
        tfreqs = numpy.nonzero(tfs)
        modOris, F0ori, F1ori, modSFS, F0sf, F1sf, modSize, F0size, F1size, modCont, F0cont, F1cont,  modTf, F0Tf, F1Tf = tuningFunctions.getResponses(modelNeur, self.expoData)

        
        ori1err = (abs(F1ori[0,:]-or81) <= or1tol)
        sf1err = (abs(F1sf[0,:]-sf1[freqs]) <= sf1tol[freqs] )
        size1err = (abs(F1size[0,:]-sizeRes1[sizes]) <= size1tol[sizes])
        cont1err = (abs(F1cont[0,:]-contRes1[conts]) <= cont1tol[conts])
        tf1err = (abs(F1Tf[0,:]-tf1[tfreqs]) <= tf1tol[tfreqs])
        
        ori0error = abs(F1ori[0,:]-or81)/F1ori.shape[1]
        size0error = abs(F1size[0,:]-sizeRes1[sizes])/F1size.shape[1]
        cont0error = abs(F1cont[0,:]-contRes1[conts])/F1cont.shape[1]
        sf0error = abs(F1sf[0,:]-sf1[freqs])/F1sf.shape[1]
        if numpy.asarray(self.expoData.tfs).any(): 
            tf0error = abs(F1Tf[0,:]-tf1[tfreqs])/F1Tf.shape[1]
        else:
            tf1error = numpy.array([0.0])
        ori0error = scipy.sqrt(scipy.dot(ori0error,ori0error))
        size0error = scipy.sqrt(scipy.dot(size0error,size0error))
        cont0error = scipy.sqrt(scipy.dot(cont0error,cont0error)) 
        sf0error = scipy.sqrt(scipy.dot(sf0error,sf0error))
        tf0error = scipy.sqrt(scipy.dot(tf0error,tf0error))
                
        corr = float(sum(ori1err)+sum(sf1err)+sum(size1err)+sum(cont1err)+sum(tf1err))/float(len(ori1err)+len(sf1err)+len(size1err)+len(cont1err)+len(tf1err))
        conf_or = float(sum(ori1err))/float(len(ori1err))
        conf_sf = float(sum(sf1err))/float(len(sf1err))
        conf_size = float(sum(size1err))/float(len(size1err))
        conf_cont = float(sum(cont1err))/float(len(cont1err))
        conf_tf = float(sum(tf1err))/float(len(tf1err))
        
        correlations = [numpy.corrcoef(F1ori[0,:], or81)[0,1]**2.0, numpy.corrcoef(F1sf[0,:], sf1[freqs])[0,1]**2.0, numpy.corrcoef(F1size[0,:], sizeRes1[sizes])[0,1]**2.0, numpy.corrcoef(F1cont[0,:],contRes1[conts])[0,1]**2.0, numpy.corrcoef(F1Tf[0,:], tf1)[0,1]**2.0]
        
        self.conf_int = [float(sum(ori1err))/float(len(ori1err)), float(sum(sf1err))/float(len(sf1err)), float(sum(size1err))/float(len(size1err)), float(sum(cont1err))/float(len(cont1err)), float(sum(tf1err))/float(len(tf1err))]
        
        #e1 = scipy.sqrt(scipy.dot(ori0error,ori0error))
        #e2 = scipy.sqrt(scipy.dot(sf0error,sf0error))
        #e3 = scipy.sqrt(scipy.dot(size0error,size0error))
        #e4 = scipy.sqrt(scipy.dot(cont0error,cont0error)) 
        #e5 = scipy.sqrt(scipy.dot(tf0error,tf0error))
        self.totErr = ori0error + size0error + sf0error + cont0error + tf0error
        
        #txt = ' Ori Phase \n crfGain crfWidth \n normPoolExpon surroundGain \n surroundWidth sf \n tau1 tau2 \n'
        #for param in range(5):
            #txt += '%.4f ' %self.args[2*param] 
            #txt += '%.4f ' %self.args[2*param+1]
            #txt +='\n'
        self.corrs = correlations
        return [conf_or, conf_size, conf_sf, conf_cont, conf_tf, corr], correlations, [ori0error, size0error, sf0error, cont0error, tf0error]
    
    def plotFit(self,data = 1, col = 'k', label='', sf_ticks = [0,1,2,3,4], size_ticks = [0,1,2,3], ori_ticks = [0,90,180,270], tf_ticks = [0,10,20,30], cell = 1):
        print '. \t Ori \t Sf \t Size \t tf' 
        print 'Ori \t .\t %.2f\t %.2f\t %.2f' %(self.expoData.orsf, self.expoData.orsize, self.expoData.ortf)
        print 'SF \t %.2f\t .\t %.2f\t %.2f' %(self.expoData.sfori, self.expoData.orsize, self.expoData.sftf)
        print 'Size \t %.2f\t %.2f\t .\t %.2f' %(self.expoData.sizeori, self.expoData.sizesf, self.expoData.sizetf)
        print 'Cont\t %.2f\t %.2f\t %.2f\t %.2f' %(self.expoData.contori, self.expoData.contsf, self.expoData.contsize, self.expoData.conttf)
        print 'TF \t %.2f\t %.2f\t %.2f\t    ' %(self.expoData.tfori, self.expoData.tfsf, self.expoData.tfsize)
        
        k_8 = self.__n_to_k(self.expoData.n_8)
        k_sf = self.__n_to_k(self.expoData.n_sf)
        k_size = self.__n_to_k(self.expoData.n_size)
        k_cont = self.__n_to_k(self.expoData.n_cont)
        k_tf = self.__n_to_k(self.expoData.n_tf)
        if numpy.asarray(self.expoData.sf0).any():
            sfs, sf0, sf1 = numpy.asarray(self.expoData.sfs), numpy.asarray(self.expoData.sf0), numpy.asarray(self.expoData.sf1)
            sf1tol = numpy.max([k_sf*numpy.asarray(self.expoData.sf1std), numpy.ones(numpy.asarray(self.expoData.sf1std).shape)],axis=0)
        else:
            sfs, sf0, sf1 = numpy.array([1]),numpy.array([1]),numpy.array([1])
            sf1tol = numpy.asarray([1.0])
            
        if numpy.asarray(self.expoData.tfs).any():
            tfs, tf0, tf1 = numpy.asarray(self.expoData.tfs), numpy.asarray(self.expoData.tf0), numpy.asarray(self.expoData.tf1)
            tf1tol = numpy.max([k_tf*numpy.asarray(self.expoData.tf1std), numpy.ones(numpy.asarray(self.expoData.tf1std).shape)],axis=0)
        else:
            tfs, tf0, tf1 = numpy.array([1]),numpy.array([1]),numpy.array([1]) 
            tf1tol = numpy.asarray([1.0])
            
        if numpy.asarray(self.expoData.or8).any():    
            or8, or80, or81 = numpy.asarray(self.expoData.or8), numpy.asarray(self.expoData.or80), numpy.asarray(self.expoData.or81)            
            or1tol = numpy.max([k_8*numpy.asarray(self.expoData.or81std), numpy.ones(numpy.asarray(self.expoData.or81std).shape)],axis=0)
        else:
            or8, or80, or81 = numpy.array([1]),numpy.array([1]),numpy.array([1])
            or1tol = numpy.asarray([1.0])
            
        if numpy.asarray(self.expoData.or_var).any():   
            or_var, or_var0, or_var1 = numpy.asarray(self.expoData.or_var), numpy.asarray(self.expoData.or_var0), numpy.asarray(self.expoData.or_var1)
        else:
            or_var, or_var0, or_var1 = numpy.array([1]),numpy.array([1]),numpy.array([1])
            
        if numpy.asarray(self.expoData.size).any():    
            size, sizeRes0, sizeRes1 = numpy.asarray(self.expoData.size), numpy.asarray(self.expoData.sizeRes0), numpy.asarray(self.expoData.sizeRes1)
            size1tol = numpy.max([k_size*numpy.asarray(self.expoData.size1std), numpy.ones(numpy.asarray(self.expoData.size1std).shape)],axis=0)
        else:
            size, sizeRes0, sizeRes1 = numpy.array([1]),numpy.array([1]),numpy.array([1])
            size1tol = numpy.asarray([1.0])
            
        if numpy.asarray(self.expoData.cont).any():
            contrasts, contRes0, contRes1 = numpy.asarray(self.expoData.cont), numpy.asarray(self.expoData.contRes0), numpy.asarray(self.expoData.contRes1)
            cont1tol = numpy.max([k_cont*numpy.asarray(self.expoData.cont1std), numpy.ones(numpy.asarray(self.expoData.cont1std).shape)],axis=0)
        else:
            contrasts, contRes0, contRes1 = numpy.array([1]),numpy.array([1]),numpy.array([1])
            cont1tol = numpy.asarray([1.0])
            

        modelNeur = Neuron.V1neuron(sizeDeg=3, sizePix=128, 
                                    crfWidth=self.args[3], ori = self.args[0],
                                    surroundWidth=self.args[6],
                                    surroundGain=self.args[5], normPoolExp=self.args[4],
                                    phase=self.args[1], sf = self.args[7], 
                                    crfGain=self.args[2], tau1 = self.args[8],
                                    tau2 = self.args[9])
        
        freqs = numpy.nonzero(sfs)
        sizes = numpy.nonzero(size)
        conts = numpy.nonzero(contrasts)
        tfreqs = numpy.nonzero(tfs)
        modOris, F0ori, F1ori, modSFS, F0sf, F1sf, modSize, F0size, F1size, modCont, F0cont, F1cont,  modTf, F0Tf, F1Tf = tuningFunctions.getResponses(modelNeur, self.expoData)

        
        ori1err = (abs(F1ori[0,:]-or81) <= or1tol)
        sf1err = (abs(F1sf[0,:]-sf1[freqs]) <= sf1tol[freqs] )
        size1err = (abs(F1size[0,:]-sizeRes1[sizes]) <= size1tol[sizes])
        cont1err = (abs(F1cont[0,:]-contRes1[conts]) <= cont1tol[conts])
        tf1err = (abs(F1Tf[0,:]-tf1[tfreqs]) <= tf1tol[tfreqs])
        
        corr = float(sum(ori1err)+sum(sf1err)+sum(size1err)+sum(cont1err)+sum(tf1err))/float(len(ori1err)+len(sf1err)+len(size1err)+len(cont1err)+len(tf1err))
        correlations = [numpy.corrcoef(F1ori[0,:], or81)[0,1]**2.0, numpy.corrcoef(F1sf[0,:], sf1[freqs])[0,1]**2.0, numpy.corrcoef(F1size[0,:], sizeRes1[sizes])[0,1]**2.0, numpy.corrcoef(F1cont[0,:], contRes1[conts])[0,1]**2.0, numpy.corrcoef(F1Tf[0,:], tf1)[0,1]**2.0]
        self.conf_int = [float(sum(ori1err))/float(len(ori1err)), float(sum(sf1err))/float(len(sf1err)), float(sum(size1err))/float(len(size1err)), float(sum(cont1err))/float(len(cont1err)), float(sum(tf1err))/float(len(tf1err))]
       
        ori0error = abs(F1ori[0,:]-or81)/F1ori.shape[1]
        size0error = abs(F1size[0,:]-sizeRes1[sizes])/F1size.shape[1]
        cont0error = abs(F1cont[0,:]-contRes1[conts])/F1cont.shape[1]
        sf0error = abs(F1sf[0,:]-sf1[freqs])/F1sf.shape[1]
        if numpy.asarray(self.expoData.tfs).any(): 
            tf0error = abs(F1Tf[0,:]-tf1[tfreqs])/F1Tf.shape[1]
        else:
            tf1error = numpy.array([0.0])
        
        e1 = scipy.sqrt(scipy.dot(ori0error,ori0error))
        e2 = scipy.sqrt(scipy.dot(sf0error,sf0error))
        e3 = scipy.sqrt(scipy.dot(size0error,size0error))
        e4 = scipy.sqrt(scipy.dot(cont0error,cont0error)) 
        e5 = scipy.sqrt(scipy.dot(tf0error,tf0error))
        self.totErr = e1 + e2 + e3 + e4 + e5
        print self.totErr
        print self.conf_int
        print correlations
        txt = ' Ori Phase \n crfGain crfWidth \n normPoolExpon surroundGain \n surroundWidth sf \n tau1 tau2 \n'
        for param in range(5):
            txt += '%.4f ' %self.args[2*param] 
            txt += '%.4f ' %self.args[2*param+1]
            txt +='\n'
        self.corrs = correlations

        print 'oris = %s' %modOris
        print 'ori_Data = %s' %or81
        print 'ori_Confidence_interval = %s' %or1tol
        print 'ori_model = %s' %F1ori[0,:]
        
        print 'sf = %s' %modSFS 
        print 'sf_Data = %s' %sf1[freqs]
        print 'sf_Confidence_interval = %s' %sf1tol[freqs]
        print 'sf_model = %s' %F1sf[0,:]
        
        print 'sizes = %s' %size[sizes]
        print 'size_Data = %s' %sizeRes1[sizes]
        print 'size_Confidence_interval = %s' %size1tol[sizes]
        print 'size_model = %s' %F1size[0,:]
        
        print 'contrasts = %s' %modCont
        print 'contrast_Data = %s' % contRes1[conts]
        print 'contrast_Confidence_interval = %s' %cont1tol[conts]
        print 'contrast_model = %s' %F1cont[0,:]
        
        print 'tfs = %s' %modTf
        print 'tf_Data = %s' %tf1
        print 'tf_Confidence_interval = %s' %tf1tol
        print 'tf_model = %s' %F1Tf[0,:]
        
        pylab.subplot(4,5,(cell-1)*5+1)
        pylab.plot(modOris, F1ori[0,:], col, linewidth=4, label=label)
        pylab.errorbar(modOris, or81, or1tol, fmt='ko',mfc='w', mec='k',ms=12)
        pylab.ylim(ymin=0)
        pylab.xticks(ori_ticks)
        pylab.xlabel('Orientation \n (deg)', fontsize=24)
        pylab.ylabel('Response (ips)', fontsize=24)
        fontsize=16
        ax = pylab.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            
        pylab.subplot(4,5,(cell-1)*5+2)
        pylab.plot(modSFS, F1sf[0,:], col, linewidth=4, label=label)
        pylab.errorbar(modSFS, sf1[freqs], sf1tol[freqs], fmt='ko',mfc='w', mec='k',ms=12) 
        pylab.xlabel('Spatial Frequency \n (c/deg)', fontsize=24)
        pylab.ylim(ymin=0)
        pylab.xticks(sf_ticks)
        ax = pylab.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            
        pylab.subplot(4,5,(cell-1)*5+3)
        pylab.plot(size[sizes], F1size[0,:], col, linewidth=4, label=label)
        pylab.errorbar(size[sizes], sizeRes1[sizes], size1tol[sizes], fmt='ko',mfc='w', mec='k',ms=12)
        pylab.xticks(size_ticks)
        pylab.xlabel('Stimulus Diameter \n (deg vis angle)', fontsize=24)
        pylab.ylim(ymin=0)
        ax = pylab.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            
        pylab.subplot(4,5,(cell-1)*5+4)
        pylab.semilogx(modCont, F1cont[0,:], col, linewidth=3, label=label)
        pylab.errorbar(modCont, contRes1[conts], cont1tol[conts], fmt='ko',mfc='w', mec='k',ms=12)
        pylab.ylim(ymin=0, fontsize=16)
        pylab.xlim(xmax=1.1, fontsize=16)
        pylab.xlabel('Stimulus Contrast \n (Michelson)', fontsize=24)
        ax = pylab.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            
        pylab.subplot(4,5,(cell-1)*5+5)
        l=pylab.plot(modTf, F1Tf[0,:], col, linewidth=4)
        pylab.errorbar(modTf, tf1, tf1tol, fmt='ko',mfc='w', mec='k',ms=12)
        pylab.xlabel('Temporal Frequency \n (Hz)', fontsize=24)
        pylab.xticks(tf_ticks)
        ax = pylab.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        pylab.ylim(ymin=0)
        

        if data:    
            pylab.figure(2)
            pylab.subplot(2,3,1)
            pylab.plot(modOris, or80, 'bo-')
            pylab.plot(modOris, or81, 'ro-')
            pylab.title('Orientation, experiment')
            pylab.ylim(ymin=0)
            pylab.subplot(2,3,2)
            pylab.plot(modSFS, sf0[freqs], 'bo-')
            pylab.plot(modSFS, sf1[freqs], 'ro-')
            pylab.title('Spatial frequency, experiment') 
            pylab.ylim(ymin=0)
            pylab.subplot(2,3,3)
            pylab.plot(modSize, sizeRes0[sizes], 'bo-')
            pylab.plot(modSize, sizeRes1[sizes], 'ro-')
            pylab.title('Size, experiment') 
            pylab.ylim(ymin=0)
            pylab.subplot(2,3,4)
            pylab.plot(modCont, contRes0[conts], 'bo-')
            pylab.plot(modCont, contRes1[conts], 'ro-')
            pylab.title('Contrast, experiment') 
            pylab.ylim(ymin=0)
            pylab.subplot(2,3,5)
            pylab.plot(tfs, tf0, 'bo-')
            pylab.plot(tfs, tf1, 'ro-')
            pylab.title('Temporal frequency, experiment') 
            pylab.ylim(ymin=0)
            
            pylab.figure(3)
            pylab.subplot(2,3,1)           
            pylab.plot(modOris, F1ori[0,:], 'r')
            pylab.title('Orientation, model')
            pylab.ylim(ymin=0)
            pylab.subplot(2,3,2)
            pylab.plot(modSFS, F1sf[0,:], 'r')
            pylab.ylim(ymin=0)
            pylab.subplot(2,3,3)
            pylab.plot(modSize, F1size[0,:], 'r')
            pylab.ylim(ymin=0)
            pylab.subplot(2,3,4)
            pylab.plot(modCont, F1cont[0,:], 'r')
            pylab.title('Contrast, model') 
            pylab.ylim(ymin=0)
            pylab.xlim(xmax=1.1)
            pylab.subplot(2,3,5)
            pylab.plot(modTf, F1Tf[0,:], 'r')
            pylab.title('Temporal frequency, model') 
            pylab.ylim(ymin=0)
        return l
   
        