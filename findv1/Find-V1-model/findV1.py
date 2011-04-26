import sys
sys.path.append(r'bin')
import wx, os, pylab, numpy, Neuron, tuningFunctions, tools, simulateData


class findV1(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(300,200), style=wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE)
        
        self.CreateStatusBar() # A StatusBar in the bottom of the window
              
        # Setting up the menu.
        filemenu= wx.Menu()
        p = wx.Panel(self)
       
        modelButton = wx.Button(p,wx.ID_ANY,"Test Model", pos = (20,20), size = (100,40))
        fitButton = wx.Button(p,wx.ID_ANY,"Fit the Model", pos = (20,100), size = (100,40))
        quitButton = wx.Button(p,wx.ID_ANY,"Quit", pos = (150,20), size = (100,40))
        
        
        menuAbout = filemenu.Append(wx.ID_ANY, "&About"," Information about this program")
        menuExit = filemenu.Append(wx.ID_ANY,"E&xit"," Terminate the program")

        
        modelmenu = wx.Menu()
        runModel = modelmenu.Append(wx.ID_ANY, "&Run Model", "A version of the model to play around with")
        fitModel = modelmenu.Append(wx.ID_ANY, "&Fit Model", "Fit the model to data")
 
        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu,"&File") # Adding the "filemenu" to the MenuBar
        menuBar.Append(modelmenu, "&Run Model")
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
 
        # Set events.
        self.Bind(wx.EVT_MENU, self.OnAbout, menuAbout)
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)
        self.Bind(wx.EVT_MENU, self.OnFit, fitModel)
        self.Bind(wx.EVT_MENU, self.OnRun, runModel)
        
        self.Bind(wx.EVT_BUTTON, self.OnFit, fitButton)
        self.Bind(wx.EVT_BUTTON, self.OnRun, modelButton)
        self.Bind(wx.EVT_BUTTON, self.OnExit, quitButton)
        self.Show(True)
        
    def OnGA(self,e):
        f = wx.Frame(self, size = (200,200))
        p = self.gaPanel(f)
        f.Show()
        
    def OnAbout(self,e):       
        dlg = wx.MessageDialog( self, "A small text editor", "About Sample Editor", wx.OK)
        dlg.ShowModal() # Show it
        dlg.Destroy() # finally destroy it when finished.
 
    def OnLoad(self,e):
        f = wx.Frame(self, size = (200,200))
        p = self.expoPanel(f)
        f.Show()
    
    def OnFit(self,e):
        f = wx.Frame(self, size = (450,400))
        p = self.fittingPanel(f)
        f.Show()
        
    def OnRun(self,e):
        f = wx.Frame(self, size = (450,500))
        p = self.modelPanel(f)
        f.Show()   
        
    def OnExit(self,e):
        self.Close(True)  # Close the frame.
            
    class expoPanel(wx.Panel):
        def __init__(self, parent):
            self.expoFile = ''
            wx.Panel.__init__(self,parent)
            self.quote = wx.StaticText(self, label="Load a file :", pos=(20, 30))
            
            self.radioList = ['m177/m177ab','m177/m177ac','m177/m177ad','m177/m177ae',
                              'm177/m177ag','m177/m177am','m177/m177b','m177/m177c',
                              'm177/m177d','m177/m177e','m177/m177f','m177/m177g',
                              'm177/m177j','m177/m177m','m177/m177n','m177/m177o',
                              'm177/m177p','m177/m177q','m177/m177r','m177/m177u',
                              'm177/m177w','m177/m177x','m177/m177z']
            rb = wx.ComboBox(self, -1, pos=(20, 100), 
                             choices=self.radioList, 
                             style=wx.CB_DROPDOWN)
            self.Bind(wx.EVT_COMBOBOX, self.EvtComboBox, rb)
            
            loadButton = wx.Button(self,wx.ID_ANY,"Load File", pos = (20,130))
            self.Bind(wx.EVT_BUTTON, self.EvtLoadButton, loadButton)
    
        def EvtComboBox(self, event):
            self.expoFile = self.radioList[event.GetSelection()]
            
        def EvtLoadButton(self, event):
            plotExpo.plotExpo(self.expoFile)
            
    class modelPanel(wx.Panel):
        def __init__(self, parent):
            self.params = numpy.array([0.0, 0.0, 100.0, 0.5, 1.1, 1.0, 5.0, 2.0, 0.004, 0.0053])
            self.oricurve = False
            self.sfcurve = False
            self.sizecurve = False
            self.contcurve = False
            self.tfcurve = False
            wx.Panel.__init__(self,parent)
            
            loadButton = wx.Button(self,wx.ID_ANY,"Run Model", pos = (20,20))
            self.Bind(wx.EVT_BUTTON, self.EvtLoadButton, loadButton)
            wx.StaticText(self,-1,pos=(20,90),label="Orientation:")
            ori = wx.TextCtrl(self, -1, pos = (300,90), value = str(self.params[0]))
            wx.StaticText(self,-1,pos=(20,120),label="Gain of the CRF:")
            crfG = wx.TextCtrl(self, -1, pos = (300,120), value = str(self.params[2]))
            wx.StaticText(self,-1,pos=(20,150),label="Size of the CRF:")
            crfW = wx.TextCtrl(self, -1, pos = (300,150), value = str(self.params[3]))
            wx.StaticText(self,-1,pos=(20,180),label="Gain of the Surround:")
            surrG = wx.TextCtrl(self, -1, pos = (300,180), value = str(self.params[5]))
            wx.StaticText(self,-1,pos=(20,210),label="Size of the Surround:")
            surrW = wx.TextCtrl(self, -1, pos = (300,210), value = str(self.params[6]))
            wx.StaticText(self,-1,pos=(20,240),label="Normalisation pool exponent:")
            NPE = wx.TextCtrl(self, -1, pos = (300,240), value = str(self.params[4]))
            wx.StaticText(self,-1,pos=(20,270),label="Spatial frequency:")
            sf = wx.TextCtrl(self, -1, pos = (300,270), value = str(self.params[7]))
            wx.StaticText(self,-1,pos=(20,300),label="First time constant of the temporal filter:")
            tau1 = wx.TextCtrl(self, -1, pos = (300,300), value = str(self.params[8]))
            wx.StaticText(self,-1,pos=(20,330),label="Second time constant of the temporal filter:")
            tau2 = wx.TextCtrl(self, -1, pos = (300,330), value = str(self.params[9]))
            crfButton = wx.Button(self,wx.ID_ANY,"View CRF", pos = (20,400))
            
            self.Bind(wx.EVT_BUTTON, self.EvtCrfButton, crfButton)
            self.allBox = wx.CheckBox(self, -1 ,'Plot all', (150, 20))
            self.oriBox = wx.CheckBox(self, -1 ,'Orientation tuning', (20, 40))
            self.sfBox = wx.CheckBox(self, -1 ,'Sf tuning', (175, 40))
            self.sizeBox = wx.CheckBox(self, -1 ,'Size tuning', (275, 40))
            self.contBox = wx.CheckBox(self, -1 ,'Contrast tuning', (20, 60))
            self.tfBox = wx.CheckBox(self, -1 ,'Tf tuning', (175, 60))
            self.Bind(wx.EVT_TEXT, self.setOri, ori)
            self.Bind(wx.EVT_TEXT, self.setCrfG, crfG)
            self.Bind(wx.EVT_TEXT, self.setCrfW, crfW)
            self.Bind(wx.EVT_TEXT, self.setSurrG, surrG)
            self.Bind(wx.EVT_TEXT, self.setSurrW, surrW)
            self.Bind(wx.EVT_TEXT, self.setNPE, NPE)
            self.Bind(wx.EVT_TEXT, self.setSf, sf)
            self.Bind(wx.EVT_TEXT, self.setTau1, tau1)
            self.Bind(wx.EVT_TEXT, self.setTau2, tau2)
            
            self.Bind(wx.EVT_CHECKBOX, self.checkAll, self.allBox)
            self.Bind(wx.EVT_CHECKBOX, self.checkOri, self.oriBox)
            self.Bind(wx.EVT_CHECKBOX, self.checkSf, self.sfBox)
            self.Bind(wx.EVT_CHECKBOX, self.checkSize, self.sizeBox)
            self.Bind(wx.EVT_CHECKBOX, self.checkCont, self.contBox)
            self.Bind(wx.EVT_CHECKBOX, self.checkTf, self.tfBox)
            
        def EvtCrfButton(self, e):
            modelNeur = Neuron.V1neuron(sizeDeg=3, sizePix=128, 
                                    crfWidth=self.params[3], ori = self.params[0],
                                    surroundWidth=self.params[6],
                                    surroundGain=self.params[5], normPoolExp=self.params[4],
                                    phase=self.params[1], sf = self.params[7], 
                                    crfGain=self.params[2], tau1 = self.params[8],
                                    tau2 = self.params[9]) 
            

            C = numpy.real(numpy.fft.ifft2(modelNeur.crfPrimaryFilter,axes=[1,2]))
            pylab.show()

            p = pylab.pcolor(numpy.linspace(-1.5,1.5,128),numpy.linspace(-1.5,1.5,128),C[0],vmin=C.min(),vmax=C.max())

            for i in range(1,len(C)/2):
                p.set_array(C[i,0:-1,0:-1].ravel())
                pylab.draw()
            
        def checkAll(self,e):
            if e.IsChecked():
                self.oriBox.SetValue(state = True)
                self.sfBox.SetValue(state = True)
                self.sizeBox.SetValue(state = True)
                self.contBox.SetValue(state = True)
                self.tfBox.SetValue(state = True)
                self.oricurve = True
                self.sfcurve = True
                self.sizecurve = True
                self.contcurve = True
                self.tfcurve = True
            else:
                self.oriBox.SetValue(state = False)
                self.sfBox.SetValue(state = False)
                self.sizeBox.SetValue(state = False)
                self.contBox.SetValue(state = False)
                self.tfBox.SetValue(state = False)
                self.oricurve = False
                self.sfcurve = False
                self.sizecurve = False
                self.contcurve = False
                self.tfcurve = False
                
        def checkOri(self,e):
            self.oricurve = e.IsChecked()
            
        def checkSf(self,e):
            self.sfcurve = e.IsChecked()
            
        def checkSize(self,e):
            self.sizecurve = e.IsChecked() 
            
        def checkCont(self,e):
            self.contcurve = e.IsChecked() 
            
        def checkTf(self,e):
            self.tfcurve = e.IsChecked()
            
        def setOri(self,e):
            self.params[0] = float(e.GetString())
            
        def setCrfG(self,e):
            self.params[2] = float(e.GetString())
            
        def setCrfW(self,e):
            self.params[3] = float(e.GetString())
            
        def setSurrG(self,e):
            self.params[5] = float(e.GetString())
            
        def setSurrW(self,e):
            self.params[6] = float(e.GetString())
            
        def setNPE(self,e):
            self.params[4] = float(e.GetString())
            
        def setSf(self,e):
            self.params[7] = float(e.GetString())
            
        def setTau1(self,e):
            self.params[8] = float(e.GetString())
            
        def setTau2(self,e):
            self.params[9] = float(e.GetString())
            
        def EvtComboBox(self, event):
            self.expoFile = self.radioList[event.GetSelection()]
            
        def EvtLoadButton(self, event):
            modelNeur = Neuron.V1neuron(sizeDeg=3, sizePix=128, 
                                    crfWidth=self.params[3], ori = self.params[0],
                                    surroundWidth=self.params[6],
                                    surroundGain=self.params[5], normPoolExp=self.params[4],
                                    phase=self.params[1], sf = self.params[7], 
                                    crfGain=self.params[2], tau1 = self.params[8],
                                    tau2 = self.params[9]) 
            
            pylab.figure(1,figsize=(21,7))
            
            if self.oricurve:
                modOris, P, F0ori, F1ori = tuningFunctions.oriTuning(modelNeur)
                pylab.subplot(1,5,1)           
                pylab.plot(modOris, F1ori[0,:], 'k', lw = 2)
                pylab.title('Orientation')
                pylab.xticks([0, 90, 180, 270])
                pylab.ylim(ymin=0)
            if self.sfcurve:  
                modSFS, P, F0sf, F1sf = tuningFunctions.sfTuning(modelNeur)
                pylab.subplot(1,5,2)
                pylab.plot(modSFS, F1sf[0,:], 'k', lw = 2)
                pylab.title('Spatial frequency')
                pylab.ylim(ymin=0)
            if self.sizecurve: 
                modSize, P, F0size, F1size = tuningFunctions.sizeTuning(modelNeur)     
                pylab.subplot(1,5,3)
                pylab.plot(modSize, F1size[0,:], 'k', lw = 2)
                pylab.title('Size')
                pylab.ylim(ymin=0)
            if self.contcurve: 
                modCont, P, F0cont, F1cont = tuningFunctions.contrastTuning(modelNeur)
                pylab.subplot(1,5,4)
                pylab.semilogx(modCont, F1cont[0,:], 'k', lw = 2)
                pylab.title('Contrast') 
                pylab.ylim(ymin=0)
                pylab.xticks([0.01, 0.1, 1.0])
            if self.tfcurve:     
                modTf, P, F0Tf, F1Tf = tuningFunctions.tfTuning(modelNeur)
                pylab.subplot(1,5,5)
                pylab.plot(modTf, F1Tf[0,:], 'k', lw = 2)
                pylab.title('Temporal frequency') 
                pylab.ylim(ymin=0)
            
            pylab.show()
            
    class gaPanel(expoPanel):
        def EvtLoadButton(self, event):
            SPEA_retrive.SPEA_retrive(self.expoFile)  
            
    class fittingPanel(wx.Panel):
        def __init__(self, parent):
            self.params = numpy.array([0.0,0.0, 100.0, 0.5, 1.1, 1.0, 5.0, 1.0, 0.004, 0.0053])
            self.simParams = numpy.array([1.05343039e+02, 9.76213328e+01, 1.14415904e+00, 1.17027548e+00, 3.46819754e+00, 2.95904032e-01, 1.99332238e+00, 3.13554779e-03, 8.91559749e-03])
            self.setDefault()
            self.hints = 0
            
            wx.Panel.__init__(self,parent)
            simButton = wx.Button(self,wx.ID_ANY,"Simulate data", pos = (20,20))
            plotDataButton = wx.Button(self,wx.ID_ANY,"Plot simulated tuning", pos = (150,20))
            plotFitButton = wx.Button(self,wx.ID_ANY,"Plot fit", pos = (330,20))
            
            hintButton =  wx.Button(self,wx.ID_ANY,"Show me a parameter", pos = (20,330))
            answerButton =  wx.Button(self,wx.ID_ANY,"Show me the answer", pos = (250,330))
            
            self.Bind(wx.EVT_BUTTON, self.EvtSimButton, simButton)
            self.Bind(wx.EVT_BUTTON, self.EvtPlotSimButton, plotDataButton)
            self.Bind(wx.EVT_BUTTON, self.EvtPlotFitButton, plotFitButton)
            self.Bind(wx.EVT_BUTTON, self.EvtHintButton, hintButton)
            self.Bind(wx.EVT_BUTTON, self.EvtAnswerButton, answerButton)
            
            wx.StaticText(self,-1,pos=(20,60),label="Orientation:")
            ori = wx.TextCtrl(self, -1, pos = (300,60), value = str(self.params[0]))
            wx.StaticText(self,-1,pos=(20,90),label="Gain of the CRF:")
            crfG = wx.TextCtrl(self, -1, pos = (300,90), value = str(self.params[2]))
            wx.StaticText(self,-1,pos=(20,120),label="Size of the CRF:")
            crfW = wx.TextCtrl(self, -1, pos = (300,120), value = str(self.params[3]))
            wx.StaticText(self,-1,pos=(20,150),label="Gain of the Surround:")
            surrG = wx.TextCtrl(self, -1, pos = (300,150), value = str(self.params[5]))
            wx.StaticText(self,-1,pos=(20,180),label="Size of the Surround:")
            surrW = wx.TextCtrl(self, -1, pos = (300,180), value = str(self.params[6]))
            wx.StaticText(self,-1,pos=(20,210),label="Normalisation pool exponent:")
            NPE = wx.TextCtrl(self, -1, pos = (300,210), value = str(self.params[4]))
            wx.StaticText(self,-1,pos=(20,240),label="Spatial frequency:")
            sf = wx.TextCtrl(self, -1, pos = (300,240), value = str(self.params[7]))
            wx.StaticText(self,-1,pos=(20,270),label="First time constant of the temporal filter:")
            tau1 = wx.TextCtrl(self, -1, pos = (300,270), value = str(self.params[8]))
            wx.StaticText(self,-1,pos=(20,300),label="Second time constant of the temporal filter:")
            tau2 = wx.TextCtrl(self, -1, pos = (300,300), value = str(self.params[9]))
            
            self.Bind(wx.EVT_TEXT, self.setOri, ori)
            self.Bind(wx.EVT_TEXT, self.setCrfG, crfG)
            self.Bind(wx.EVT_TEXT, self.setCrfW, crfW)
            self.Bind(wx.EVT_TEXT, self.setSurrG, surrG)
            self.Bind(wx.EVT_TEXT, self.setSurrW, surrW)
            self.Bind(wx.EVT_TEXT, self.setNPE, NPE)
            self.Bind(wx.EVT_TEXT, self.setSf, sf)
            self.Bind(wx.EVT_TEXT, self.setTau1, tau1)
            self.Bind(wx.EVT_TEXT, self.setTau2, tau2)
        
        def setDefault(self):
            self.modOris = numpy.array([ 0.,40.,80.,120.,160.,200.,240.,280.,320.,360.])
            self.F1ori = numpy.array([[1.02846992e-03,1.46457027e-03,1.21368653e+00,6.00252913e+00,7.12598679e-03,5.10922188e-04,1.92160428e-03,2.20473281e+00,2.16186657e-02,1.02846992e-03]])
            self.modSFS = numpy.array([0.1, 0.64444444, 1.18888889, 1.73333333, 2.27777778, 2.82222222, 3.36666667, 3.91111111, 4.45555556,5.])
            self.F1sf = numpy.array([[0.07829737, 0.60486854, 2.15158881, 3.53118444, 3.37018641, 2.14168597, 0.72962723, 0.06288583, 0.03441861, 0.08202204]])
            self.modSize = numpy.array([0.1, 0.4625, 0.825, 1.1875, 1.55, 1.9125, 2.275, 2.6375, 3.])
            self.F1size = numpy.array([[0.01642062, 0.66102566, 2.42317178, 5.62242689, 8.72942597, 11.00279346, 12.42118782, 12.90178358, 12.96191376]])
            self.modCont = numpy.array([5.00000000e-04, 3.00000000e-02, 7.00000000e-02, 1.25000000e-01, 2.50000000e-01, 5.00000000e-01, 1.00000000e+00])
            self.F1cont = numpy.array([[ 0.03698687, 1.06298268, 2.0464756, 3.12305085, 4.98359496, 7.62979167, 11.44100442]])
            self.modTf = numpy.array([1.00000000e-02, 3.13375000e+00, 6.25750000e+00, 9.38125000e+00, 1.25050000e+01, 1.56287500e+01, 1.87525000e+01, 2.18762500e+01, 2.50000000e+01])
            self.F1Tf  = numpy.array([[0.24777051, 7.06629939, 9.41203233, 5.70825721, 3.65618796, 2.93503845, 2.61555329, 2.03054383, 1.15735793]])

        #Parameter boxes
        def setOri(self,e):
            try:
                self.params[0] = float(e.GetString())
            except:
                print "Could not read Orientation"    
            
        def setCrfG(self,e):
            try:
                self.params[2] = float(e.GetString())
            except:
                print "Could not read CrfG"
        def setCrfW(self,e):
            try:
                self.params[3] = float(e.GetString())
            except:
                print "Could not read CrfW"
            
        def setSurrG(self,e):
            try:
                self.params[5] = float(e.GetString())
            except:
                print "Could not read SurrG"
        def setSurrW(self,e):
            try:
                self.params[6] = float(e.GetString())
            except:
                print "Could not read SurrW"
        def setNPE(self,e):
            try:
                self.params[4] = float(e.GetString())
            except:
                print "Could not read NPE"
        def setSf(self,e):
            try:
                self.params[7] = float(e.GetString())
            except:
                print "Could not read spatial frequency"
        def setTau1(self,e):
            try:
                self.params[8] = float(e.GetString())
            except:
                print "Could not read tau1"
        def setTau2(self,e):
            try:
                self.params[9] = float(e.GetString())
            except:
                print "Could not read tau2"
        #Button actions
        def EvtSimButton(self, event):
            dat = simulateData.simulateData()
            self.simParams = dat.getParams()
            self.hints = 0
            self.modOris, self.F1ori, self.modSFS, self.F1sf, self.modSize, self.F1size, self.modCont, self.F1cont, self.modTf, self.F1Tf = dat.getResponses()
       
        def EvtPlotSimButton(self, event):
            pylab.figure(1,figsize=(21,7))
            
            pylab.subplot(1,5,1)           
            pylab.plot(self.modOris, self.F1ori[0,:], 'k', lw = 2)
            pylab.title('Orientation')
            pylab.xticks([0, 90, 180, 270])
            pylab.ylim(ymin=0)
            
            pylab.subplot(1,5,2)
            pylab.plot(self.modSFS, self.F1sf[0,:], 'k', lw = 2)
            pylab.title('Spatial frequency')
            pylab.ylim(ymin=0)
          
            pylab.subplot(1,5,3)
            pylab.plot(self.modSize, self.F1size[0,:], 'k', lw = 2)
            pylab.title('Size')
            pylab.ylim(ymin=0)
           
            pylab.subplot(1,5,4)
            pylab.semilogx(self.modCont, self.F1cont[0,:], 'k', lw = 2)
            pylab.title('Contrast') 
            pylab.ylim(ymin=0)
            pylab.xticks([0.01, 0.1, 1.0])
            
            pylab.subplot(1,5,5)
            pylab.plot(self.modTf, self.F1Tf[0,:], 'k', lw = 2)
            pylab.title('Temporal frequency') 
            pylab.ylim(ymin=0)
            
            pylab.show()
        def EvtPlotFitButton(self, event):
            pylab.close()
            dialog = wx.ProgressDialog( 'Progress', 'Computing Responses', maximum = 7, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME )
            modelNeur = Neuron.V1neuron(sizeDeg=3, sizePix=128, 
                                    crfWidth=self.params[3], ori = self.params[0],
                                    surroundWidth=self.params[6],
                                    surroundGain=self.params[5], normPoolExp=self.params[4],
                                    phase=self.params[1], sf = self.params[7], 
                                    crfGain=self.params[2], tau1 = self.params[8],
                                    tau2 = self.params[9]) 
            dialog.Update ( 1, 'Computing Orientation')
            guessOris, P, F0ori, guessF1ori = tuningFunctions.oriTuning(modelNeur)
            dialog.Update ( 2, 'Computing Spatial Frequency')
            guessSFS, P, F0sf, guessF1sf = tuningFunctions.sfTuning(modelNeur)
            dialog.Update ( 3, 'Computing Size tuning')
            guessSize, P, F0size, guessF1size = tuningFunctions.sizeTuning(modelNeur)
            dialog.Update ( 4, 'Computing Contrast tuning')
            guessCont, P, F0cont, guessF1cont = tuningFunctions.contrastTuning(modelNeur)
            dialog.Update ( 5, 'Computing Temporal Frequency')
            guessTf, P, F0Tf, guessF1Tf = tuningFunctions.tfTuning(modelNeur)
            
            dialog.Update ( 6, 'Plotting')
            pylab.figure(1,figsize=(21,7))
            
            pylab.subplot(1,5,1)           
            pylab.plot(self.modOris, self.F1ori[0,:], 'k', lw = 2)
            pylab.plot(guessOris, guessF1ori[0,:], 'k--', lw = 2)
            pylab.title('Orientation')
            pylab.xticks([0, 90, 180, 270])
            pylab.ylim(ymin=0)
            
            pylab.subplot(1,5,2)
            pylab.plot(self.modSFS, self.F1sf[0,:], 'k', lw = 2)
            pylab.plot(guessSFS, guessF1sf[0,:], 'k--', lw = 2)
            pylab.title('Spatial frequency')
            pylab.ylim(ymin=0)
          
            pylab.subplot(1,5,3)
            pylab.plot(self.modSize, self.F1size[0,:], 'k', lw = 2)
            pylab.plot(guessSize, guessF1size[0,:], 'k--', lw = 2)
            pylab.title('Size')
            pylab.ylim(ymin=0)
           
            pylab.subplot(1,5,4)
            pylab.semilogx(self.modCont, self.F1cont[0,:], 'k', lw = 2)
            pylab.semilogx(guessCont, guessF1cont[0,:], 'k--', lw = 2)
            pylab.title('Contrast') 
            pylab.ylim(ymin=0)
            pylab.xticks([0.01, 0.1, 1.0])
            
            pylab.subplot(1,5,5)
            pylab.plot(self.modTf, self.F1Tf[0,:], 'k', lw = 2, label='Simulated Data')
            pylab.plot(guessTf, guessF1Tf[0,:], 'k--', lw = 2, label='Current guess')
            pylab.title('Temporal frequency')      
            pylab.ylim(ymin=0)
            pylab.legend()
            dialog.Update ( 7, 'Done')
            dialog.Close(True)
            
            pylab.show()
        def EvtHintButton(self,event):
            
            if self.hints == 0:
                hint = " Second time constant has value %0.5f \n This is a hard parameter to guess based on the tuning curves \n \n It was probably a good idea to get this one." %self.simParams[8]
                self.hints += 1
            elif self.hints == 1:
                hint = " First time constant has value %0.5f \n Strongly correlated with tau2 \n Also quite hard to guess." %self.simParams[7]
                self.hints += 1
            elif self.hints == 2:
                hint = " Surround width has value %0.2f \n The width of the surround can be deducted from how \n quickly the size tuning curve falls off after the maximum. \n \n This parameter can be hard to guess due to the fact that many \n combinations of parameters can produce similar tunings" %self.simParams[5]
                self.hints += 1
            elif self.hints == 3:
                hint = " The gain of the CRF is %0.2f \n \n Setting the gain to the correct value should help a lot \n in setting the remaining parameters" %self.simParams[1]
                self.hints += 1
            elif self.hints == 4:
                hint = " The gain of the surround is %0.2f \n This parameter effects the general gain of all curves and especially \n how much the the size tuning curve decreases after its maximum" %self.simParams[4]
                self.hints += 1    
            elif self.hints == 5:
                hint = " The width of the CRF is %0.2f \n This parameter can be deduced from the max of the size tuning curve" %self.simParams[4]
                self.hints += 1    
            elif self.hints == 5:
                hint = " The Normalisation pool exponent is %0.2f \n The exponent determines the saturation of the contrast tuning curve" %self.simParams[3]
                self.hints += 1  
            elif self.hints == 6:
                hint = " Spatial frequency is %0.2f \n This value is close to the peak of the spatial frequency curve" %self.simParams[6]
                self.hints += 1
            elif self.hints == 6:
                hint = " Orientation is %0.2f \n This is your last hint, now you should have a match" %self.simParams[0]
                self.hints += 1   
            else:
                hint = " You have seen all parameters. Still does not fit? \n Try looking at the full answer."
            f = wx.Frame(self, size = (500,200))
            p = self.hintPanel(f, hint)
            f.Show()
        
        def EvtAnswerButton(self,event):
            f = wx.Frame(self, size = (550,450))
            p = self.answerPanel(f,p=self.simParams)
            f.Show()
        
        class answerPanel(wx.Panel):
            def __init__(self, parent, p=[]):
                wx.Panel.__init__(self,parent)                
                wx.StaticText(self,-1,pos=(20,60),label="Orientation:") 
                o = '%.2f' %p[0]
                wx.StaticText(self, -1, pos = (350,60), label = o)
                
                wx.StaticText(self,-1,pos=(20,90),label="Gain of the CRF:")
                o = '%.2f' %p[1]
                wx.StaticText(self, -1, pos = (350,90), label = o)
                
                wx.StaticText(self,-1,pos=(20,120),label="Size of the CRF:")
                o = '%.2f' %p[2]
                wx.StaticText(self, -1, pos = (350,120), label = o)
                
                wx.StaticText(self,-1,pos=(20,150),label="Gain of the Surround:")
                o = '%.2f' %p[4]
                wx.StaticText(self, -1, pos = (350,150), label = o)
                
                wx.StaticText(self,-1,pos=(20,180),label="Size of the Surround:")
                o = '%.2f' %p[5]
                wx.StaticText(self, -1, pos = (350,180), label = o)
                
                wx.StaticText(self,-1,pos=(20,210),label="Normalisation pool exponent:")
                o = '%.2f' %p[3]
                wx.StaticText(self, -1, pos = (350,210), label = o)
                
                wx.StaticText(self,-1,pos=(20,240),label="Spatial frequency:")
                o = '%.2f' %p[6]
                wx.StaticText(self, -1, pos = (350,240), label = o)
                
                wx.StaticText(self,-1,pos=(20,270),label="First time constant of the temporal filter:")
                o = '%.5f' %p[7]
                wx.StaticText(self, -1, pos = (350,270), label = o)
                
                wx.StaticText(self,-1,pos=(20,300),label="Second time constant of the temporal filter:")
                o = '%.5f' %p[8]
                wx.StaticText(self, -1, pos = (350,300), label = o)
                
                wx.StaticText(self, -1, pos = (20,350), label = "Did you get close to the real parameters? \n Maybe you found a different solution that gave very similar tuning properties")
                
        class hintPanel(wx.Panel):
            def __init__(self, parent, hint = "Here is a hint"):
                wx.Panel.__init__(self,parent) 
                self.parent = parent
                wx.StaticText(self,-1,pos=(10,10),label=hint)
                
                closeButton = wx.Button(self,1001,"Close", pos = (230,150))
                self.Bind(wx.EVT_BUTTON, self.EvtCloseButton, closeButton, id=1001)
                self.Show(True)
            
            def EvtCloseButton(self, event):
                self.parent.Close(True)
                
app = wx.PySimpleApp()
frame = findV1(None,"Find V1")
app.MainLoop()