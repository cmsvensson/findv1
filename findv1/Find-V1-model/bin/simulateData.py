import numpy as N
import Neuron, tuningFunctions, wx
from Params import P

class simulateData():
    def __init__(self):
        self.P = N.asarray(P)
        self.PP, self.Pmean, self.V, self.vh = self.princComp()
        
        self.Psim = N.asarray(self.generateParams()[0,:])
        print self.Psim
        self.modOris, self.F1ori, self.modSFS, self.F1sf, self.modSize, self.F1size, self.modCont, self.F1cont, self.modTf, self.F1Tf = self.compResponses(self.Psim)
        print self.modOris, self.F1ori, self.modSFS, self.F1sf, self.modSize, self.F1size, self.modCont, self.F1cont, self.modTf, self.F1Tf

        
    def princComp(self):
        P7 = N.min([self.P[7,:],self.P[8,:]],0)
        P8 = N.max([self.P[7,:],self.P[8,:]],0)
        self.P[7,:] = P7; self.P[8,:] = P8

        self.P[0,:]=self.P[0,:]%360.0
        self.P=self.P.T
        Pmean = N.mean(self.P,0);
        for ii in range(9):
            self.P[:,ii] -= Pmean[ii]
        u,s,vh = N.linalg.svd(self.P); V=vh.T
        
        PP = N.dot(self.P,V[:,0:])
        
        return PP, Pmean, V, vh
        
    def generateParams(self):
        Np = N.zeros([9,19]); Xp = N.zeros([9,20]);
        for ii in range(9):
            bins = N.linspace(N.max([N.min(self.PP[:,ii]),N.mean(self.PP[:,ii])-2.5*N.std(self.PP[:,ii])]),N.min([N.max(self.PP[:,ii]),N.mean(self.PP[:,ii])+2.5*N.std(self.PP[:,ii])]),20)
            Np[ii,:], Xp[ii,:] = N.histogram(self.PP[:,ii], bins)
            Np[ii,:] = Np[ii,:]/float(max(Np[ii,:]))

        good_sim = False
        while not good_sim:   
            Psim = N.zeros([1,9]);    
            ii=0
            while ii < 9:
                ps = N.random.rand()*(N.max(self.PP[:,ii])-N.min(self.PP[:,ii]))+N.min(self.PP[:,ii]);
                II = N.argmin(abs(Xp[ii,:]-ps));
                if II == 19:
                    II = 18
                if N.random.rand()<Np[ii,II]:
                    Psim[0,ii] = ps
                    ii += 1;
            if (N.dot(Psim[0,:],self.vh)+self.Pmean>=0.0).all():        
                good_sim = True
                Psim = N.dot(Psim,self.vh[0:,:])
                for ii in range(9):
                    Psim[:,ii] += self.Pmean[ii]

        Psim = abs(Psim)
        return Psim
    
    def compResponses(self,params):
        dialog = wx.ProgressDialog( 'Progress', 'Computing Responses', maximum = 6, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME )
        modelNeur = Neuron.V1neuron(sizeDeg=3, sizePix=128, 
                                    crfWidth=params[2], ori = params[0],
                                    surroundWidth=params[5],
                                    surroundGain=params[4], normPoolExp=params[3],
                                    phase=0.0, sf = params[6], 
                                    crfGain=params[1], tau1 = params[7],
                                    tau2 = params[8]) 
        dialog.Update ( 1, 'Computing Orientation')
        modOris, P, F0ori, F1ori = tuningFunctions.oriTuning(modelNeur)
        dialog.Update ( 2, 'Computing Spatial Frequency')
        modSFS, P, F0sf, F1sf = tuningFunctions.sfTuning(modelNeur)
        dialog.Update ( 3, 'Computing Size tuning')
        modSize, P, F0size, F1size = tuningFunctions.sizeTuning(modelNeur)
        dialog.Update ( 4, 'Computing Contrast tuning')
        modCont, P, F0cont, F1cont = tuningFunctions.contrastTuning(modelNeur)
        dialog.Update ( 5, 'Computing Temporal Frequency')
        modTf, P, F0Tf, F1Tf = tuningFunctions.tfTuning(modelNeur)
        dialog.Update ( 6, 'Done')
        dialog.Close(True)
        
        return modOris, F1ori, modSFS, F1sf, modSize, F1size, modCont, F1cont, modTf, F1Tf
    #Getters
    def getParams(self):
        return self.Psim
    
    def getResponses(self):
        return self.modOris, self.F1ori, self.modSFS, self.F1sf, self.modSize, self.F1size, self.modCont, self.F1cont, self.modTf, self.F1Tf
    