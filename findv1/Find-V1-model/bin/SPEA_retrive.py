import pylab, sys, numpy, glob, manualFit
#cell = 'm177/m177q'

class SPEA_retrive:
    def __init__(self, cell):
        args = numpy.load('SPEA_fits'+cell[cell.find('/'):] + 'args.npy')
        errs = numpy.load('SPEA_fits'+cell[cell.find('/'):] + 'errs.npy') 

        for ii in range(args.shape[0]):
            dat = manualFit.manualFit(cell,args[ii,:])
            pylab.figure(ii+1,figsize=[12,7])
            dat.plotFit(data=0)

        pylab.show()
        