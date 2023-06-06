
import ctypes as C
from ctypes import *
import numpy as np
import numpy.ctypeslib as npc
import pandas as pd
from reixs.LoadData import *


# These are the input and output spectra of type float
# ['Column'] = [0=Energy, 1=Counts]
# ['Column'] = [0=Energy, 1=Counts, 2=CoreLifeXAS, 3=Intermediate Step, 4=Delta E, 5=Intermediate Step, 6=Final Gaussian Counts]
# ['Row'] = data
# ['XES, XANES'] = [0=XES, 1=XANES]
# ['XES, XAS, or XANES'] = [0=XES,1=XAS,2=XANES]
ExpSXS = np.zeros([2,1500,2]) # Experimental Spectra ['Column']['Row']['XES or XANES']
CalcSXS = np.zeros([2,3500,3,40]) # Calculated Spectra ['Column']['Row']['XES,XAS or XANES']['Site']
#BroadSXS = np.zeros([7,3500,3,40]) # Broadened Calculated Spectra ['Column']['Row']['XES,XAS or XANES']['Site']
BroadSXS = (C.c_float*40*3*3500*7)() 
SumSXS = np.zeros([2,3500,3]) # Total Summed Spectra
Gauss = np.zeros([3500,3500]) # Gauss broadening matrix for each spectrum
Lorentz = np.zeros([3500,3500]) # Lorentz broadening matrix for each spectrum
Disorder = np.zeros([3500,3500]) # Disorder broadening matrix for each spectrum

ExpSXSCount = np.zeros([2],dtype=int) # Stores number of elements in the arrays
CalcSXSCase = 0
CalcSXSCount = np.zeros([3,40],dtype=int) # Stores number of elements in the arrays
BroadSXSCount = np.zeros([3,40],dtype=int) # Stores number of elements in the arrays
SumSXSCount = np.zeros([3],dtype=int)

# These store data for generating the broadening criteria
scaleXES = np.zeros([40,50])
Bands = np.zeros([50,40,2])
BandNum = np.zeros([40],dtype=int)
Fermi = 0
Fermis = np.zeros([40])
Binds = np.zeros([40])
shiftXES = np.zeros([40,50])
scalar = np.zeros([3,40])
Edge = np.zeros([40],dtype=str)
Site = np.zeros([40])

# Misc
bandshift = np.zeros([40,40])
bands_temp = np.zeros([3500,40,40])
bands_temp_count = np.zeros([40,40],dtype=int)
BandGap = 0

class Broaden():
    """Class to take in data and broaden it.
    We have to load the experimental, and then load the calculations. With the calculations we can load multiple different sets of files
    so that we account for the different sites that can exist. Just call the loadCalc function multiple times and it should account for the diffent sites.
    """

    def __init__(self):
        self.data = list()

    def loadExp(self, basedir, XES, XANES):
        """
        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        XES, XANES : string
            Specify the file name (ASCII).
        """

        with open(basedir+"/"+XES, "r") as xesFile:
            df = pd.read_csv(xesFile, delimiter='\s+',header=None) # Change to '\s*' and specify engine='python' if this breaks in jupyter notebook
            c1 = 0
            for i in range(len(df)): 
                ExpSXS[0][c1][0] = df[0][c1] # Energy
                ExpSXS[1][c1][0] = df[1][c1] # Counts
                c1 += 1
            ExpSXSCount[0] = c1 # Length

        with open(basedir+"/"+XANES, "r") as xanesFile:
            df = pd.read_csv(xanesFile, delimiter='\s+',header=None)
            c1 = 0
            for i in range(len(df)):
                ExpSXS[0][c1][1] = df[0][c1] # Energy
                ExpSXS[1][c1][1] = df[1][c1] # Counts
                c1 += 1
            ExpSXSCount[1] = c1 # Length
        return

    def loadCalc(self, basedir, XES, XAS, XANES, sites=1):
        """
        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        XES, XAS, XANES : string
            Specify the file name (.txspec).
        """
        global CalcSXSCase
        global Site

        with open(basedir+"/"+XES, "r") as xesFile:
            df = pd.read_csv(xesFile, delimiter='\s+',header=None)
            c1 = 0
            for i in range(len(df)):
                CalcSXS[0][c1][0][CalcSXSCase] = df[0][c1] # Energy
                CalcSXS[1][c1][0][CalcSXSCase] = df[1][c1] # Counts
                c1 += 1
            CalcSXSCount[0][CalcSXSCase] = c1 # Length for each Site

        with open(basedir+"/"+XAS, "r") as xasFile:
            df = pd.read_csv(xasFile, delimiter='\s+',header=None)
            c1 = 0
            for i in range(len(df)):
                CalcSXS[0][c1][1][CalcSXSCase] = df[0][c1] # Energy
                CalcSXS[1][c1][1][CalcSXSCase] = df[1][c1] # Counts
                c1 += 1
            CalcSXSCount[1][CalcSXSCase] = c1 # Length for each Site

        with open(basedir+"/"+XANES, "r") as xanesFile:
            df = pd.read_csv(xanesFile, delimiter='\s+',header=None)
            c1 = 0
            for i in range(len(df)):
                CalcSXS[0][c1][2][CalcSXSCase] = df[0][c1] # Energy
                CalcSXS[1][c1][2][CalcSXSCase] = df[1][c1] # Counts
                c1 += 1
            CalcSXSCount[2][CalcSXSCase] = c1 # Length for each Site

        Site[CalcSXSCase] = sites
        CalcSXSCase += 1
        return

    def FindBands(self): # Perhaps change the while loops to for in range()
        c1 = 0
        while c1 < CalcSXSCase: # For each site (.loadCalc)
            starter = False
            c3 = 0
            c2 = 0
            while c2 < CalcSXSCount[0][c1]: # For each data point
                if starter is False:
                    if CalcSXS[1][c2][0][c1] != 0: # Spectrum is not zero
                        Bands[c3][c1][0] = CalcSXS[0][c2][0][c1] # Start point of band
                        starter = True
                if starter is True:
                    if CalcSXS[1][c2][0][c1] == 0: # Spectrum hits zero
                        Bands[c3][c1][1] = CalcSXS[0][c2][0][c1] # End point of band
                        starter = False
                        c3 += 1
                c2 += 1
            BandNum[c1] = c3 # The number of bands in each spectrum
            c1 += 1
        return
    
    def Shift(self,XESshift, XASshift, XESbandshift=0):
        """
        This will shift the files initially when uploaded, then by user specifed shifts to XES and XAS until alligned with experimental spectra.

        Parameters
        ----------
        XESshift : float
            Specify a constant shift to the entire XES spectrum
        XASshift : float
            Specify a constant shift to the entire XAS spectrum
        XESbandshift : [float]
            Specify a shift for each individual band found in FindBands()
        """
        Ryd = 13.605698066 # Rydberg energy to eV
        Eval = 0 # Location of valence band
        Econ = 0 # Location of conduction band
        if XESshift != 0: # Constant shift to all bands
            for c1 in range(CalcSXSCase):
                for c2 in range(BandNum[c1]):
                    shiftXES[c1][c2] = XESshift
        else: # Shift bands separately.
            for c1 in range(CalcSXSCase):
                for c2 in range(BandNum[c1]):
                    shiftXES[c1][c2] = XESshift 
                    #TODO This is something that should be done eventually, but has low usage
                    # Need to figure out how to get the individual shifts to work.
                    # Perhaps this could be put into the .loadCalc so that they.
                    # I would need to call find bands in .loadCalc and then print them out. Not a big issue rn.

        shiftXAS = XASshift
        for c1 in range(CalcSXSCase): # This goes through the XAS spectra
            for c2 in range(CalcSXSCount[1][c1]): #Line 504
                BroadSXS[1][c2][1][c1] = CalcSXS[1][c2][1][c1] # Counts from calc go into Broad
                BroadSXSCount[1][c1] = CalcSXSCount[1][c1]
                BroadSXS[0][c2][1][c1] = CalcSXS[0][c2][1][c1] + shiftXAS + (Binds[c1]+Fermi) * Ryd # Shift the energy of XAS appropriately
        
        for c1 in range(CalcSXSCase): # This goes through the XANES spectra
            for c2 in range(CalcSXSCount[2][c1]): #Line 514
                BroadSXS[1][c2][2][c1] = CalcSXS[1][c2][2][c1] # Counts from calc go into Broad
                BroadSXSCount[2][c1] = CalcSXSCount[2][c1]
                BroadSXS[0][c2][2][c1] = CalcSXS[0][c2][2][c1] + shiftXAS + (Binds[c1]+Fermis[c1]) * Ryd # Shift the energy of XANES appropriately

        for c1 in range(CalcSXSCase): # If there are a different shift between bands find that difference
            for c2 in range(BandNum[c1]): # Line 526
                bandshift[c1][c2] = shiftXES[c1][c2] - shiftXES[c1][0]

        for c1 in range(CalcSXSCase): # This goes through the XES spectra
            BroadSXSCount[0][c1] = CalcSXSCount[0][c1]
            for c2 in range(CalcSXSCount[0][c1]): # Line 535
                BroadSXS[0][c2][0][c1] = CalcSXS[0][c2][0][c1] + bandshift[c1][0] # Still confused why bandshift[c1][0] is here. Always zero
                BroadSXS[1][c2][0][c1] = CalcSXS[1][c2][0][c1]

        for c1 in range(CalcSXSCase): # No idea why this is here
            c2 = 1 # Line 544
            c3 = 0
            while c3 < BroadSXSCount[0][c1]:
                if BroadSXS[0][c3][0][c1] >= (Bands[c2][c1][0]+bandshift[c1][0]):
                    c4 = 0
                    while BroadSXS[1][c3][0][c1] != 0:
                        bands_temp[c4][c2][c1] = BroadSXS[1][c3][0][c1]
                        BroadSXS[1][c3][0][c1] = 0
                        c3 += 1
                        c4 +=1
                    bands_temp_count[c1][c2] = c4
                    c2 += 1
                    if c2 >= BandNum[c1]:
                        c3 = 99999
                c3 += 1

        for c1 in range(CalcSXSCase):
            for c2 in range(2,BandNum[c1]): # Line 570
                c3 = 0
                while c3 < BroadSXSCount[0][c1]:
                    if BroadSXS[0][c3][0][c1] >= (Bands[c2][c1][0] + bandshift[c1][c2]):
                        c4 = 0
                        while c4 < bands_temp_count[c1][c2]:
                            BroadSXS[1][c3][0][c1] = bands_temp[c4][c2][c1]
                            c4 += 1
                            c3 += 1
                        c3 = 99999
                    c3 += 1
        
        for c1 in range(CalcSXSCase):
            for c2 in range(BroadSXSCount[0][c1]): # Line 592
                BroadSXS[0][c2][0][c1] = BroadSXS[0][c2][0][c1] + shiftXES[c1][0] + (Binds[c1]+Fermi) * Ryd

        c1 = BroadSXSCount[0][0]-1
        while c1 >= 0:
            if BroadSXS[1][c1][0][0] > 0:
                Eval = BroadSXS[0][c1][0][0]
                c1 = -1
                print(Eval)
            c1 -= 1

        c1 = 0
        while c1 < BroadSXSCount[1][0]:
            if BroadSXS[1][c1][1][0] > 0:
                Econ = BroadSXS[0][c1][1][0]
                c1 = 999999
                print(Econ)
            c1 += 1

        for c3 in range(3):
            for c1 in range(CalcSXSCase):
                for c2 in range(BroadSXSCount[c3][c1]):
                    BroadSXS[1][c2][c3][c1] = BroadSXS[1][c2][c3][c1]*(BroadSXS[0][c2][c3][c1]/Econ)
        global BandGap
        BandGap = Econ - Eval
        print(BandGap)
        return

    def broaden(self):
        Econd = np.zeros(40)
        type = False
        energy_0 = 20

        if XESscale != 0:
            for c1 in range(CalcSXSCase):
                for c2 in range(BandNum[c1]):
                    scaleXES[c1][c2] = XESscale
        else:
            return # TODO implement the band where you can scale individually. Same as up above problem with the shifting.
        
        for c1 in range(CalcSXSCase): # Line 791
            c2 = 0
            while c2 < BroadSXSCount[2][c1]:
                if BroadSXS[1][c2][2][c1] != 0:
                    Econd[c1] = BroadSXS[0][c2][2][c1]
                    c2 = 999999
                c2 += 1
        
        for c1 in range(CalcSXSCase): # Using scaling factor for corehole lifetime for XAS and XANES
            for c2 in range(1,3): # Line 805
                for c3 in range(BroadSXSCount[c2][c1]):
                    if BroadSXS[0][c3][c2][c1] <= Econd[c1]:
                        BroadSXS[2][c3][c2][c1] = corelifeXAS
                    else:
                        if BroadSXS[0][c3][c2][c1] < Econd[c1] + energy_0:
                            BroadSXS[2][c3][c2][c1] = scaleXAS/100 * ((BroadSXS[0][c3][c2][c1]-Econd[c1]) * (BroadSXS[0][c3][c2][c1]-Econd[c1])) + corelifeXAS # Replace with **2 ??
                        else:
                            BroadSXS[2][c3][c2][c1] = scaleXAS/100 * (energy_0 * energy_0) + corelifeXAS
                    BroadSXS[4][c3][c2][c1] = BroadSXS[0][c3][c2][c1] / mono

        for c1 in range(CalcSXSCase): # Corehole lifetime scaling for XES
            type = False # Line 830
            c3 = 0
            for c2 in range(BroadSXSCount[0][c1]):
                BroadSXS[4][c2][0][c1] = BroadSXS[0][c2][0][c1]/spec
                if type is False:
                    if BroadSXS[1][c2][0][c1] != 0:
                        type = True
                    else:
                        BroadSXS[2][c2][0][c1] = scaleXES[c1][c3]/100 * ((BroadSXS[0][c2][0][c1]-Econd[c1]) * (BroadSXS[0][c2][0][c1]-Econd[c1])) + corelifeXES
                if type is True:
                    if BroadSXS[1][c2][0][c1] == 0:
                        BroadSXS[2][c2][0][c1] = scaleXES[c1][c3]/100 * ((BroadSXS[0][c2][0][c1]-Econd[c1]) * (BroadSXS[0][c2][0][c1]-Econd[c1])) + corelifeXES
                        type = False
                        c3 += 1
                        if c3 > BandNum[c1]:
                            c3 = BandNum[c1]-1
                    else:
                        BroadSXS[2][c2][0][c1] = scaleXES[c1][c3]/100 * ((BroadSXS[0][c2][0][c1]-Econd[c1]) * (BroadSXS[0][c2][0][c1]-Econd[c1])) + corelifeXES

        mylib = cdll.LoadLibrary('./libmatrices.so')
        cCalcSXSCase = C.c_int(CalcSXSCase)

        cBroadSXSCount = (C.c_int*40*3)()
        for c1 in range(3):
            for c2 in range(40):
                cBroadSXSCount[c1][c2] = BroadSXSCount[c1][c2]
        
        #cBroadSXS = (C.c_float*40*3*3500*7)() # There might be a way to speed this up so that it doesn't take as long
        #for c1 in range(7):
        #    for c2 in range(3500):
        #        for c3 in range(3):
        #            for c4 in range(40):
        #                cBroadSXS[c1][c2][c3][c4] = BroadSXS[c1][c2][c3][c4]
        
        #tBroadSXS = BroadSXS.ctypes.data_as(c_float)
        cdisord = C.c_float(disord)
        print(BroadSXS[0][2000][1][0])
        print(BroadSXS[1][2000][1][0])
        mylib.broadXAS(cCalcSXSCase,cBroadSXSCount,BroadSXS,cdisord)
        print(BroadSXS[6][2000][1][0])
        return

    def initParam(self, fermi, fermis, binds, edge):
        """
        Parameters
        ----------
        fermi : float
            Specify the fermi energy for the ground state spectra
        fermis : [float] A list of floats
            Specify the fermi energy for all of the excited state calculations
        bind : [float] A list of floats
            Specify the binding energy of the ground states for each site
        edge : [String] A list of strings
            Specify the excitation edges
        """
        global Fermi
        global Fermis
        global Binds
        global Edge
        Fermi = fermi
        Fermis = fermis # Might have to change this so that you keep the size of the arrays the same.
        Binds = binds
        Edge = edge
        #TODO, put these parameters into loadCalc preferably. Same way that sites is being done

        return

    def initResolution(self, XEScorelife, specResolution, monoResolution, disorder, XESscaling, XASscaling):
        """
        Parameters
        ----------
        XEScorelife : float
            Specify the corehole lifetime broadening factor
        specResolution : float
            Specify spectrometer resolving power
        monoResolution : float
            Specify monochromator resolving power
        disorder : float
            Specify disorder factor
        XESscaling : float
            Specify corehole lifetime scaling factor for XES
        XASscaling : float
            Specify corehole lifetime scaling factor for XAS
        """
        global corelifeXES
        global corelifeXAS
        global spec
        global mono
        global disord
        global XESscale
        global scaleXAS
        corelifeXES = XEScorelife
        corelifeXAS = XEScorelife
        spec = specResolution
        mono = monoResolution
        disord = disorder
        XESscale = XESscaling
        scaleXAS = XASscaling
        return
