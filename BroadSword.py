
import numpy as np
import pandas as pd
from reixs.LoadData import *


# These are the input and output spectra of type float
ExpSXS = np.zeros([2,1500,2]) # Experimental Spectra ['Column']['Row']['XES or XANES']
CalcSXS = np.zeros([2,3500,3,40]) # Calculated Spectra ['Column']['Row']['XES,XAS or XANES']['Site']
BroadSXS = np.zeros([7,3500,3,40]) # Broadened Calculated Spectra ['Column']['Row']['XES,XAS or XANES']['Site']
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

    def loadCalc(self, basedir, XES, XAS, XANES):
        """
        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        XES, XAS, XANES : string
            Specify the file name (.txspec).
        """
        global CalcSXSCase

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

        CalcSXSCase += 1
        return

    def FindBands(self):
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
    
    def Shift(self):

        return