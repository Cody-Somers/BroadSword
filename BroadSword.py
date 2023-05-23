
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
#CalcSXSCase = np.zeros(0,dtype=int) # The number of sites used for the project
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
SiteCount = 0

# Misc
bandshift = np.zeros([40,40])
bands_temp = np.zeros([3500,40,40])
bands_temp_count = np.zeros([40,40],dtype=int)

# print("Hello")
# print(ExpSXSCount)
# # print(CalcSXSCase)
# CalcSXSCase = 5
# print(CalcSXSCase)

# So the plan is to take in the calculations as a text file which we have to specify the name somehow...
# OR we could use the XAS loader from patrick that would allow us to be able to just open it from jupyter.
# I think having hardcoded text files would be easier though.

# THen for the txspec we will just load it the same way, so its two birds with one stone. Have a function that takes in all of the files
# Then it will deal with them appropriately. We need multiple functions though for multiple files of that type.

# We

class Broaden():
    """Class to take in data and broaden it."""

    def __init__(self):
        self.data = list()

    def loadExp(self, basedir, XES, XANES):
        """
        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        XES, XANES : string
            Specify the file name (either ASCII or HDF5).
        """

        with open(basedir+"/"+XES, "r") as xesFile:
            df = pd.read_csv(xesFile, delimiter=' ',header=None)
            c1 = 0
            for i in range(len(df)):
                ExpSXS[0][c1][0] = df[0][c1] # Energy
                ExpSXS[1][c1][0] = df[1][c1] # Counts
                c1 += 1
            ExpSXSCount[0] = c1 # Length

        with open(basedir+"/"+XANES, "r") as xanesFile:
            df = pd.read_csv(xanesFile, delimiter=' ',header=None)
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
            Specify the file name (either ASCII or HDF5).
        """
        global SiteCount

        with open(basedir+"/"+XES, "r") as xesFile:
            df = pd.read_csv(xesFile, delimiter=' ',header=None)
            c1 = 0
            for i in range(len(df)):
                CalcSXS[0][c1][0][SiteCount] = df[0][c1] # Energy
                CalcSXS[1][c1][0][SiteCount] = df[1][c1] # Counts
                c1 += 1
            CalcSXSCount[0] = c1

        with open(basedir+"/"+XAS, "r") as xasFile:
            df = pd.read_csv(xasFile, delimiter=' ',header=None)
            c1 = 0
            for i in range(len(df)):
                CalcSXS[0][c1][1][SiteCount] = df[0][c1] # Energy
                CalcSXS[1][c1][1][SiteCount] = df[1][c1] # Counts
                c1 += 1
            CalcSXSCount[1] = c1

        with open(basedir+"/"+XANES, "r") as xanesFile:
            df = pd.read_csv(xanesFile, delimiter=' ',header=None)
            c1 = 0
            for i in range(len(df)):
                CalcSXS[0][c1][2][SiteCount] = df[0][c1] # Energy
                CalcSXS[1][c1][2][SiteCount] = df[1][c1] # Counts
                c1 += 1
            CalcSXSCount[2] = c1
            
        SiteCount += 1
        return

