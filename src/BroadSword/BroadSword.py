
import ctypes as C
from ctypes import *
import numpy as np
import numpy.ctypeslib as npc
import pandas as pd
import csv
from reixs.LoadData import *

# Plotting
from bokeh.io import push_notebook
from bokeh.plotting import show, figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, LogColorMapper, ColorBar, Span, Label

# Widgets
import ipywidgets as widgets
from IPython.display import display
from ipyfilechooser import FileChooser


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
#SumSXS = np.zeros([2,3500,3]) # Total Summed Spectra
SumSXS = (C.c_float*3*3500*2)() 
Gauss = np.zeros([3500,3500]) # Gauss broadening matrix for each spectrum
Lorentz = np.zeros([3500,3500]) # Lorentz broadening matrix for each spectrum
Disorder = np.zeros([3500,3500]) # Disorder broadening matrix for each spectrum

ExpSXSCount = np.zeros([2],dtype=int) # Stores number of elements in the arrays
CalcSXSCase = 0
CalcSXSCount = np.zeros([3,40],dtype=int) # Stores number of elements in the arrays
BroadSXSCount = np.zeros([3,40],dtype=int) # Stores number of elements in the arrays
#SumSXSCount = np.zeros([3],dtype=int)
SumSXSCount = (C.c_int*3)()

# These store data for generating the broadening criteria
scaleXES = np.zeros([40,50])
Bands = np.zeros([50,40,2])
BandNum = np.zeros([40],dtype=int)
Fermi = 0
Fermis = np.zeros([40])
Binds = np.zeros([40])
shiftXES = np.zeros([40,50])
scalar = np.zeros([3,40])
# Edge = np.zeros([40],dtype=str)
Edge = []
# Edge = ["" for x in range(40)]
Site = np.zeros([40])

# Misc
bandshift = np.zeros([40,40])
bands_temp = np.zeros([3500,40,40])
bands_temp_count = np.zeros([40,40],dtype=int)
BandGap = 0

class Broaden():
    """
    Class to take in data and broaden it.
    We have to load the experimental, and then load the calculations. With the calculations we can load multiple different sets of files
    so that we account for the different sites that can exist. Just call the loadCalc function multiple times and it should account for the diffent sites.
    """

    def __init__(self):
        self.data = list() # Why is this here?

    def loadExp(self, basedir, XES, XANES, fermi):
        """
        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        XES, XANES : string
            Specify the file name (ASCII).
        fermi : float
            Specify the fermi energy for the ground state calculated spectra
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
        
        global CalcSXSCase
        global Fermi
        global Edge
        CalcSXSCase = 0
        Edge = []
        Fermi = fermi
        return

    def loadCalc(self, basedir, XES, XAS, XANES, fermis, binds, edge="K", sites=1):
        """
        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        XES, XAS, XANES : string
            Specify the file name (.txspec).
        fermis : float
            Specify the fermi energy for the excited state calculation
        bind : float
            Specify the binding energy of the ground state
        edge : string
            Specify the excitation edge "K","L2","L3","M4","M5"
        """
        global CalcSXSCase
        global Site

        with open(basedir+"/"+XES, "r") as xesFile: # XES Calculation
            df = pd.read_csv(xesFile, delimiter='\s+',header=None)
            c1 = 0
            for i in range(len(df)):
                CalcSXS[0][c1][0][CalcSXSCase] = df[0][c1] # Energy
                CalcSXS[1][c1][0][CalcSXSCase] = df[1][c1] # Counts
                c1 += 1
            CalcSXSCount[0][CalcSXSCase] = c1 # Length for each Site

        with open(basedir+"/"+XAS, "r") as xasFile: # XAS Calculation
            df = pd.read_csv(xasFile, delimiter='\s+',header=None)
            c1 = 0
            for i in range(len(df)):
                CalcSXS[0][c1][1][CalcSXSCase] = df[0][c1] # Energy
                CalcSXS[1][c1][1][CalcSXSCase] = df[1][c1] # Counts
                c1 += 1
            CalcSXSCount[1][CalcSXSCase] = c1 # Length for each Site

        with open(basedir+"/"+XANES, "r") as xanesFile: # XANES Calculation
            df = pd.read_csv(xanesFile, delimiter='\s+',header=None)
            c1 = 0
            for i in range(len(df)):
                CalcSXS[0][c1][2][CalcSXSCase] = df[0][c1] # Energy
                CalcSXS[1][c1][2][CalcSXSCase] = df[1][c1] # Counts
                c1 += 1
            CalcSXSCount[2][CalcSXSCase] = c1 # Length for each Site

        # Update the global variables with the parameters for that site.
        Fermis[CalcSXSCase] = fermis
        Binds[CalcSXSCase] = binds
        Edge.append(edge)
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
        self.FindBands()
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
                #print(Eval)
            c1 -= 1

        c1 = 0
        while c1 < BroadSXSCount[1][0]:
            if BroadSXS[1][c1][1][0] > 0:
                Econ = BroadSXS[0][c1][1][0]
                c1 = 999999
                #print(Econ)
            c1 += 1

        for c3 in range(3):
            for c1 in range(CalcSXSCase):
                for c2 in range(BroadSXSCount[c3][c1]):
                    BroadSXS[1][c2][c3][c1] = BroadSXS[1][c2][c3][c1]*(BroadSXS[0][c2][c3][c1]/Econ)
        global BandGap
        BandGap = Econ - Eval
        print("BandGap = " + str(BandGap) + " eV")
        p = figure(height=450, width=700, title="Un-Broadened Data", x_axis_label="Energy (eV)", y_axis_label="Normalized Intensity (arb. units)",
                   tools="pan,wheel_zoom,box_zoom,reset,crosshair,save")
        p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
            ("(x,y)", "(Energy, Intensity)"),
            ("(x,y)", "($x, $y)")
        ]))
        self.plotShiftCalc(p)
        self.plotExp(p)
        show(p)
        return

    def broaden(self, libpath="./"):
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

        #mylib = cdll.LoadLibrary('./libmatrices.so')
        try:
            mylib = cdll.LoadLibrary(libpath + "libmatrices.so")
        except OSError:
            try:
                mylib = cdll.LoadLibrary(libpath + "libmatrices_ARM64.dylib")
            except OSError:
                try:
                    mylib = cdll.LoadLibrary(libpath + "libmatrices_x86_64.dylib")
                except OSError as e:
                    print("Download the source and use the .c file to compile your own shared library and rename one of the existing .so or .dylib files.")
                    print(e)

        # These convert existing parameters into their respective ctypes. This takes very little time, but is super inefficient.
        # Can probably change the global variable declaration so that they are existing only as c types to begin with.

        cCalcSXSCase = C.c_int(CalcSXSCase)

        cBroadSXSCount = (C.c_int*40*3)()
        for c1 in range(3):
            for c2 in range(40):
                cBroadSXSCount[c1][c2] = BroadSXSCount[c1][c2]
        
        cdisord = C.c_float(disord)

        cscalar = (C.c_float*40*3)()
        for c1 in range(3):
            for c2 in range(40):
                cscalar[c1][c2] = scalar[c1][c2]
        
        
        cEdge = (C.c_int*40)()
        for c1 in range(len(Edge)):
            if Edge[c1] == "K":
                cEdge[c1] = 1
            elif Edge[c1] == "L2":
                cEdge[c1] = 2
            elif Edge[c1] == "L3":
                cEdge[c1] = 3
            elif Edge[c1] == "M4":
                cEdge[c1] = 4
            elif Edge[c1] == "M5":
                cEdge[c1] = 5
            else:
                cEdge[c1] = 1

        cSite = (C.c_float*40)()
        for c1 in range(40):
            cSite[c1] = Site[c1]

        mylib.broadXAS(cCalcSXSCase,cBroadSXSCount,BroadSXS,cdisord)
        mylib.add(cCalcSXSCase,cscalar,cEdge,cSite,BroadSXS,cBroadSXSCount,SumSXS,SumSXSCount)

        p = figure(height=450, width=700, title="Broadened Data", x_axis_label="Energy (eV)", y_axis_label="Normalized Intensity (arb. units)",
                   tools="pan,wheel_zoom,box_zoom,reset,crosshair,save")
        p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
            ("(x,y)", "(Energy, Intensity)"),
            ("(x,y)", "($x, $y)")
        ]))
        self.plotBroadCalc(p)
        self.plotExp(p)
        show(p)
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
    
    def add(self):
        Edge_check = ["K","L2","L3","M4","M5"]
        Edge_scale = [1,0.3333333,0.6666667,0.4,0.6]
        max = 0
        for c1 in range(CalcSXSCase):
            for c2 in range(3):
                scalar[c2][c1] = 1

        for c1 in range(CalcSXSCase): # Line 1159
            for c2 in range(5):
                if Edge[c1] == Edge_check[c2]:
                    for c3 in range(3):
                        scalar[c3][c1] = scalar[c3][c1]*Site[c1]*Edge_scale[c2]
        
        for c1 in range(3): # Line 1175
            first = 0
            value = BroadSXS[0][0][c1][0]
            c2 = 1
            while c2 < CalcSXSCase:
                if BroadSXS[0][0][c1][c2] >= value:
                    first = c2
                c2 += 1
            
            for c3 in range(BroadSXSCount[c1][first]):
                SumSXS[0][c3][c1] = BroadSXS[0][c3][c1][first]
                SumSXS[1][c3][c1] = scalar[c1][first]*BroadSXS[6][c3][c1][first]

            SumSXSCount[c1] = c3

            for c2 in range(CalcSXSCase):
                if c2 != first:
                    for c3 in range(SumSXSCount[c1]):
                        for c4 in range(BroadSXSCount[c1][c2]):
                            if BroadSXS[0][c4][c1][c2] > SumSXS[0][c3][c1]:
                                x1 = BroadSXS[0][c4-1][c1][c2]
                                x2 = BroadSXS[0][c4][c1][c2]
                                y1 = BroadSXS[6][c4-1][c1][c2]
                                y2 = BroadSXS[6][c4][c1][c2]
                                slope = (y2-y1)/(x2-x1)
                                SumSXS[1][c3][c1] = SumSXS[1][c3][c1] + scalar[c1][c2]*(slope*(SumSXS[0][c3][c1]-x1)+y1)
                                c4 = 9999999
                                max = c3

                    SumSXSCount[c1] = max
        return

    def plotExp(self,p):
        xesX = np.zeros([ExpSXSCount[0]])
        xesY = np.zeros([ExpSXSCount[0]])
        xanesX = np.zeros([ExpSXSCount[1]])
        xanesY = np.zeros([ExpSXSCount[1]])

        for c1 in range(ExpSXSCount[0]): # Experimental xes spectra
            xesX[c1] = ExpSXS[0][c1][0]
            xesY[c1] = ExpSXS[1][c1][0]
        
        for c1 in range(ExpSXSCount[1]): # Experimental xanes spectra
            xanesX[c1] = ExpSXS[0][c1][1]
            xanesY[c1] = ExpSXS[1][c1][1]
        
        #p = figure()
        p.line(xanesX,xanesY,line_color="red") # XANES plot
        p.line(xesX,xesY,line_color="red") # XES plot
        #show(p)
        return

    def plotShiftCalc(self,p):
        MaxCalcSXS = np.zeros([3,40])
        for c1 in range(CalcSXSCase):
            for c3 in range(3):
                for c2 in range(CalcSXSCount[c3][c1]):
                    if MaxCalcSXS[c3][c1] < BroadSXS[1][c2][c3][c1]:
                        MaxCalcSXS[c3][c1] = BroadSXS[1][c2][c3][c1]
        #p = figure()
        for c1 in range(CalcSXSCase):
            calcxesX = np.zeros([CalcSXSCount[0][c1]])
            calcxesY = np.zeros([CalcSXSCount[0][c1]])
            calcxasX = np.zeros([CalcSXSCount[1][c1]])
            calcxasY = np.zeros([CalcSXSCount[1][c1]])
            calcxanesX = np.zeros([CalcSXSCount[2][c1]])
            calcxanesY = np.zeros([CalcSXSCount[2][c1]])
            for c2 in range(CalcSXSCount[0][c1]): # Calculated XES spectra
                calcxesX[c2] = BroadSXS[0][c2][0][c1]
                calcxesY[c2] = BroadSXS[1][c2][0][c1] / (MaxCalcSXS[0][c1]) # This normalization doesn't work perfectly. The amplitude changes somewhere going into broadsxs
                #y = (x - x_min) / (x_max - x_min) Where x_min = 0

            for c2 in range(CalcSXSCount[1][c1]): # Calculated XAS spectra
                calcxasX[c2] = BroadSXS[0][c2][1][c1]
                calcxasY[c2] = BroadSXS[1][c2][1][c1] / (MaxCalcSXS[1][c1])

            for c2 in range(CalcSXSCount[2][c1]): # Calculated XANES spectra
                calcxanesX[c2] = BroadSXS[0][c2][2][c1]
                calcxanesY[c2] = BroadSXS[1][c2][2][c1] / (MaxCalcSXS[2][c1])
            colour = COLORP[c1]

            if colour == "#d60000": # So that there are no red spectra since the experimental is red
                colour = "Magenta"
                
            p.line(calcxesX,calcxesY,line_color=colour) # XES plot
            #p.line(calcxasX,calcxasY,line_color=colour) # XAS plot is not needed for lining up the spectra. Use XANES
            p.line(calcxanesX,calcxanesY,line_color=colour) # XANES plot
        #show(p)
        return
    
    def plotCalc(self):
        MaxCalcSXS = np.zeros([3,40])
        p = figure()
        for c1 in range(CalcSXSCase): # Since this is np array you can use :
            p.line(CalcSXS[0,:,0,c1], CalcSXS[1,:,0,c1]/ (MaxCalcSXS[0][c1])) # XES plot
            p.line(CalcSXS[0,:,1,c1], CalcSXS[1,:,1,c1]/ (MaxCalcSXS[1][c1])) # XAS plot
            p.line(CalcSXS[0,:,2,c1], CalcSXS[1,:,2,c1]/ (MaxCalcSXS[2][c1])) # XANES plot
        show(p)
        return

    def plotBroadCalc(self,p):
        MaxBroadSXS = np.zeros([3])
        for c3 in range(3):
            for c2 in range(SumSXSCount[c3]):
                if MaxBroadSXS[c3] < SumSXS[1][c2][c3]:
                    MaxBroadSXS[c3] = SumSXS[1][c2][c3]
        #p = figure()
        sumxesX = np.zeros([SumSXSCount[0]])
        sumxesY = np.zeros([SumSXSCount[0]])
        sumxasX = np.zeros([SumSXSCount[1]])
        sumxasY = np.zeros([SumSXSCount[1]])
        sumxanesX = np.zeros([SumSXSCount[2]])
        sumxanesY = np.zeros([SumSXSCount[2]])
        for c2 in range(SumSXSCount[0]): # Calculated XES spectra
            sumxesX[c2] = SumSXS[0][c2][0]
            sumxesY[c2] = SumSXS[1][c2][0] / MaxBroadSXS[0]

        for c2 in range(SumSXSCount[1]): # Calculated XAS spectra
            sumxasX[c2] = SumSXS[0][c2][1]
            sumxasY[c2] = SumSXS[1][c2][1] / MaxBroadSXS[1]

        for c2 in range(SumSXSCount[2]): # Calculated XANES spectra
            sumxanesX[c2] = SumSXS[0][c2][2]
            sumxanesY[c2] = SumSXS[1][c2][2] / MaxBroadSXS[2]

        p.line(sumxesX,sumxesY,line_color="limegreen") # XES plot
        p.line(sumxasX,sumxasY,line_color="blue") # XAS plot
        p.line(sumxanesX,sumxanesY,line_color="limegreen") # XANES plot
        #show(p)
        return


    def export(self, filename):
        """
        Export and write data to specified file.

        Parameters
        ----------
        filename : string
        """

        with open(f"{filename}_XES.csv", 'w', newline='') as f:
            writer = csv.writer(f,delimiter=" ")
            writer.writerow(["Energy","XES"])
            for c1 in range(SumSXSCount[0]):
                writer.writerow([SumSXS[0][c1][0],SumSXS[1][c1][0]])

        with open(f"{filename}_XAS.csv", 'w', newline='') as f:
            writer = csv.writer(f,delimiter=" ")
            writer.writerow(["Energy","XAS"])
            for c1 in range(SumSXSCount[1]):
                writer.writerow([SumSXS[0][c1][1],SumSXS[1][c1][1]])

        with open(f"{filename}_XANES.csv", 'w', newline='') as f:
            writer = csv.writer(f,delimiter=" ")
            writer.writerow(["Energy","XANES"])
            for c1 in range(SumSXSCount[2]):
                writer.writerow([SumSXS[0][c1][2],SumSXS[1][c1][2]])

        print(f"Successfully wrote DataFrame to {filename}.csv")
