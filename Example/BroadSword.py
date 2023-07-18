
import ctypes as C
from ctypes import *
import numpy as np
import numpy.ctypeslib as npc
import pandas as pd
import csv
import time
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
BroadSXS = np.zeros([7,3500,3,40]) # Broadened Calculated Spectra ['Column']['Row']['XES,XAS or XANES']['Site']
#BroadSXS = (C.c_float*40*3*3500*7)() 
SumSXS = np.zeros([2,3500,3]) # Total Summed Spectra
#SumSXS = (C.c_float*3*3500*2)() 
Gauss = np.zeros([3500,3500]) # Gauss broadening matrix for each spectrum
Lorentz = np.zeros([3500,3500]) # Lorentz broadening matrix for each spectrum
Disorder = np.zeros([3500,3500]) # Disorder broadening matrix for each spectrum
placeholder = np.zeros([3500,3500]) # An empty thing to hold stuff

ExpSXSCount = np.zeros([2],dtype=int) # Stores number of elements in the arrays of Experimental data
CalcSXSCase = 0
CalcSXSCount = np.zeros([3,40],dtype=int) # Stores number of elements in the arrays of Calculated data
BroadSXSCount = np.zeros([3,40],dtype=int) # Stores number of elements in the arrays of Shifted/Intermediate data
SumSXSCount = np.zeros([3],dtype=int)
#SumSXSCount = (C.c_int*3)() # Store number of elements in the arrays of Final Data

# These store data for generating the broadening criteria
scaleXES = np.zeros([40,50])
Bands = np.zeros([50,40,2]) 
BandNum = np.zeros([40],dtype=int)
Fermi = 0 # Ground state fermi energy
Fermis = np.zeros([40]) # Excited state fermi energy for each inequivalent site
Binds = np.zeros([40]) # Ground statet binding energy for each inequivalent site
shiftXES = np.zeros([40,50])
scalar = np.zeros([3,40])
# Edge = np.zeros([40],dtype=str)
Edge = []
Site = np.zeros([40])

# Misc
bandshift = np.zeros([40,40])
bands_temp = np.zeros([3500,40,40])
bands_temp_count = np.zeros([40,40],dtype=int)
BandGap = 0

class Broaden():
    """
    Class designed to take in calculated spectral data, align it with experimental data, then broaden it appropriately.
    First: Load the experimental. Second: Load all of the calculations sequentially. 
    Third: Generate the parameters used for shifting and broadening. Finally: Broaden the spectra.
    """

    def __init__(self):
        self.data = list() # Why is this here?

    def loadExp(self, basedir, XES, XANES, fermi):
        """
        Loads the experimental data.

        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        XES, XANES : string
            Specify the file name (ASCII) including the extension.
            Reminder that there is no header allowed in this file.
        fermi : float
            Specify the fermi energy for the ground state calculated spectra. Found in .scf2
        """

        with open(basedir+"/"+XES, "r") as xesFile: # Measured XES
            df = pd.read_csv(xesFile, delimiter='\s+',header=None) # Change to '\s*' and specify engine='python' if this breaks in jupyter notebook
            c1 = 0
            for i in range(len(df)): 
                ExpSXS[0][c1][0] = df[0][c1] # Energy
                ExpSXS[1][c1][0] = df[1][c1] # Counts
                c1 += 1
            ExpSXSCount[0] = c1 # Length of data points

        with open(basedir+"/"+XANES, "r") as xanesFile: # Measured XANES
            df = pd.read_csv(xanesFile, delimiter='\s+',header=None)
            c1 = 0
            for i in range(len(df)):
                ExpSXS[0][c1][1] = df[0][c1] # Energy
                ExpSXS[1][c1][1] = df[1][c1] # Counts
                c1 += 1
            ExpSXSCount[1] = c1 # Length of data points
        
        global CalcSXSCase
        global Fermi
        global Edge
        CalcSXSCase = 0 # Stores number of calculated inequivalent sites
        Edge = []
        Fermi = fermi
        return

    def loadCalc(self, basedir, XES, XAS, XANES, fermis, binds, edge="K", sites=1):
        """
        Loads the calculated data.
        
        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        XES, XAS, XANES : string
            Specify the file name including the extension (.txspec).
        fermis : float
            Specify the fermi energy for the excited state calculation. Found in .scf2
        binds : float
            Specify the binding energy of the ground state. Found in .scfc
        edge : string
            Specify the excitation edge "K","L2","L3","M4","M5".
        sites : float
            Specify the number of atomic positions present in the inequivalent site.
        """
        global CalcSXSCase

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

    def FindBands(self): 
        """
        Finds the number of bands present in the calculated data.
        Bands are where the calculated data hits zero.
        """
        # The while loops can be changed to "for in range()"
        c1 = 0
        while c1 < CalcSXSCase: # For each site (number of .loadCalc)
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
    
    def printBands(self):
        """
        Prints the value of the band start and end locations, then plots the unshifted spectra.
        """
        self.FindBands()
        for c1 in range(CalcSXSCase):
            print("In inequivalent atom #" + str(c1))
            for c2 in range(BandNum[c1]):
                print("Band #" + str(c2) + " is located at " + str(Bands[c2][c1][0]) + " to " + str(Bands[c2][c1][1]))
        print("Reminder that these values are unshifted by the binding and fermi energies")
        self.plotCalc()
        return
    
    def Shift(self, XESshift, XASshift, XESbandshift=0):
        """
        This will shift the files initially based on binding and fermi energy, then by user specifed shifts to XES and XAS 
        until alligned with experimental spectra.

        Parameters
        ----------
        XESshift : float
            Specify a constant shift to the entire XES spectrum in eV.
        XASshift : float
            Specify a constant shift to the entire XAS spectrum in eV.
        XESbandshift : [float]
            Specify a shift for each individual band found in printBands().
            Should be in the format of [[Bands in inequivalent atom 0] , [Bands in inequivalent atom 2], [Bands in inequivalent atom 3]]
            For example, with 2 inequivalent site and 3 bands in each site: [[17, 18, 18] , [16.5, 18, 18]]
            In atom 1 this shifts the first band by 17 and the other two by 18. In atom 2 it shifts first by 16.5 and the other by 18.
        """
        self.FindBands()
        Ryd = 13.605698066 # Rydberg energy to eV
        Eval = 0 # Location of valence band
        Econ = 0 # Location of conduction band
        if XESbandshift == 0: # Constant shift to all bands
            for c1 in range(CalcSXSCase):
                for c2 in range(BandNum[c1]):
                    shiftXES[c1][c2] = XESshift
        else: # Shift bands separately.
            for c1 in range(CalcSXSCase):
                for c2 in range(BandNum[c1]):
                    shiftXES[c1][c2] = XESbandshift[c1][c2]

        shiftXAS = XASshift
        for c1 in range(CalcSXSCase): # This goes through the XAS spectra
            for c2 in range(CalcSXSCount[1][c1]): # Line 504
                BroadSXS[1][c2][1][c1] = CalcSXS[1][c2][1][c1] # Counts from calc go into Broad
                BroadSXSCount[1][c1] = CalcSXSCount[1][c1]
                BroadSXS[0][c2][1][c1] = CalcSXS[0][c2][1][c1] + shiftXAS + (Binds[c1]+Fermi) * Ryd # Shift the energy of XAS based on binding, fermi energy, and user input
        
        for c1 in range(CalcSXSCase): # This goes through the XANES spectra
            for c2 in range(CalcSXSCount[2][c1]): # Line 514
                BroadSXS[1][c2][2][c1] = CalcSXS[1][c2][2][c1] # Counts from calc go into Broad
                BroadSXSCount[2][c1] = CalcSXSCount[2][c1]
                BroadSXS[0][c2][2][c1] = CalcSXS[0][c2][2][c1] + shiftXAS + (Binds[c1]+Fermis[c1]) * Ryd # Shift the energy of XANES based on binding, fermi energy, and user input

        for c1 in range(CalcSXSCase): # If there are a different shift between bands find that difference
            for c2 in range(BandNum[c1]): # Line 526
                bandshift[c1][c2] = shiftXES[c1][c2] - shiftXES[c1][0]

        for c1 in range(CalcSXSCase): # This goes through the XES spectra
            BroadSXSCount[0][c1] = CalcSXSCount[0][c1]
            for c2 in range(CalcSXSCount[0][c1]): # Line 535
                BroadSXS[0][c2][0][c1] = CalcSXS[0][c2][0][c1] + bandshift[c1][0] # Still confused why bandshift[c1][0] is here. Always zero
                BroadSXS[1][c2][0][c1] = CalcSXS[1][c2][0][c1]

        for c1 in range(CalcSXSCase): # Not entirely sure the purpose of the next portion of code
            c2 = 1 # Line 544
            c3 = 0
            while c3 < BroadSXSCount[0][c1]:
                if BroadSXS[0][c3][0][c1] >= (Bands[c2][c1][0] + bandshift[c1][0]):
                    c4 = 0
                    while BroadSXS[1][c3][0][c1] != 0:
                        bands_temp[c4][c2][c1] = BroadSXS[1][c3][0][c1]
                        BroadSXS[1][c3][0][c1] = 0
                        c3 += 1
                        c4 += 1
                    bands_temp_count[c1][c2] = c4
                    c2 += 1
                    if c2 >= BandNum[c1]:
                        c3 = 999999
                c3 += 1

        for c1 in range(CalcSXSCase):
            for c2 in range(1,BandNum[c1]): # Line 570
                c3 = 0
                while c3 < BroadSXSCount[0][c1]:
                    if BroadSXS[0][c3][0][c1] >= (Bands[c2][c1][0] + bandshift[c1][c2]):
                        c4 = 0
                        while c4 < bands_temp_count[c1][c2]:
                            BroadSXS[1][c3][0][c1] = bands_temp[c4][c2][c1]
                            c4 += 1
                            c3 += 1
                        c3 = 999999
                    c3 += 1
        
        for c1 in range(CalcSXSCase):
            for c2 in range(BroadSXSCount[0][c1]): # Line 592
                BroadSXS[0][c2][0][c1] = BroadSXS[0][c2][0][c1] + shiftXES[c1][0] + (Binds[c1]+Fermi) * Ryd # Shift XES spectra based on binding, fermi energy, and user input

        c1 = BroadSXSCount[0][0]-1
        while c1 >= 0: # Starts from the top and moves down until it finds the point where the valence band != 0
            if BroadSXS[1][c1][0][0] > 0:
                Eval = BroadSXS[0][c1][0][0]
                c1 = -1
            c1 -= 1

        c1 = 0
        while c1 < BroadSXSCount[1][0]: # Starts from the bottom and moves up until it finds the point where the conduction bands != 0
            if BroadSXS[1][c1][1][0] > 0:
                Econ = BroadSXS[0][c1][1][0]
                c1 = 999999
            c1 += 1

        for c3 in range(3):
            for c1 in range(CalcSXSCase):
                for c2 in range(BroadSXSCount[c3][c1]):
                    BroadSXS[1][c2][c3][c1] = BroadSXS[1][c2][c3][c1] * (BroadSXS[0][c2][c3][c1] / Econ)
        
        global BandGap
        BandGap = Econ - Eval # Calculate the band gap
        print("BandGap = " + str(BandGap) + " eV")

        # Create the figure for plotting shifted spectra
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
        """
        This will take the shifted calculated spectra and broaden it based on the lifetime, instrument, and general disorder broadening.
        It creates a series of gaussians and lorentzians before applying it to the spectra appropriately.

        Parameters
        ----------
        libpath : string
            Path location of the .so or .dylib files. Ex: "/Users/cas003/opt/anaconda3/lib/python3.9/site-packages/BroadSword/"
            If compiling from source put the path of the install folder. 
            If using "pip install" make a note of where it installed the program and copy that directory path here.
        """

        Econd = np.zeros(40)
        type = False
        energy_0 = 20

        if XESbandScale == 0: # Applying a singular scale to XES
            for c1 in range(CalcSXSCase):
                for c2 in range(BandNum[c1]):
                    scaleXES[c1][c2] = XESscale
        else: # Applying scale to individual bands in XES
            for c1 in range(CalcSXSCase):
                for c2 in range(BandNum[c1]):
                    scaleXES[c1][c2] = XESbandScale[c1][c2]
        
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

        # Three different compilations of the .c file exist as c code has to be compiled based on the operating system that runs it.
        # TODO: There is currently no file that exists for windows OS
        try:
            mylib = cdll.LoadLibrary(libpath + "libmatrices.so")
        except OSError:
            try:
                mylib = cdll.LoadLibrary(libpath + "libmatrices_ARM64.dylib")
            except OSError:
                try:
                    mylib = cdll.LoadLibrary(libpath + "libmatrices_x86_64.dylib")
                except OSError:
                    try:
                        mylib = cdll.LoadLibrary(libpath)
                    except OSError as e:
                        print("Download the source and use the .c file to compile your own shared library and rename one of the existing .so or .dylib files.")
                        print("If compiling from source the pathname can include the filename. Ex: '/Users/cas003/opt/anaconda3/lib/python3.9/site-packages/BroadSword/MYLIBRARY.so' ")
                        print("No file currently exists for Windows OS (.dll).")
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
        for c1 in range(len(Edge)): # Convert the strings into integers to make it easier when transferring to the c program
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

        # Here we call the command to run program contained within the .c file
        mylib.broadXAS(cCalcSXSCase,cBroadSXSCount,BroadSXS,cdisord)
        mylib.add(cCalcSXSCase,cscalar,cEdge,cSite,BroadSXS,cBroadSXSCount,SumSXS,SumSXSCount)

        # Creating the figure for plotting the broadened data.
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

    def broaden2(self):
        """
        This will take the shifted calculated spectra and broaden it based on the lifetime, instrument, and general disorder broadening.
        It creates a series of gaussians and lorentzians before applying it to the spectra appropriately.

        Parameters
        ----------
        """

        Econd = np.zeros(40)
        type = False
        energy_0 = 20
        Pi = 3.14159265; # The Pi constant used for the distribution functions.

        if XESbandScale == 0: # Applying a singular scale to XES
            for c1 in range(CalcSXSCase):
                for c2 in range(BandNum[c1]):
                    scaleXES[c1][c2] = XESscale
        else: # Applying scale to individual bands in XES
            for c1 in range(CalcSXSCase):
                for c2 in range(BandNum[c1]):
                    scaleXES[c1][c2] = XESbandScale[c1][c2]
        
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

        # Creating the broadening matrices.
        for c1 in range(CalcSXSCase): # This is only for the XES spectra
            start = time.perf_counter()
            for c3 in range(BroadSXSCount[0][c1]): # Takes about 1 second to complete a full cycle of c3 * # of input files
                width = BroadSXS[4][c3][0][c1]/2.3548; # We extract the variance for the Gaussian Distribution
                position = BroadSXS[0][c3][0][c1]; # We extract the centroid of the Gaussian Distribution
                #print("This is position")
                #print(position)
                Gauss[c3,:] = np.reciprocal(np.sqrt(2*Pi*width*width))*np.exp(-(BroadSXS[0,:,0,c1]-position)*(BroadSXS[0,:,0,c1]-position)/2/width/width)
                #width = disord/2.3548; # We extract the variance for the Gaussian Distribution
                #position = BroadSXS[0][c3][0][c1]; # We extract the centroid of the Gaussian Distribution
                #Disorder[c3,:] = np.reciprocal(np.sqrt(2*Pi*width*width))*np.exp(-(BroadSXS[0,:,0,c1]-position)*(BroadSXS[0,:,0,c1]-position)/2/width/width)
                # Can I remove disorder? Since does not affect this???
                
                # Uncomment out the lorentz
                #width = BroadSXS[2][c3][0][c1]/2; # We extract the variance for the Gaussian Distribution
                #position = BroadSXS[0][c3][0][c1]; # We extract the centroid of the Gaussian Distribution
                #Lorentz[c3,:] = np.reciprocal(Pi)*(width/((BroadSXS[0,:,0,c1]-position)*(BroadSXS[0,:,0,c1]-position)+(width*width)))
            
            BroadSXS[3,:,0,c1] = 0 # Line 901
            for c3 in range(BroadSXSCount[0][c1]):
                BroadSXS[3,:,0,c1] = BroadSXS[3,:,0,c1]+(Lorentz[c3,:]*BroadSXS[1][c3][0][c1]*(BroadSXS[0][1][0][c1]-BroadSXS[0][0][0][c1]))
            
            BroadSXS[6,:,0,c1] = 0 # Line 912
            for c3 in range(BroadSXSCount[0][c1]):
                BroadSXS[6,:,0,c1] = BroadSXS[6,:,0,c1]+(Gauss[c3,:]*BroadSXS[3][c3][0][c1]*(BroadSXS[0][1][0][c1]-BroadSXS[0][0][0][c1]))

            #BroadSXS[6,:,0,c1] = 0 # Line 924 Originally commented out in C code because disorder does not impact XES.
            #for c4 in range(BroadSXSCount[0][c1]):
            #    BroadSXS[6,:,0,c1] = BroadSXS[6,:,0,c1]+(Disorder[c4,:]*BroadSXS[5][c4][0][c1]*(BroadSXS[0][1][0][c1]-BroadSXS[0][0][0][c1]))
            end = time.perf_counter()
            print(end-start)
            print("Hello")

        for c1 in range(CalcSXSCase): # Line 938
            for c2 in range(1,3): # TODO: Check if this is the right range. Want 1 and 2.
                start = time.perf_counter()
                for c3 in range(BroadSXSCount[c2][c1]): # Takes about 1 second to complete a full cycle of c3 * # of input files
                    width = BroadSXS[4][c3][c2][c1]/2.3548; # We extract the variance for the Gaussian Distribution
                    position = BroadSXS[0][c3][c2][c1]; # We extract the centroid of the Gaussian Distribution
                    Gauss[c3,:] = np.reciprocal(np.sqrt(2*Pi*width*width))*np.exp(-(BroadSXS[0,:,c2,c1]-position)*(BroadSXS[0,:,c2,c1]-position)/2/width/width)

                    width = disord/2.3548; # We extract the variance for the Gaussian Distribution
                    position = BroadSXS[0][c3][c2][c1]; # We extract the centroid of the Gaussian Distribution
                    Disorder[c3,:] = np.reciprocal(np.sqrt(2*Pi*width*width))*np.exp(-(BroadSXS[0,:,c2,c1]-position)*(BroadSXS[0,:,c2,c1]-position)/2/width/width)
                    
                    width = BroadSXS[2][c3][c2][c1]/2; # We extract the variance for the Gaussian Distribution
                    position = BroadSXS[0][c3][c2][c1]; # We extract the centroid of the Gaussian Distribution
                    Lorentz[c3,:] = np.reciprocal(Pi)*(width/((BroadSXS[0,:,c2,c1]-position)*(BroadSXS[0,:,c2,c1]-position)+(width*width)))
                
                BroadSXS[3,:,c2,c1] = 0 # Line 967
                for c3 in range(BroadSXSCount[c2][c1]):
                    BroadSXS[3,:,c2,c1] = BroadSXS[3,:,c2,c1]+(Lorentz[:,c3]*BroadSXS[1][c3][c2][c1]*(BroadSXS[0][1][c2][c1]-BroadSXS[0][0][c2][c1]))
                
                BroadSXS[5,:,c2,c1] = 0 # Line 978
                for c3 in range(BroadSXSCount[c2][c1]):
                    BroadSXS[5,:,c2,c1] = BroadSXS[5,:,c2,c1]+(Gauss[:,c3]*BroadSXS[3][c3][c2][c1]*(BroadSXS[0][1][c2][c1]-BroadSXS[0][0][c2][c1]))

                BroadSXS[6,:,c2,c1] = 0 # Line 990
                for c3 in range(BroadSXSCount[c2][c1]):
                    BroadSXS[6,:,c2,c1] = BroadSXS[6,:,c2,c1]+(Disorder[c3,:]*BroadSXS[5][c3][c2][c1]*(BroadSXS[0][1][c2][c1]-BroadSXS[0][0][c2][c1]))
                end = time.perf_counter()
                print(end-start)
                print("Hello2")

        # Creating the figure for plotting the broadened data.
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
    
    def broaden3(self):
        """
        This will take the shifted calculated spectra and broaden it based on the lifetime, instrument, and general disorder broadening.
        It creates a series of gaussians and lorentzians before applying it to the spectra appropriately.

        Parameters
        ----------
        """

        Econd = np.zeros(40)
        type = False
        energy_0 = 20
        Pi = 3.14159265; # The Pi constant used for the distribution functions.

        if XESbandScale == 0: # Applying a singular scale to XES
            for c1 in range(CalcSXSCase):
                for c2 in range(BandNum[c1]):
                    scaleXES[c1][c2] = XESscale
        else: # Applying scale to individual bands in XES
            for c1 in range(CalcSXSCase):
                for c2 in range(BandNum[c1]):
                    scaleXES[c1][c2] = XESbandScale[c1][c2]
        
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

        global placeholder
        global Gauss
        # Creating the broadening matrices.
        for c1 in range(CalcSXSCase): # This is only for the XES spectra
            start = time.perf_counter()
            for c3 in range(BroadSXSCount[0][c1]): # Takes about 1 second to complete a full cycle of c3 * # of input files
                placeholder[:,c3] = BroadSXS[0,:,0,c1] - BroadSXS[0][c3][0][c1]

            Gauss[:,:] = np.reciprocal(np.sqrt(2*Pi*(np.square(BroadSXS[4,:,0,c1]/2.3548))))*np.exp(-np.square(placeholder)/2/(np.square(BroadSXS[4,:,0,c1]/2.3548)))
            Gauss = np.transpose(Gauss)
            #Gauss[:,:] = np.reciprocal(np.sqrt(2*Pi*(BroadSXS[4,:,0,c1]/2.3548)*(BroadSXS[4,:,0,c1]/2.3548)))*np.exp(-(BroadSXS[0,:,0,c1]-BroadSXS[0,:,0,c1])*(BroadSXS[0,:,0,c1]-BroadSXS[0,:,0,c1])/2/(BroadSXS[4,:,0,c1]/2.3548)/(BroadSXS[4,:,0,c1]/2.3548))
            #Lorentz[:,:] = np.reciprocal(Pi)*((BroadSXS[2,:,0,c1]/2)/((BroadSXS[0,:,0,c1]-BroadSXS[0,:,0,c1])*(BroadSXS[0,:,0,c1]-BroadSXS[0,:,0,c1])+((BroadSXS[2,:,0,c1]/2)*(BroadSXS[2,:,0,c1]/2))))
            # These get the diagonal elements and broadcast them across everywhere. Close, but not quite.
            # So the position is subtracting down to 0 every time. Which means that we probably have to go and make it diagonal or something.
            # Basically so that a single row will only subtract from one position, instead of all positions sequentially.
            # I think this is doable.
            
            BroadSXS[3,:,0,c1] = 0 # Line 901
            for c3 in range(BroadSXSCount[0][c1]):
                BroadSXS[3,:,0,c1] = BroadSXS[3,:,0,c1]+(Lorentz[c3,:]*BroadSXS[1][c3][0][c1]*(BroadSXS[0][1][0][c1]-BroadSXS[0][0][0][c1]))
            
            BroadSXS[6,:,0,c1] = 0 # Line 912
            for c3 in range(BroadSXSCount[0][c1]):
                BroadSXS[6,:,0,c1] = BroadSXS[6,:,0,c1]+(Gauss[c3,:]*BroadSXS[3][c3][0][c1]*(BroadSXS[0][1][0][c1]-BroadSXS[0][0][0][c1]))

            #BroadSXS[6,:,0,c1] = 0 # Line 924 Originally commented out in C code because disorder does not impact XES.
            #for c4 in range(BroadSXSCount[0][c1]):
            #    BroadSXS[6,:,0,c1] = BroadSXS[6,:,0,c1]+(Disorder[c4,:]*BroadSXS[5][c4][0][c1]*(BroadSXS[0][1][0][c1]-BroadSXS[0][0][0][c1]))
            end = time.perf_counter()
            print(end-start)
            print("Hello3")

        for c1 in range(CalcSXSCase): # Line 938
            for c2 in range(1,3): # TODO: Check if this is the right range. Want 1 and 2.
                start = time.perf_counter()
                for c3 in range(BroadSXSCount[c2][c1]): # Takes about 1 second to complete a full cycle of c3 * # of input files
                    width = BroadSXS[4][c3][c2][c1]/2.3548; # We extract the variance for the Gaussian Distribution
                    position = BroadSXS[0][c3][c2][c1]; # We extract the centroid of the Gaussian Distribution
                    Gauss[c3,:] = np.reciprocal(np.sqrt(2*Pi*width*width))*np.exp(-(BroadSXS[0,:,c2,c1]-position)*(BroadSXS[0,:,c2,c1]-position)/2/width/width)

                    width = disord/2.3548; # We extract the variance for the Gaussian Distribution
                    position = BroadSXS[0][c3][c2][c1]; # We extract the centroid of the Gaussian Distribution
                    Disorder[c3,:] = np.reciprocal(np.sqrt(2*Pi*width*width))*np.exp(-(BroadSXS[0,:,c2,c1]-position)*(BroadSXS[0,:,c2,c1]-position)/2/width/width)
                    
                    width = BroadSXS[2][c3][c2][c1]/2; # We extract the variance for the Gaussian Distribution
                    position = BroadSXS[0][c3][c2][c1]; # We extract the centroid of the Gaussian Distribution
                    Lorentz[c3,:] = np.reciprocal(Pi)*(width/((BroadSXS[0,:,c2,c1]-position)*(BroadSXS[0,:,c2,c1]-position)+(width*width)))
                
                BroadSXS[3,:,c2,c1] = 0 # Line 967
                for c3 in range(BroadSXSCount[c2][c1]):
                    BroadSXS[3,:,c2,c1] = BroadSXS[3,:,c2,c1]+(Lorentz[:,c3]*BroadSXS[1][c3][c2][c1]*(BroadSXS[0][1][c2][c1]-BroadSXS[0][0][c2][c1]))
                
                BroadSXS[5,:,c2,c1] = 0 # Line 978
                for c3 in range(BroadSXSCount[c2][c1]):
                    BroadSXS[5,:,c2,c1] = BroadSXS[5,:,c2,c1]+(Gauss[:,c3]*BroadSXS[3][c3][c2][c1]*(BroadSXS[0][1][c2][c1]-BroadSXS[0][0][c2][c1]))

                BroadSXS[6,:,c2,c1] = 0 # Line 990
                for c3 in range(BroadSXSCount[c2][c1]):
                    BroadSXS[6,:,c2,c1] = BroadSXS[6,:,c2,c1]+(Disorder[c3,:]*BroadSXS[5][c3][c2][c1]*(BroadSXS[0][1][c2][c1]-BroadSXS[0][0][c2][c1]))
                end = time.perf_counter()
                print(end-start)
                print("Hello2")

        # Creating the figure for plotting the broadened data.
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

    def initResolution(self, corelifetime, specResolution, monoResolution, disorder, XESscaling, XASscaling, XESbandScaling=0):
        """
        Specify the parameters for the broadening criteria.

        Parameters
        ----------
        XEScorelife : float
            Specify the corehole lifetime broadening factor
        specResolution : float
            Specify spectrometer resolving power
        monoResolution : float
            Specify monochromator resolving power
        disorder : float
            Specify general disorder factor in the sample
        XESscaling : float
            Specify corehole lifetime scaling factor for XES
        XASscaling : float
            Specify corehole lifetime scaling factor for XAS
        XESbandScaling : [float]
        """
        global corelifeXES # A terrible way to do this, but it works.
        global corelifeXAS
        global spec
        global mono
        global disord
        global XESscale
        global scaleXAS
        global XESbandScale
        corelifeXES = corelifetime
        corelifeXAS = corelifetime
        spec = specResolution
        mono = monoResolution
        disord = disorder
        XESscale = XESscaling
        scaleXAS = XASscaling
        XESbandScale = XESbandScaling
        return

    def plotExp(self,p):
        """
        Plot the measured experimental data.
        The bokeh figure needs to be created and configured outside of the function. This simply adds the XANES and XES to a figure.

        Parameters
        ----------
        p : figure()
            The bokeh figure needs to be created outside of the function.
        """
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
        """
        Plot the shifted calculated data.
        The bokeh figure needs to be created and configured outside of the function. This simply adds the XANES and XES to a figure.

        Parameters
        ----------
        p : figure()
            The bokeh figure needs to be created outside of the function.
        """

        MaxCalcSXS = np.zeros([3,40]) # Find the maximum value in the spectra to normalize it for plotting.
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
                calcxesY[c2] = BroadSXS[1][c2][0][c1] / (MaxCalcSXS[0][c1])
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
        """
        Plot the unshifted calculated data. This is purely the raw data read from .loadCalc()
        """
        p = figure()
        for c1 in range(CalcSXSCase): # Since this is np array you can use : to get all data points
            colour = COLORP[c1]
            p.line(CalcSXS[0,:,0,c1], CalcSXS[1,:,0,c1],line_color=colour) # XES plot
            p.line(CalcSXS[0,:,1,c1], CalcSXS[1,:,1,c1],line_color=colour) # XAS plot
            p.line(CalcSXS[0,:,2,c1], CalcSXS[1,:,2,c1],line_color=colour) # XANES plot
        show(p)
        return

    def plotBroadCalc(self,p):
        """
        Plot the final calculated and broadened data.
        The bokeh figure needs to be created and configured outside of the function. This simply adds the XANES, XAS, and XES to a figure.

        Parameters
        ----------
        p : figure()
            The bokeh figure needs to be created outside of the function.
        """
        MaxBroadSXS = np.zeros([3])
        for c3 in range(3): # Find the maximum value for normalization
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
        Export and write data to the specified files.
        This will export only the broadened data. This data has not been normalized however.

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
