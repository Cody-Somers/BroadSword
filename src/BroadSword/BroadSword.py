
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

# TODO: Replace these all with a self.var so that they are not globals
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
    First: Load the experimental. 
    Second: Load all of the calculations sequentially. 
    Third: Generate the parameters used for shifting and broadening. Finally: Broaden the spectra.
    """

    def __init__(self):
        self.data = list() # Why is this here?

    def loadExp(self, basedir, XES, XANES, GS_fermi, headerlines=[0,0]):
        """
        Loads the experimental data.

        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        XES, XANES : string
            Specify the file name (ASCII or .csv) including the extension.
            Header lines are allowed, as long as they are specified properly in the function.
        GS_fermi : float
            Specify the fermi energy for the ground state calculated spectra. Found in .scf2
        headerlines : [int]
            Specify the number of headerlines for the XES and XANES files respectively. 
        """

        try:
            with open(basedir+"/"+XES, "r") as xesFile: # Measured XES
                df = pd.read_csv(xesFile, delimiter='\s+', header=None, skiprows=headerlines[0]) # Change to '\s*' and specify engine='python' if this breaks in jupyter notebook
                c1 = 0
                maxEXP = 0
                for i in range(len(df)): 
                    ExpSXS[0][c1][0] = df[0][c1] # Energy
                    ExpSXS[1][c1][0] = df[1][c1] # Counts
                    if ExpSXS[1][c1][0] > maxEXP:
                        maxEXP = ExpSXS[1][c1][0] # Get max value in experimental XES
                    c1 += 1
                ExpSXSCount[0] = c1 # Length of data points
                for i in range(ExpSXSCount[0]): # Normalize spectra
                    ExpSXS[1][i][0] = ExpSXS[1][i][0]/maxEXP
        except:
            with open(basedir+"/"+XES, "r") as xesFile: # Measured XES
                # This trys it as a .csv instead of a .txt
                df = pd.read_csv(xesFile, header=None, skiprows=headerlines[0]) # Change to '\s*' and specify engine='python' if this breaks in jupyter notebook                
                c1 = 0
                maxEXP = 0
                for i in range(len(df)): 
                    ExpSXS[0][c1][0] = df[0][c1] # Energy
                    ExpSXS[1][c1][0] = df[1][c1] # Counts
                    if ExpSXS[1][c1][0] > maxEXP:
                        maxEXP = ExpSXS[1][c1][0] # Get max value in experimental XES
                    c1 += 1
                ExpSXSCount[0] = c1 # Length of data points
                for i in range(ExpSXSCount[0]): # Normalize spectra
                    ExpSXS[1][i][0] = ExpSXS[1][i][0]/maxEXP
        try:
            with open(basedir+"/"+XANES, "r") as xanesFile: # Measured XANES
                df = pd.read_csv(xanesFile, delimiter='\s+', header=None, skiprows=headerlines[1])
                c1 = 0
                for i in range(len(df)):
                    ExpSXS[0][c1][1] = df[0][c1] # Energy
                    ExpSXS[1][c1][1] = df[1][c1] # Counts
                    c1 += 1
                ExpSXSCount[1] = c1 # Length of data points
        except:
            with open(basedir+"/"+XANES, "r") as xanesFile: # Measured XANES
                # This trys it as a .csv instead of a .txt
                df = pd.read_csv(xanesFile, header=None, skiprows=headerlines[1])
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
        Fermi = GS_fermi
        return

    def loadCalc(self, basedir, XES, XAS, GS_bindingEnergy, XANES=0, ES_fermi=0, edge="K", sites=1, headerlines=[0,0,0]):
        """
        Loads the calculated data.
        
        Parameters
        ----------
        basedir : string
            Specifiy the absolute or relative path to experimental data.
        XES, XAS, XANES : string
            Specify the file name including the extension (.txspec).
        GS_bindingEnergy : float
            Specify the binding energy of the ground state. Found in .scfc
        ES_fermi : float
            Specify the fermi energy for the excited state calculation. Found in .scf2
        edge : string
            Specify the excitation edge "K","L2","L3","M4","M5".
        sites : float
            Specify the number of atomic positions present in the inequivalent site.
        headerlines : [int]
            Specify the number of headerlines for the XES and XANES files respectively. 
        """
        global CalcSXSCase
        global Fermi

        if XANES == 0:
            XANES = XAS # For when no core hole exists
            ES_fermi = Fermi # Make the fermi level equal to the ground state.

        with open(basedir+"/"+XES, "r") as xesFile: # XES Calculation
            df = pd.read_csv(xesFile, delimiter='\s+',header=None, skiprows=headerlines[0])
            c1 = 0
            for i in range(len(df)):
                CalcSXS[0][c1][0][CalcSXSCase] = df[0][c1] # Energy
                CalcSXS[1][c1][0][CalcSXSCase] = df[1][c1] # Counts
                c1 += 1
            CalcSXSCount[0][CalcSXSCase] = c1 # Length for each Site

        with open(basedir+"/"+XAS, "r") as xasFile: # XAS Calculation
            df = pd.read_csv(xasFile, delimiter='\s+',header=None, skiprows=headerlines[1])
            c1 = 0
            for i in range(len(df)):
                CalcSXS[0][c1][1][CalcSXSCase] = df[0][c1] # Energy
                CalcSXS[1][c1][1][CalcSXSCase] = df[1][c1] # Counts
                c1 += 1
            CalcSXSCount[1][CalcSXSCase] = c1 # Length for each Site

        with open(basedir+"/"+XANES, "r") as xanesFile: # XANES Calculation
            df = pd.read_csv(xanesFile, delimiter='\s+',header=None, skiprows=headerlines[2])
            c1 = 0
            for i in range(len(df)):
                CalcSXS[0][c1][2][CalcSXSCase] = df[0][c1] # Energy
                CalcSXS[1][c1][2][CalcSXSCase] = df[1][c1] # Counts
                c1 += 1
            CalcSXSCount[2][CalcSXSCase] = c1 # Length for each Site

        # Update the global variables with the parameters for that site.
        Fermis[CalcSXSCase] = ES_fermi
        Binds[CalcSXSCase] = GS_bindingEnergy
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
    
    def Shift(self,XESshift, XASshift, XESbandshift=0, separate=False):
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
        separate : True/False
            Specify whether or not to create a separate output plot of XES and XAS
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

        if separate is False:
            # Creating the figure for plotting the broadened data.
            p = figure(height=450, width=900, title="Un-Broadened Data", x_axis_label="Energy (eV)", y_axis_label="Normalized Intensity (arb. units)",
                    tools="pan,wheel_zoom,box_zoom,reset,crosshair,save")
            p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
                ("(x,y)", "(Energy, Intensity)"),
                ("(x,y)", "($x, $y)")
            ]))
            self.plotShiftCalc(p)
            self.plotExp(p)
            p.add_layout(p.legend[0], 'right')
            show(p)
        else:
            p = figure(height=450, width=900, title="Un-Broadened Data", x_axis_label="Energy (eV)", y_axis_label="Normalized Intensity (arb. units)",
                    tools="pan,wheel_zoom,box_zoom,reset,crosshair,save")
            p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
                ("(x,y)", "(Energy, Intensity)"),
                ("(x,y)", "($x, $y)")
            ]))
            self.plotExpXES(p)
            self.plotShiftXES(p)
            p.add_layout(p.legend[0], 'right')
            show(p)

            p = figure(height=450, width=900, title="Un-Broadened Data", x_axis_label="Energy (eV)", y_axis_label="Normalized Intensity (arb. units)",
                    tools="pan,wheel_zoom,box_zoom,reset,crosshair,save")
            p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
                ("(x,y)", "(Energy, Intensity)"),
                ("(x,y)", "($x, $y)")
            ]))
            self.plotExpXANES(p)
            self.plotShiftXANES(p)
            p.add_layout(p.legend[0], 'right')
            show(p)
        
        return

    def broadenC(self, libpath="./"):
        """
        This will take the shifted calculated spectra and broaden it based on the lifetime, instrument, and general disorder broadening.
        It creates a series of gaussians and lorentzians before applying it to the spectra appropriately.
        Depreciated Function: Currently not usable, but can easily be reinstated by changing the globals back to ctypes.

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
        p = figure(height=450, width=900, title="Broadened Data", x_axis_label="Energy (eV)", y_axis_label="Normalized Intensity (arb. units)",
                   tools="pan,wheel_zoom,box_zoom,reset,crosshair,save")
        p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
            ("(x,y)", "(Energy, Intensity)"),
            ("(x,y)", "($x, $y)")
        ]))
        self.plotBroadCalc(p)
        self.plotExp(p)
        show(p)
        return

    def broaden(self,separate=True):
        """
        This will take the shifted calculated spectra and broaden it based on the lifetime, instrument, and general disorder broadening.
        It creates a series of gaussians and lorentzians before applying it to the spectra appropriately.

        Parameters
        ----------
        separate : True/False
            Specify whether or not to create a separate output plot of XES and XAS
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
            for c3 in range(BroadSXSCount[0][c1]): # Takes about 1 second to complete a full cycle of c3 * # of input files
                width = BroadSXS[4][c3][0][c1]/2.3548; # We extract the variance for the Gaussian Distribution
                position = BroadSXS[0][c3][0][c1]; # We extract the centroid of the Gaussian Distribution
                Gauss[c3,:] = np.reciprocal(np.sqrt(2*Pi*width*width))*np.exp(-(BroadSXS[0,:,0,c1]-position)*(BroadSXS[0,:,0,c1]-position)/2/width/width)

                #  Commented out since disorder does not affect XES
                #width = disord/2.3548; # We extract the variance for the Gaussian Distribution
                #position = BroadSXS[0][c3][0][c1]; # We extract the centroid of the Gaussian Distribution
                #Disorder[c3,:] = np.reciprocal(np.sqrt(2*Pi*width*width))*np.exp(-(BroadSXS[0,:,0,c1]-position)*(BroadSXS[0,:,0,c1]-position)/2/width/width)

                width = BroadSXS[2][c3][0][c1]/2; # We extract the variance for the Gaussian Distribution
                position = BroadSXS[0][c3][0][c1]; # We extract the centroid of the Gaussian Distribution
                Lorentz[c3,:] = np.reciprocal(Pi)*(width/((BroadSXS[0,:,0,c1]-position)*(BroadSXS[0,:,0,c1]-position)+(width*width)))
            
            BroadSXS[3,:,0,c1] = 0 # Line 901
            for c3 in range(BroadSXSCount[0][c1]):
                BroadSXS[3,:,0,c1] = BroadSXS[3,:,0,c1]+(Lorentz[c3,:]*BroadSXS[1][c3][0][c1]*(BroadSXS[0][1][0][c1]-BroadSXS[0][0][0][c1]))
            
            BroadSXS[6,:,0,c1] = 0 # Line 912
            for c3 in range(BroadSXSCount[0][c1]):
                BroadSXS[6,:,0,c1] = BroadSXS[6,:,0,c1]+(Gauss[c3,:]*BroadSXS[3][c3][0][c1]*(BroadSXS[0][1][0][c1]-BroadSXS[0][0][0][c1]))

            #BroadSXS[6,:,0,c1] = 0 # Line 924 Originally commented out in C code because disorder does not impact XES.
            #for c4 in range(BroadSXSCount[0][c1]):
            #    BroadSXS[6,:,0,c1] = BroadSXS[6,:,0,c1]+(Disorder[c4,:]*BroadSXS[5][c4][0][c1]*(BroadSXS[0][1][0][c1]-BroadSXS[0][0][0][c1]))


        for c1 in range(CalcSXSCase): # Line 938
            for c2 in range(1,3):
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
        self.add()

        if separate is False:
            # Creating the figure for plotting the broadened data.
            p = figure(height=450, width=900, title="Broadened Data", x_axis_label="Energy (eV)", y_axis_label="Normalized Intensity (arb. units)",
                    tools="pan,wheel_zoom,box_zoom,reset,crosshair,save")
            p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
                ("(x,y)", "(Energy, Intensity)"),
                ("(x,y)", "($x, $y)")
            ]))
            self.plotBroadCalc(p)
            self.plotExp(p)
            p.add_layout(p.legend[0], 'right')
            show(p)
        else:
            p = figure(height=450, width=900, title="Broadened Data", x_axis_label="Energy (eV)", y_axis_label="Normalized Intensity (arb. units)",
                    tools="pan,wheel_zoom,box_zoom,reset,crosshair,save")
            p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
                ("(x,y)", "(Energy, Intensity)"),
                ("(x,y)", "($x, $y)")
            ]))
            self.plotExpXES(p)
            self.plotBroadXES(p)
            p.add_layout(p.legend[0], 'right')
            show(p)

            p = figure(height=450, width=900, title="Broadened Data", x_axis_label="Energy (eV)", y_axis_label="Normalized Intensity (arb. units)",
                    tools="pan,wheel_zoom,box_zoom,reset,crosshair,save")
            p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
                ("(x,y)", "(Energy, Intensity)"),
                ("(x,y)", "($x, $y)")
            ]))
            self.plotExpXANES(p)
            self.plotBroadXANES(p)
            p.add_layout(p.legend[0], 'right')
            show(p)

        return

    def add(self):
        """
        A function that will sum together the individual inequivalent sites after they have been broadened with the matrices above.
        Scales the sites as appropriate, then does a linear interpolation of each data point to sum together different sites.
        """
        Edge_check = ["K","L2","L3","M4","M5"]
        Edge_scale = [1,0.3333333,0.6666667,0.4,0.6]
        max = 0
        for c1 in range(CalcSXSCase): # Determine the relative addition scale factor XES
            for c2 in range(3):
                scalar[c2][c1]=1

        for c1 in range(CalcSXSCase): # Apply the scaling to the running scalar
            for c2 in range(5): # Counts through the types of edges
                if Edge[c1] == Edge_check[c2]:
                    for c3 in range(3):
                        scalar[c3][c1] = scalar[c3][c1]*Site[c1]*Edge_scale[c2]

        statement = 0 # Print statement tracker

        for c1 in range(3):
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
                    c4 = 0
                    for c3 in range(SumSXSCount[c1]):
                        c4 = c3 - 5 # This speeds up the program significantly by not starting at 0 every time.
                        # It will work as long as the interpolated data point is within -5 x values. Can be as far forward as neccesary
                        # Based on the criteria though, it should never be a negative point.
                        if c4 < 0:
                            c4 = 0
                        if BroadSXS[0][c4][c1][c2] > SumSXS[0][c3][c1] and c3 != 0:
                            c4 = c3 - 50 # We try again with a larger range to start out with.
                            if c4 < 0:
                                c4 = 0
                            if statement == 0:
                                print("Report this to cas003@usask.ca") # I want to know if it is possible for this point to be reached.
                                statement = 1
                            if BroadSXS[0][c4][c1][c2] > SumSXS[0][c3][c1]:
                                c4 = c3 - 500 # Try once again with a larger range.
                                if c4 < 0:
                                    c4 = 0
                                if BroadSXS[0][c4][c1][c2] > SumSXS[0][c3][c1]:
                                    c4 = 0 # This just ensures that if it is as far back as allowed, it will instead start at 0 to go through all values
                                    # This would slow down the program, but only in cases where necessary.
                                    if statement == 1:
                                        print("Report this to cas003@usask.ca and attach the txspec files used in the Jupyter Notebook")
                                        print("The broadening will take several minutes. To avoid this, try making all of the .txspec files the same length. (-2 to 50eV for example)")
                                        statement = 2
                        
                        while c4 < BroadSXSCount[c1][c2]:
                            if BroadSXS[0][c4][c1][c2] > SumSXS[0][c3][c1]:
                                x1 = BroadSXS[0][c4-1][c1][c2]
                                x2 = BroadSXS[0][c4][c1][c2]
                                y1 = BroadSXS[6][c4-1][c1][c2]
                                y2 = BroadSXS[6][c4][c1][c2]
                                slope = (y2-y1)/(x2-x1)
                                SumSXS[1][c3][c1] = SumSXS[1][c3][c1] + scalar[c1][c2]*(slope*(SumSXS[0][c3][c1]-x1)+y1)
                                max = c3
                                c4 = 9999999
                            c4 += 1
                    SumSXSCount[c1] = max
        return

    def initResolution(self, corelifetime, specResolution, monoResolution, disorder, XESscaling, XASscaling, XESbandScaling=0):
        """
        Specify the parameters for the broadening criteria.

        Parameters
        ----------
        XEScorelife : float
            Specify the corehole lifetime broadening factor in eV. https://xpslibrary.com/core-hole-lifetimes-fwhm/ has examples for several gasses.
        specResolution : float
            Specify spectrometer resolving power. Dictates the instrumental broadening of the spectra.
        monoResolution : float
            Specify monochromator resolving power. Dictates the instrumental broadening of the spectra.
        disorder : float
            Specify general disorder factor in the sample. Only affects XAS/XANES
        XESscaling : float
            Specify corehole lifetime scaling factor for XES. Will scale the lifetime parabola to broaden more aggresively away from the onset.
        XASscaling : float
            Specify corehole lifetime scaling factor for XAS. Will scale the lifetime parabola to broaden more aggresively away from the onset.
        XESbandScaling : [float]
            Specify corehole lifetime scaling factor for each of the bands separately in XES
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
        p.line(xanesX,xanesY,line_color="red",legend_label="Experimental XES/XANES") # XANES plot
        p.line(xesX,xesY,line_color="red") # XES plot
        #show(p)
        return

    def plotExpXES(self,p):
        """
        Plot the measured experimental XES data.
        The bokeh figure needs to be created and configured outside of the function. This simply adds the XES to a figure.

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
        
        p.line(xesX,xesY,line_color="red",legend_label="Experimental XES") # XES plot
        return
    
    def plotExpXANES(self,p):
        """
        Plot the measured experimental XANES data.
        The bokeh figure needs to be created and configured outside of the function. This simply adds the XANES to a figure.

        Parameters
        ----------
        p : figure()
            The bokeh figure needs to be created outside of the function.
        """
        xanesX = np.zeros([ExpSXSCount[1]])
        xanesY = np.zeros([ExpSXSCount[1]])
        
        for c1 in range(ExpSXSCount[1]): # Experimental xanes spectra
            xanesX[c1] = ExpSXS[0][c1][1]
            xanesY[c1] = ExpSXS[1][c1][1]
        
        p.line(xanesX,xanesY,line_color="red",legend_label="Experimental XANES") # XANES plot
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
    
    def plotShiftXES(self,p):
        """
        Plot the shifted calculated XES data.
        The bokeh figure needs to be created and configured outside of the function. This simply adds the XES to a figure.

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

            for c2 in range(CalcSXSCount[0][c1]): # Calculated XES spectra
                calcxesX[c2] = BroadSXS[0][c2][0][c1]
                calcxesY[c2] = BroadSXS[1][c2][0][c1] / (MaxCalcSXS[0][c1])
                #y = (x - x_min) / (x_max - x_min) Where x_min = 0
            colour = COLORP[c1]

            if colour == "#d60000": # So that there are no red spectra since the experimental is red
                colour = "Magenta"
                
            p.line(calcxesX,calcxesY,line_color=colour) # XES plot
        #show(p)
        return
    
    def plotShiftXANES(self,p):
        """
        Plot the shifted calculated XANES data.
        The bokeh figure needs to be created and configured outside of the function. This simply adds the XANES to a figure.

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
            calcxasX = np.zeros([CalcSXSCount[1][c1]])
            calcxasY = np.zeros([CalcSXSCount[1][c1]])
            calcxanesX = np.zeros([CalcSXSCount[2][c1]])
            calcxanesY = np.zeros([CalcSXSCount[2][c1]])

            for c2 in range(CalcSXSCount[1][c1]): # Calculated XAS spectra
                calcxasX[c2] = BroadSXS[0][c2][1][c1]
                calcxasY[c2] = BroadSXS[1][c2][1][c1] / (MaxCalcSXS[1][c1])

            for c2 in range(CalcSXSCount[2][c1]): # Calculated XANES spectra
                calcxanesX[c2] = BroadSXS[0][c2][2][c1]
                calcxanesY[c2] = BroadSXS[1][c2][2][c1] / (MaxCalcSXS[2][c1])
            colour = COLORP[c1]

            if colour == "#d60000": # So that there are no red spectra since the experimental is red
                colour = "Magenta"
            
            # p.line(calcxasX,calcxasY,line_color=colour) # XAS plot is not needed for lining up the spectra. Use XANES
            p.line(calcxanesX,calcxanesY,line_color=colour) # XANES plot
        #show(p)
        return

    def plotCalc(self):
        """
        Plot the unshifted calculated data. This is purely the raw data read from .loadCalc()
        """
        p = figure()

        MaxCalcSXS = np.zeros([3,40]) # Find the maximum value in the spectra to normalize it for plotting.
        for c1 in range(CalcSXSCase):
            for c3 in range(3):
                for c2 in range(CalcSXSCount[c3][c1]):
                    if MaxCalcSXS[c3][c1] < CalcSXS[1][c2][c3][c1]:
                        MaxCalcSXS[c3][c1] = CalcSXS[1][c2][c3][c1]
        for c1 in range(CalcSXSCase): # Since this is np array you can use : to get all data points
            calcxesX = np.zeros([CalcSXSCount[0][c1]])
            calcxesY = np.zeros([CalcSXSCount[0][c1]])

            for c2 in range(CalcSXSCount[0][c1]): # Calculated XES spectra
                calcxesX[c2] = CalcSXS[0][c2][0][c1]
                calcxesY[c2] = CalcSXS[1][c2][0][c1] / (MaxCalcSXS[0][c1])
                #y = (x - x_min) / (x_max - x_min) Where x_min = 0
            colour = COLORP[c1]
                
            p.line(calcxesX,calcxesY,line_color=colour) # XES plot
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

        p.line(sumxesX,sumxesY,line_color="limegreen",legend_label="Broadened XES/XANES") # XES plot
        p.line(sumxasX,sumxasY,line_color="blue",legend_label="Broadened XAS") # XAS plot
        p.line(sumxanesX,sumxanesY,line_color="limegreen") # XANES plot
        #show(p)
        return
    
    def plotBroadXANES(self,p):
        """
        Plot the final calculated and broadened data for XAS and XANES
        The bokeh figure needs to be created and configured outside of the function. This simply adds the XANES and XAS to a figure.

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
        sumxasX = np.zeros([SumSXSCount[1]])
        sumxasY = np.zeros([SumSXSCount[1]])
        sumxanesX = np.zeros([SumSXSCount[2]])
        sumxanesY = np.zeros([SumSXSCount[2]])

        for c2 in range(SumSXSCount[1]): # Calculated XAS spectra
            sumxasX[c2] = SumSXS[0][c2][1]
            sumxasY[c2] = SumSXS[1][c2][1] / MaxBroadSXS[1]

        for c2 in range(SumSXSCount[2]): # Calculated XANES spectra
            sumxanesX[c2] = SumSXS[0][c2][2]
            sumxanesY[c2] = SumSXS[1][c2][2] / MaxBroadSXS[2]

        p.line(sumxasX,sumxasY,line_color="blue",legend_label="Broadened XAS") # XAS plot
        p.line(sumxanesX,sumxanesY,line_color="limegreen",legend_label="Broadened XANES") # XANES plot
        return

    def plotBroadXES(self,p):
        """
        Plot the final calculated and broadened data for XES
        The bokeh figure needs to be created and configured outside of the function. This simply adds the XES to a figure.

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
        
        sumxesX = np.zeros([SumSXSCount[0]])
        sumxesY = np.zeros([SumSXSCount[0]])
        for c2 in range(SumSXSCount[0]): # Calculated XES spectra
            sumxesX[c2] = SumSXS[0][c2][0]
            sumxesY[c2] = SumSXS[1][c2][0] / MaxBroadSXS[0]

        p.line(sumxesX,sumxesY,line_color="limegreen",legend_label="Broadened XES") # XES plot
        return

    def export(self, filename, element, individual=False):
        """
        Export and write data to the specified files.
        This will export only the broadened data. This data has not been normalized however.

        Parameters
        ----------
        filename : string
            Specify the desired filename. Usually the compound name or molecular formula.
        element : string
            The edge of the excited element 
        individual : True/False
            Specify whether to export the individual inequivalent sites, or only the broadened sum.
        """

        with open(f"{filename}_{element}_XES.csv", 'w', newline='') as f:
            writer = csv.writer(f,delimiter=",") # TODO: Check that this actually makes a new column in the output.
            writer.writerow(["Energy",element+"_XES"])
            for c1 in range(SumSXSCount[0]):
                writer.writerow([SumSXS[0][c1][0],SumSXS[1][c1][0]])

        with open(f"{filename}_{element}_XAS.csv", 'w', newline='') as f:
            writer = csv.writer(f,delimiter=",")
            writer.writerow(["Energy",element+"_XAS"])
            for c1 in range(SumSXSCount[1]):
                writer.writerow([SumSXS[0][c1][1],SumSXS[1][c1][1]])

        with open(f"{filename}_{element}_XANES.csv", 'w', newline='') as f:
            writer = csv.writer(f,delimiter=",")
            writer.writerow(["Energy",element+"_XANES"])
            for c1 in range(SumSXSCount[2]):
                writer.writerow([SumSXS[0][c1][2],SumSXS[1][c1][2]])
        if individual is True:
            for c2 in range(CalcSXSCase):
                with open(f"{filename}_{element}"+str(c2+1)+"_XES.csv", 'w', newline='') as f:
                    writer = csv.writer(f,delimiter=",")
                    writer.writerow(["Energy",element+str(c2+1)+"_XES"])
                    for c1 in range(BroadSXSCount[0][c2]):
                        writer.writerow([BroadSXS[0][c1][0][c2],BroadSXS[6][c1][0][c2]])

                with open(f"{filename}_{element}"+str(c2+1)+"_XAS.csv", 'w', newline='') as f:
                    writer = csv.writer(f,delimiter=",")
                    writer.writerow(["Energy",element+str(c2+1)+"_XAS"])
                    for c1 in range(BroadSXSCount[1][c2]):
                        writer.writerow([BroadSXS[0][c1][1][c2],BroadSXS[6][c1][1][c2]])

                with open(f"{filename}_{element}"+str(c2+1)+"_XANES.csv", 'w', newline='') as f:
                    writer = csv.writer(f,delimiter=",")
                    writer.writerow(["Energy",element+str(c2+1)+"_XANES"])
                    for c1 in range(BroadSXSCount[2][c2]):
                        writer.writerow([BroadSXS[0][c1][2][c2],BroadSXS[6][c1][2][c2]])


        print(f"Successfully wrote DataFrame to {filename}.csv")
