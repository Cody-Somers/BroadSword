
import numpy as np
import pandas as pd
import csv

# Plotting
from bokeh.plotting import show, figure
from bokeh.models import HoverTool
COLORP = ['#d60000', '#8c3bff', '#018700', '#00acc6', '#e6a500', '#ff7ed1', '#6b004f', '#573b00', '#005659', '#15e18c', '#0000dd', '#a17569', '#bcb6ff', '#bf03b8', '#645472', '#790000', '#0774d8', '#729a7c', '#ff7752', '#004b00', '#8e7b01', '#f2007b', '#8eba00', '#a57bb8', '#5901a3', '#e2afaf', '#a03a52', '#a1c8c8', '#9e4b00', '#546744', '#bac389', '#5e7b87',
          '#60383b', '#8287ff', '#380000', '#e252ff', '#2f5282', '#7ecaff', '#c4668e', '#008069', '#919eb6', '#cc7407', '#7e2a8e', '#00bda3', '#2db152', '#4d33ff', '#00e400', '#ff00cd', '#c85748', '#e49cff', '#1ca1ff', '#6e70aa', '#c89a69', '#77563b', '#03dae6', '#c1a3c3', '#ff6989', '#ba00fd', '#915280', '#9e0174', '#93a14f', '#364424', '#af6dff', '#596d00',
          '#ff3146', '#828056', '#006d2d', '#8956af', '#5949a3', '#773416', '#85c39a', '#5e1123', '#d48580', '#a32818', '#0087b1', '#ca0044', '#ffa056', '#eb4d00', '#6b9700', '#528549', '#755900', '#c8c33f', '#91d370', '#4b9793', '#4d230c', '#60345b', '#8300cf', '#8a0031', '#9e6e31', '#ac8399', '#c63189', '#015438', '#086b83', '#87a8eb', '#6466ef', '#c35dba',
          '#019e70', '#805059', '#826e8c', '#b3bfda', '#b89028', '#ff97b1', '#a793e1', '#698cbd', '#4b4f01', '#4801cc', '#60006e', '#446966', '#9c5642', '#7bacb5', '#cd83bc', '#0054c1', '#7b2f4f', '#fb7c00', '#34bf00', '#ff9c87', '#e1b669', '#526077', '#5b3a7c', '#eda5da', '#ef52a3', '#5d7e69', '#c3774f', '#d14867', '#6e00eb', '#1f3400', '#c14103', '#6dd4c1',
          '#46709e', '#a101c3', '#0a8289', '#afa501', '#a55b6b', '#fd77ff', '#8a85ae', '#c67ee8', '#9aaa85', '#876bd8', '#01baf6', '#af5dd1', '#59502a', '#b5005e', '#7cb569', '#4985ff', '#00c182', '#d195aa', '#a34ba8', '#e205e2', '#16a300', '#382d00', '#832f33', '#5d95aa', '#590f00', '#7b4600', '#6e6e31', '#335726', '#4d60b5', '#a19564', '#623f28', '#44d457',
          '#70aacf', '#2d6b4d', '#72af9e', '#fd1500', '#d8b391', '#79893b', '#7cc6d8', '#db9036', '#eb605d', '#eb5ed4', '#e47ba7', '#a56b97', '#009744', '#ba5e21', '#bcac52', '#87d82f', '#873472', '#aea8d1', '#e28c62', '#d1b1eb', '#36429e', '#3abdc1', '#669c4d', '#9e0399', '#4d4d79', '#7b4b85', '#c33431', '#8c6677', '#aa002d', '#7e0175', '#01824d', '#724967',
          '#727790', '#6e0099', '#a0ba52', '#e16e31', '#c46970', '#6d5b95', '#a33b74', '#316200', '#87004f', '#335769', '#ba8c7c', '#1859ff', '#909101', '#2b8ad4', '#1626ff', '#21d3ff', '#a390af', '#8a6d4f', '#5d213d', '#db03b3', '#6e56ca', '#642821', '#ac7700', '#a3bff6', '#b58346', '#9738db', '#b15093', '#7242a3', '#878ed1', '#8970b1', '#6baf36', '#5979c8',
          '#c69eff', '#56831a', '#00d6a7', '#824638', '#11421c', '#59aa75', '#905b01', '#f64470', '#ff9703', '#e14231', '#ba91cf', '#34574d', '#f7807c', '#903400', '#b3cd00', '#2d9ed3', '#798a9e', '#50807c', '#c136d6', '#eb0552', '#b8ac7e', '#487031', '#839564', '#d89c89', '#0064a3', '#4b9077', '#8e6097', '#ff5238', '#a7423b', '#006e70', '#97833d', '#dbafc8']

class Broaden:
    """
    Class designed to take in calculated spectral data, align it with experimental data, then broaden it appropriately.
    First: Load the experimental. 
    Second: Load all the calculations sequentially.
    Third: Generate the parameters used for shifting and broadening. Finally: Broaden the spectra.
    """

    def __init__(self):
        self.data = list() # Why is this here?

        self.maxSites = 40

        # These are the input and output spectra of type float
        # ['Column'] = [0=Energy, 1=Counts]
        # ['Column'] = [0=Energy, 1=Counts, 2=CoreLifeXAS, 3=Intermediate Step, 4=Delta E, 5=Intermediate Step, 6=Final Gaussian Counts]
        # ['Row'] = data
        # ['XES, XANES'] = [0=XES, 1=XANES]
        # ['XES, XAS, or XANES'] = [0=XES,1=XAS,2=XANES]
        self.ExpSXS = np.zeros([2, 1500, 2])  # Experimental Spectra ['Column']['Row']['XES or XANES']
        self.CalcSXS = np.zeros([2,3500,3,self.maxSites]) # Calculated Spectra ['Column']['Row']['XES,XAS or XANES']['Site']
        self.BroadSXS = np.zeros([7, 3500, 3, self.maxSites])  # Broadened Calculated Spectra ['Column']['Row']['XES,XAS or XANES']['Site']
        self.SumSXS = np.zeros([2, 3500, 3])  # Total Summed Spectra
        self.Gauss = np.zeros([3500,3500]) # Gauss broadening matrix for each spectrum
        self.Lorentz = np.zeros([3500, 3500])  # Lorentz broadening matrix for each spectrum
        self.Disorder = np.zeros([3500, 3500])  # Disorder broadening matrix for each spectrum

        self.ExpSXSCount = np.zeros([2], dtype=int)  # Stores number of elements in the arrays of Experimental data
        self.CalcSXSCase = 0
        self.CalcSXSCount = np.zeros([3, self.maxSites], dtype=int)  # Stores number of elements in the arrays of Calculated data
        self.BroadSXSCount = np.zeros([3,self.maxSites],dtype=int) # Stores number of elements in the arrays of Shifted/Intermediate data
        self.SumSXSCount = np.zeros([3],dtype=int) # Store number of elements in the arrays of Final Data

        # These store data for generating the broadening criteria
        self.scaleXES = np.zeros([self.maxSites, 50])
        self.Bands = np.zeros([50,self.maxSites,2])
        self.BandNum = np.zeros([self.maxSites], dtype=int)
        self.Fermi = 0 # Ground state fermi energy
        self.Fermis = np.zeros([self.maxSites])  # Excited state fermi energy for each inequivalent site
        self.Binds = np.zeros([self.maxSites])  # Ground state binding energy for each inequivalent site
        self.shiftXES = np.zeros([self.maxSites,50])
        self.scalar = np.zeros([3,self.maxSites])
        self.Edge = []
        self.Site = np.zeros([self.maxSites])

        # Misc
        self.bandshift = np.zeros([self.maxSites, self.maxSites])
        self.bands_temp = np.zeros([3500, self.maxSites, self.maxSites])
        self.bands_temp_count = np.zeros([self.maxSites, self.maxSites], dtype=int)
        self.BandGap = 0

        # Resolution Parameters
        # Set by initResolution()
        self.corelifeXES = None
        self.corelifeXAS = None
        self.spec = None
        self.mono = None
        self.disord = None
        self.XESscale = None
        self.scaleXAS = None
        self.XESbandScale = None

    def setMaxSites(self, maxNumberSites):
        """
        Used to increase the maximum number of sites. Default is 40
        """
        self.maxSites = maxNumberSites

    def clearFigures(self):
        """
        Used to clear previous figures. Shouldn't be necessary now that no global variables exist.
        """
        self.CalcSXSCase = 0

    def setFermi(self,groundStateFermi):
        """
        Sets the ground state fermi level for when no experimental data is present.
        """
        self.Fermi = groundStateFermi

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
                    self.ExpSXS[0][c1][0] = df[0][c1] # Energy
                    self.ExpSXS[1][c1][0] = df[1][c1] # Counts
                    if self.ExpSXS[1][c1][0] > maxEXP:
                        maxEXP = self.ExpSXS[1][c1][0] # Get max value in experimental XES
                    c1 += 1
                self.ExpSXSCount[0] = c1 # Length of data points
                for i in range(self.ExpSXSCount[0]): # Normalize spectra
                    self.ExpSXS[1][i][0] = self.ExpSXS[1][i][0]/maxEXP
        except:
            with open(basedir+"/"+XES, "r") as xesFile: # Measured XES
                # This trys it as a .csv instead of a .txt
                df = pd.read_csv(xesFile, header=None, skiprows=headerlines[0]) # Change to '\s*' and specify engine='python' if this breaks in jupyter notebook                
                c1 = 0
                maxEXP = 0
                for i in range(len(df)): 
                    self.ExpSXS[0][c1][0] = df[0][c1] # Energy
                    self.ExpSXS[1][c1][0] = df[1][c1] # Counts
                    if self.ExpSXS[1][c1][0] > maxEXP:
                        maxEXP = self.ExpSXS[1][c1][0] # Get max value in experimental XES
                    c1 += 1
                self.ExpSXSCount[0] = c1 # Length of data points
                for i in range(self.ExpSXSCount[0]): # Normalize spectra
                    self.ExpSXS[1][i][0] = self.ExpSXS[1][i][0]/maxEXP
        try:
            with open(basedir+"/"+XANES, "r") as xanesFile: # Measured XANES
                df = pd.read_csv(xanesFile, delimiter='\s+', header=None, skiprows=headerlines[1])
                c1 = 0
                for i in range(len(df)):
                    self.ExpSXS[0][c1][1] = df[0][c1] # Energy
                    self.ExpSXS[1][c1][1] = df[1][c1] # Counts
                    c1 += 1
                self.ExpSXSCount[1] = c1 # Length of data points
        except:
            with open(basedir+"/"+XANES, "r") as xanesFile: # Measured XANES
                # This trys it as a .csv instead of a .txt
                df = pd.read_csv(xanesFile, header=None, skiprows=headerlines[1])
                c1 = 0
                for i in range(len(df)):
                    self.ExpSXS[0][c1][1] = df[0][c1] # Energy
                    self.ExpSXS[1][c1][1] = df[1][c1] # Counts
                    c1 += 1
                self.ExpSXSCount[1] = c1 # Length of data points

        self.CalcSXSCase = 0 # Stores number of calculated inequivalent sites
        self.Edge = []
        self.Fermi = GS_fermi
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

        if XANES == 0:
            XANES = XAS # For when no core hole exists
            ES_fermi = self.Fermi # Make the fermi level equal to the ground state.

        with open(basedir+"/"+XES, "r") as xesFile: # XES Calculation
            df = pd.read_csv(xesFile, delimiter='\s+',header=None, skiprows=headerlines[0])
            c1 = 0
            for i in range(len(df)):
                self.CalcSXS[0][c1][0][self.CalcSXSCase] = df[0][c1] # Energy
                self.CalcSXS[1][c1][0][self.CalcSXSCase] = df[1][c1] # Counts
                c1 += 1
            self.CalcSXSCount[0][self.CalcSXSCase] = c1 # Length for each Site

        with open(basedir+"/"+XAS, "r") as xasFile: # XAS Calculation
            df = pd.read_csv(xasFile, delimiter='\s+',header=None, skiprows=headerlines[1])
            c1 = 0
            for i in range(len(df)):
                self.CalcSXS[0][c1][1][self.CalcSXSCase] = df[0][c1] # Energy
                self.CalcSXS[1][c1][1][self.CalcSXSCase] = df[1][c1] # Counts
                c1 += 1
            self.CalcSXSCount[1][self.CalcSXSCase] = c1 # Length for each Site

        with open(basedir+"/"+XANES, "r") as xanesFile: # XANES Calculation
            df = pd.read_csv(xanesFile, delimiter='\s+',header=None, skiprows=headerlines[2])
            c1 = 0
            for i in range(len(df)):
                self.CalcSXS[0][c1][2][self.CalcSXSCase] = df[0][c1] # Energy
                self.CalcSXS[1][c1][2][self.CalcSXSCase] = df[1][c1] # Counts
                c1 += 1
            self.CalcSXSCount[2][self.CalcSXSCase] = c1 # Length for each Site

        # Update the global variables with the parameters for that site.
        self.Fermis[self.CalcSXSCase] = ES_fermi
        self.Binds[self.CalcSXSCase] = GS_bindingEnergy
        self.Edge.append(edge)
        self.Site[self.CalcSXSCase] = sites
        self.CalcSXSCase += 1
        return

    def FindBands(self): 
        """
        Finds the number of bands present in the calculated data.
        Bands are where the calculated data hits zero.
        """
        # The while loops can be changed to "for in range()"
        c1 = 0
        while c1 < self.CalcSXSCase: # For each site (number of .loadCalc)
            starter = False
            c3 = 0
            c2 = 0
            while c2 < self.CalcSXSCount[0][c1]: # For each data point
                if starter is False:
                    if self.CalcSXS[1][c2][0][c1] != 0: # Spectrum is not zero
                        self.Bands[c3][c1][0] = self.CalcSXS[0][c2][0][c1] # Start point of band
                        starter = True
                if starter is True:
                    if self.CalcSXS[1][c2][0][c1] == 0: # Spectrum hits zero
                        self.Bands[c3][c1][1] = self.CalcSXS[0][c2][0][c1] # End point of band
                        starter = False
                        c3 += 1
                c2 += 1
            self.BandNum[c1] = c3 # The number of bands in each spectrum
            c1 += 1
        return
    
    def printBands(self):
        """
        Prints the value of the band start and end locations, then plots the unshifted spectra.
        """
        self.FindBands()
        for c1 in range(self.CalcSXSCase):
            print("In inequivalent atom #" + str(c1))
            for c2 in range(self.BandNum[c1]):
                print("Band #" + str(c2) + " is located at " + str(self.Bands[c2][c1][0]) + " to " + str(self.Bands[c2][c1][1]))
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
            for c1 in range(self.CalcSXSCase):
                for c2 in range(self.BandNum[c1]):
                    self.shiftXES[c1][c2] = XESshift
        else: # Shift bands separately.
            for c1 in range(self.CalcSXSCase):
                for c2 in range(self.BandNum[c1]):
                    self.shiftXES[c1][c2] = XESbandshift[c1][c2]

        shiftXAS = XASshift
        for c1 in range(self.CalcSXSCase): # This goes through the XAS spectra
            for c2 in range(self.CalcSXSCount[1][c1]): # Line 504
                self.BroadSXS[1][c2][1][c1] = self.CalcSXS[1][c2][1][c1] # Counts from calc go into Broad
                self.BroadSXSCount[1][c1] = self.CalcSXSCount[1][c1]
                self.BroadSXS[0][c2][1][c1] = self.CalcSXS[0][c2][1][c1] + shiftXAS + (self.Binds[c1]+self.Fermi) * Ryd # Shift the energy of XAS based on binding, fermi energy, and user input
        
        for c1 in range(self.CalcSXSCase): # This goes through the XANES spectra
            for c2 in range(self.CalcSXSCount[2][c1]): # Line 514
                self.BroadSXS[1][c2][2][c1] = self.CalcSXS[1][c2][2][c1] # Counts from calc go into Broad
                self.BroadSXSCount[2][c1] = self.CalcSXSCount[2][c1]
                self.BroadSXS[0][c2][2][c1] = self.CalcSXS[0][c2][2][c1] + shiftXAS + (self.Binds[c1]+self.Fermis[c1]) * Ryd # Shift the energy of XANES based on binding, fermi energy, and user input

        for c1 in range(self.CalcSXSCase): # If there are a different shift between bands find that difference
            for c2 in range(self.BandNum[c1]): # Line 526
                self.bandshift[c1][c2] = self.shiftXES[c1][c2] - self.shiftXES[c1][0]

        for c1 in range(self.CalcSXSCase): # This goes through the XES spectra
            self.BroadSXSCount[0][c1] = self.CalcSXSCount[0][c1]
            for c2 in range(self.CalcSXSCount[0][c1]): # Line 535
                self.BroadSXS[0][c2][0][c1] = self.CalcSXS[0][c2][0][c1] + self.bandshift[c1][0] # Still confused why bandshift[c1][0] is here. Always zero
                self.BroadSXS[1][c2][0][c1] = self.CalcSXS[1][c2][0][c1]

        for c1 in range(self.CalcSXSCase): # Not entirely sure the purpose of the next portion of code
            c2 = 1 # Line 544
            c3 = 0
            while c3 < self.BroadSXSCount[0][c1]:
                if self.BroadSXS[0][c3][0][c1] >= (self.Bands[c2][c1][0] + self.bandshift[c1][0]):
                    c4 = 0
                    while self.BroadSXS[1][c3][0][c1] != 0:
                        self.bands_temp[c4][c2][c1] = self.BroadSXS[1][c3][0][c1]
                        self.BroadSXS[1][c3][0][c1] = 0
                        c3 += 1
                        c4 += 1
                    self.bands_temp_count[c1][c2] = c4
                    c2 += 1
                    if c2 >= self.BandNum[c1]:
                        c3 = 999999
                c3 += 1

        for c1 in range(self.CalcSXSCase):
            for c2 in range(1,self.BandNum[c1]): # Line 570
                c3 = 0
                while c3 < self.BroadSXSCount[0][c1]:
                    if self.BroadSXS[0][c3][0][c1] >= (self.Bands[c2][c1][0] + self.bandshift[c1][c2]):
                        c4 = 0
                        while c4 < self.bands_temp_count[c1][c2]:
                            self.BroadSXS[1][c3][0][c1] = self.bands_temp[c4][c2][c1]
                            c4 += 1
                            c3 += 1
                        c3 = 999999
                    c3 += 1
        
        for c1 in range(self.CalcSXSCase):
            for c2 in range(self.BroadSXSCount[0][c1]): # Line 592
                self.BroadSXS[0][c2][0][c1] = self.BroadSXS[0][c2][0][c1] + self.shiftXES[c1][0] + (self.Binds[c1]+self.Fermi) * Ryd # Shift XES spectra based on binding, fermi energy, and user input

        c1 = self.BroadSXSCount[0][0]-1
        while c1 >= 0: # Starts from the top and moves down until it finds the point where the valence band != 0
            if self.BroadSXS[1][c1][0][0] > 0:
                Eval = self.BroadSXS[0][c1][0][0]
                c1 = -1
            c1 -= 1

        c1 = 0
        while c1 < self.BroadSXSCount[1][0]: # Starts from the bottom and moves up until it finds the point where the conduction bands != 0
            if self.BroadSXS[1][c1][1][0] > 0:
                Econ = self.BroadSXS[0][c1][1][0]
                c1 = 999999
            c1 += 1

        for c3 in range(3):
            for c1 in range(self.CalcSXSCase):
                for c2 in range(self.BroadSXSCount[c3][c1]):
                    self.BroadSXS[1][c2][c3][c1] = self.BroadSXS[1][c2][c3][c1] * (self.BroadSXS[0][c2][c3][c1] / Econ)

        self.BandGap = Econ - Eval # Calculate the band gap
        print("BandGap = " + str(self.BandGap) + " eV")

        
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

    def broaden(self,separate=True, Ængus=False):
        """
        This will take the shifted calculated spectra and broaden it based on the lifetime, instrument, and general disorder broadening.
        It creates a series of gaussians and lorentzians before applying it to the spectra appropriately.

        Parameters
        ----------
        separate : True/False
            Specify whether or not to create a separate output plot of XES and XAS
        """
        if Ængus == "yup":
            Ængus = True

        Econd = np.zeros(self.maxSites)
        type = False
        energy_0 = 20
        Pi = 3.14159265 # The Pi constant used for the distribution functions.

        if self.XESbandScale == 0: # Applying a singular scale to XES
            for c1 in range(self.CalcSXSCase):
                for c2 in range(self.BandNum[c1]):
                    self.scaleXES[c1][c2] = self.XESscale
        else: # Applying scale to individual bands in XES
            for c1 in range(self.CalcSXSCase):
                for c2 in range(self.BandNum[c1]):
                    self.scaleXES[c1][c2] = self.XESbandScale[c1][c2]
        
        for c1 in range(self.CalcSXSCase): # Line 791
            c2 = 0
            while c2 < self.BroadSXSCount[2][c1]:
                if self.BroadSXS[1][c2][2][c1] != 0:
                    Econd[c1] = self.BroadSXS[0][c2][2][c1]
                    c2 = 999999
                c2 += 1
        
        for c1 in range(self.CalcSXSCase): # Using scaling factor for corehole lifetime for XAS and XANES
            for c2 in range(1,3): # Line 805
                for c3 in range(self.BroadSXSCount[c2][c1]):
                    if self.BroadSXS[0][c3][c2][c1] <= Econd[c1]:
                        self.BroadSXS[2][c3][c2][c1] = self.corelifeXAS
                    else:
                        if self.BroadSXS[0][c3][c2][c1] < Econd[c1] + energy_0:
                            self.BroadSXS[2][c3][c2][c1] = self.scaleXAS/100 * ((self.BroadSXS[0][c3][c2][c1]-Econd[c1]) * (self.BroadSXS[0][c3][c2][c1]-Econd[c1])) + self.corelifeXAS # Replace with **2 ??
                        else:
                            self.BroadSXS[2][c3][c2][c1] = self.scaleXAS/100 * (energy_0 * energy_0) + self.corelifeXAS
                    self.BroadSXS[4][c3][c2][c1] = self.BroadSXS[0][c3][c2][c1] / self.mono

        for c1 in range(self.CalcSXSCase): # Corehole lifetime scaling for XES
            type = False # Line 830
            c3 = 0
            for c2 in range(self.BroadSXSCount[0][c1]):
                self.BroadSXS[4][c2][0][c1] = self.BroadSXS[0][c2][0][c1]/self.spec
                if type is False:
                    if self.BroadSXS[1][c2][0][c1] != 0:
                        type = True
                    else:
                        self.BroadSXS[2][c2][0][c1] = self.scaleXES[c1][c3]/100 * ((self.BroadSXS[0][c2][0][c1]-Econd[c1]) * (self.BroadSXS[0][c2][0][c1]-Econd[c1])) + self.corelifeXES
                if type is True:
                    if self.BroadSXS[1][c2][0][c1] == 0:
                        self.BroadSXS[2][c2][0][c1] = self.scaleXES[c1][c3]/100 * ((self.BroadSXS[0][c2][0][c1]-Econd[c1]) * (self.BroadSXS[0][c2][0][c1]-Econd[c1])) + self.corelifeXES
                        type = False
                        c3 += 1
                        if c3 > self.BandNum[c1]:
                            c3 = self.BandNum[c1]-1
                    else:
                        self.BroadSXS[2][c2][0][c1] = self.scaleXES[c1][c3]/100 * ((self.BroadSXS[0][c2][0][c1]-Econd[c1]) * (self.BroadSXS[0][c2][0][c1]-Econd[c1])) + self.corelifeXES

        # Creating the broadening matrices.
        for c1 in range(self.CalcSXSCase): # This is only for the XES spectra
            for c3 in range(self.BroadSXSCount[0][c1]): # Takes about 1 second to complete a full cycle of c3 * # of input files
                width = self.BroadSXS[4][c3][0][c1]/2.3548 # We extract the variance for the Gaussian Distribution
                position = self.BroadSXS[0][c3][0][c1] # We extract the centroid of the Gaussian Distribution
                self.Gauss[c3,:] = np.reciprocal(np.sqrt(2*Pi*width*width))*np.exp(-(self.BroadSXS[0,:,0,c1]-position)*(self.BroadSXS[0,:,0,c1]-position)/2/width/width)

                #  Commented out since disorder does not affect XES
                #width = self.disord/2.3548; # We extract the variance for the Gaussian Distribution
                #position = self.BroadSXS[0][c3][0][c1]; # We extract the centroid of the Gaussian Distribution
                #self.Disorder[c3,:] = np.reciprocal(np.sqrt(2*Pi*width*width))*np.exp(-(self.BroadSXS[0,:,0,c1]-position)*(self.BroadSXS[0,:,0,c1]-position)/2/width/width)

                width = self.BroadSXS[2][c3][0][c1]/2 # We extract the variance for the Gaussian Distribution
                position = self.BroadSXS[0][c3][0][c1] # We extract the centroid of the Gaussian Distribution
                self.Lorentz[c3,:] = np.reciprocal(Pi)*(width/((self.BroadSXS[0,:,0,c1]-position)*(self.BroadSXS[0,:,0,c1]-position)+(width*width)))
            
            self.BroadSXS[3,:,0,c1] = 0 # Line 901
            for c3 in range(self.BroadSXSCount[0][c1]):
                self.BroadSXS[3,:,0,c1] = self.BroadSXS[3,:,0,c1]+(self.Lorentz[c3,:]*self.BroadSXS[1][c3][0][c1]*(self.BroadSXS[0][1][0][c1]-self.BroadSXS[0][0][0][c1]))
            
            self.BroadSXS[6,:,0,c1] = 0 # Line 912
            for c3 in range(self.BroadSXSCount[0][c1]):
                self.BroadSXS[6,:,0,c1] = self.BroadSXS[6,:,0,c1]+(self.Gauss[c3,:]*self.BroadSXS[3][c3][0][c1]*(self.BroadSXS[0][1][0][c1]-self.BroadSXS[0][0][0][c1]))

            #self.BroadSXS[6,:,0,c1] = 0 # Line 924 Originally commented out in C code because disorder does not impact XES.
            #for c4 in range(self.BroadSXSCount[0][c1]):
            #    self.BroadSXS[6,:,0,c1] = self.BroadSXS[6,:,0,c1]+(self.Disorder[c4,:]*self.BroadSXS[5][c4][0][c1]*(self.BroadSXS[0][1][0][c1]-self.BroadSXS[0][0][0][c1]))


        for c1 in range(self.CalcSXSCase): # Line 938
            for c2 in range(1,3):
                for c3 in range(self.BroadSXSCount[c2][c1]): # Takes about 1 second to complete a full cycle of c3 * # of input files
                    width = self.BroadSXS[4][c3][c2][c1]/2.3548 # We extract the variance for the Gaussian Distribution
                    position = self.BroadSXS[0][c3][c2][c1] # We extract the centroid of the Gaussian Distribution
                    self.Gauss[c3,:] = np.reciprocal(np.sqrt(2*Pi*width*width))*np.exp(-(self.BroadSXS[0,:,c2,c1]-position)*(self.BroadSXS[0,:,c2,c1]-position)/2/width/width)

                    width = self.disord/2.3548 # We extract the variance for the Gaussian Distribution
                    position = self.BroadSXS[0][c3][c2][c1] # We extract the centroid of the Gaussian Distribution
                    self.Disorder[c3,:] = np.reciprocal(np.sqrt(2*Pi*width*width))*np.exp(-(self.BroadSXS[0,:,c2,c1]-position)*(self.BroadSXS[0,:,c2,c1]-position)/2/width/width)
                    
                    width = self.BroadSXS[2][c3][c2][c1]/2 # We extract the variance for the Gaussian Distribution
                    position = self.BroadSXS[0][c3][c2][c1] # We extract the centroid of the Gaussian Distribution
                    self.Lorentz[c3,:] = np.reciprocal(Pi)*(width/((self.BroadSXS[0,:,c2,c1]-position)*(self.BroadSXS[0,:,c2,c1]-position)+(width*width)))
                
                self.BroadSXS[3,:,c2,c1] = 0 # Line 967
                for c3 in range(self.BroadSXSCount[c2][c1]):
                    self.BroadSXS[3,:,c2,c1] = self.BroadSXS[3,:,c2,c1]+(self.Lorentz[:,c3]*self.BroadSXS[1][c3][c2][c1]*(self.BroadSXS[0][1][c2][c1]-self.BroadSXS[0][0][c2][c1]))
                
                self.BroadSXS[5,:,c2,c1] = 0 # Line 978
                for c3 in range(self.BroadSXSCount[c2][c1]):
                    self.BroadSXS[5,:,c2,c1] = self.BroadSXS[5,:,c2,c1]+(self.Gauss[:,c3]*self.BroadSXS[3][c3][c2][c1]*(self.BroadSXS[0][1][c2][c1]-self.BroadSXS[0][0][c2][c1]))

                self.BroadSXS[6,:,c2,c1] = 0 # Line 990
                for c3 in range(self.BroadSXSCount[c2][c1]):
                    self.BroadSXS[6,:,c2,c1] = self.BroadSXS[6,:,c2,c1]+(self.Disorder[c3,:]*self.BroadSXS[5][c3][c2][c1]*(self.BroadSXS[0][1][c2][c1]-self.BroadSXS[0][0][c2][c1]))
        self.add()

        if separate is False:
            if Ængus is True:
                # Creating the figure for plotting the broadened data.
                p = figure(height=450, width=900, title="Broadened Data", x_axis_label="Energy (eV)", y_axis_label="Normalized Intensity (arb. units)",
                        tools="pan,wheel_zoom,box_zoom,reset,crosshair,save")
                p.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
                    ("(x,y)", "(Energy, Intensity)"),
                    ("(x,y)", "($x, $y)")
                ]))
                self.plotÆngus(p)
                self.plotExp(p)
                p.add_layout(p.legend[0], 'right')
                show(p)
            else:
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
            if Ængus is True:
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
                self.plotÆngusXANES(p)
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
        for c1 in range(self.CalcSXSCase): # Determine the relative addition scale factor XES
            for c2 in range(3):
                self.scalar[c2][c1]=1

        for c1 in range(self.CalcSXSCase): # Apply the scaling to the running scalar
            for c2 in range(5): # Counts through the types of edges
                if self.Edge[c1] == Edge_check[c2]:
                    for c3 in range(3):
                        self.scalar[c3][c1] = self.scalar[c3][c1]*self.Site[c1]*Edge_scale[c2]

        statement = 0 # Print statement tracker

        for c1 in range(3):
            first = 0
            value = self.BroadSXS[0][0][c1][0]
            c2 = 1
            while c2 < self.CalcSXSCase:
                if self.BroadSXS[0][0][c1][c2] >= value:
                    first = c2
                c2 += 1
            for c3 in range(self.BroadSXSCount[c1][first]):
                self.SumSXS[0][c3][c1] = self.BroadSXS[0][c3][c1][first]
                self.SumSXS[1][c3][c1] = self.scalar[c1][first]*self.BroadSXS[6][c3][c1][first]

            self.SumSXSCount[c1] = c3
            for c2 in range(self.CalcSXSCase):
                if c2 != first:
                    c4 = 0
                    for c3 in range(self.SumSXSCount[c1]):
                        c4 = c3 - 5 # This speeds up the program significantly by not starting at 0 every time.
                        # It will work as long as the interpolated data point is within -5 x values. Can be as far forward as neccesary
                        # Based on the criteria though, it should never be a negative point.
                        if c4 < 0:
                            c4 = 0
                        if self.BroadSXS[0][c4][c1][c2] > self.SumSXS[0][c3][c1] and c3 != 0:
                            c4 = c3 - 50 # We try again with a larger range to start out with.
                            if c4 < 0:
                                c4 = 0
                            if statement == 0:
                                statement = 1
                            if self.BroadSXS[0][c4][c1][c2] > self.SumSXS[0][c3][c1]:
                                c4 = c3 - 500 # Try once again with a larger range.
                                if c4 < 0:
                                    c4 = 0
                                if self.BroadSXS[0][c4][c1][c2] > self.SumSXS[0][c3][c1]:
                                    c4 = 0 # This just ensures that if it is as far back as allowed, it will instead start at 0 to go through all values
                                    # This would slow down the program, but only in cases where necessary.
                                    if statement == 1:
                                        print("The broadening will longer than normal. To avoid this, try making all of the .txspec files the same length. (-2 to 50eV for example)")
                                        statement = 2
                        
                        while c4 < self.BroadSXSCount[c1][c2]:
                            if self.BroadSXS[0][c4][c1][c2] > self.SumSXS[0][c3][c1]:
                                x1 = self.BroadSXS[0][c4-1][c1][c2]
                                x2 = self.BroadSXS[0][c4][c1][c2]
                                y1 = self.BroadSXS[6][c4-1][c1][c2]
                                y2 = self.BroadSXS[6][c4][c1][c2]
                                slope = (y2-y1)/(x2-x1)
                                self.SumSXS[1][c3][c1] = self.SumSXS[1][c3][c1] + self.scalar[c1][c2]*(slope*(self.SumSXS[0][c3][c1]-x1)+y1)
                                max = c3
                                c4 = 9999999
                            c4 += 1
                    self.SumSXSCount[c1] = max
        return

    def initResolution(self, corelifetime, specResolution, monoResolution, disorder, XESscaling, XASscaling, XESbandScaling=0):
        """
        Specify the parameters for the broadening criteria.

        Parameters
        ----------
        corelifetime : float
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
        self.corelifeXES = corelifetime
        self.corelifeXAS = corelifetime
        self.spec = specResolution
        self.mono = monoResolution
        self.disord = disorder
        self.XESscale = XESscaling
        self.scaleXAS = XASscaling
        self.XESbandScale = XESbandScaling
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
        xesX = np.zeros([self.ExpSXSCount[0]])
        xesY = np.zeros([self.ExpSXSCount[0]])
        xanesX = np.zeros([self.ExpSXSCount[1]])
        xanesY = np.zeros([self.ExpSXSCount[1]])

        for c1 in range(self.ExpSXSCount[0]): # Experimental xes spectra
            xesX[c1] = self.ExpSXS[0][c1][0]
            xesY[c1] = self.ExpSXS[1][c1][0]
        
        for c1 in range(self.ExpSXSCount[1]): # Experimental xanes spectra
            xanesX[c1] = self.ExpSXS[0][c1][1]
            xanesY[c1] = self.ExpSXS[1][c1][1]
        
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
        xesX = np.zeros([self.ExpSXSCount[0]])
        xesY = np.zeros([self.ExpSXSCount[0]])
        xanesX = np.zeros([self.ExpSXSCount[1]])
        xanesY = np.zeros([self.ExpSXSCount[1]])

        for c1 in range(self.ExpSXSCount[0]): # Experimental xes spectra
            xesX[c1] = self.ExpSXS[0][c1][0]
            xesY[c1] = self.ExpSXS[1][c1][0]
        
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
        xanesX = np.zeros([self.ExpSXSCount[1]])
        xanesY = np.zeros([self.ExpSXSCount[1]])
        
        for c1 in range(self.ExpSXSCount[1]): # Experimental xanes spectra
            xanesX[c1] = self.ExpSXS[0][c1][1]
            xanesY[c1] = self.ExpSXS[1][c1][1]
        
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

        MaxCalcSXS = np.zeros([3,self.maxSites]) # Find the maximum value in the spectra to normalize it for plotting.
        for c1 in range(self.CalcSXSCase):
            for c3 in range(3):
                for c2 in range(self.CalcSXSCount[c3][c1]):
                    if MaxCalcSXS[c3][c1] < self.BroadSXS[1][c2][c3][c1]:
                        MaxCalcSXS[c3][c1] = self.BroadSXS[1][c2][c3][c1]
        #p = figure()
        for c1 in range(self.CalcSXSCase):
            calcxesX = np.zeros([self.CalcSXSCount[0][c1]])
            calcxesY = np.zeros([self.CalcSXSCount[0][c1]])
            calcxasX = np.zeros([self.CalcSXSCount[1][c1]])
            calcxasY = np.zeros([self.CalcSXSCount[1][c1]])
            calcxanesX = np.zeros([self.CalcSXSCount[2][c1]])
            calcxanesY = np.zeros([self.CalcSXSCount[2][c1]])
            for c2 in range(self.CalcSXSCount[0][c1]): # Calculated XES spectra
                calcxesX[c2] = self.BroadSXS[0][c2][0][c1]
                calcxesY[c2] = self.BroadSXS[1][c2][0][c1] / (MaxCalcSXS[0][c1])
                #y = (x - x_min) / (x_max - x_min) Where x_min = 0

            for c2 in range(self.CalcSXSCount[1][c1]): # Calculated XAS spectra
                calcxasX[c2] = self.BroadSXS[0][c2][1][c1]
                calcxasY[c2] = self.BroadSXS[1][c2][1][c1] / (MaxCalcSXS[1][c1])

            for c2 in range(self.CalcSXSCount[2][c1]): # Calculated XANES spectra
                calcxanesX[c2] = self.BroadSXS[0][c2][2][c1]
                calcxanesY[c2] = self.BroadSXS[1][c2][2][c1] / (MaxCalcSXS[2][c1])
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

        MaxCalcSXS = np.zeros([3,self.maxSites]) # Find the maximum value in the spectra to normalize it for plotting.
        for c1 in range(self.CalcSXSCase):
            for c3 in range(3):
                for c2 in range(self.CalcSXSCount[c3][c1]):
                    if MaxCalcSXS[c3][c1] < self.BroadSXS[1][c2][c3][c1]:
                        MaxCalcSXS[c3][c1] = self.BroadSXS[1][c2][c3][c1]
        #p = figure()
        for c1 in range(self.CalcSXSCase):
            calcxesX = np.zeros([self.CalcSXSCount[0][c1]])
            calcxesY = np.zeros([self.CalcSXSCount[0][c1]])

            for c2 in range(self.CalcSXSCount[0][c1]): # Calculated XES spectra
                calcxesX[c2] = self.BroadSXS[0][c2][0][c1]
                calcxesY[c2] = self.BroadSXS[1][c2][0][c1] / (MaxCalcSXS[0][c1])
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

        MaxCalcSXS = np.zeros([3,self.maxSites]) # Find the maximum value in the spectra to normalize it for plotting.
        for c1 in range(self.CalcSXSCase):
            for c3 in range(3):
                for c2 in range(self.CalcSXSCount[c3][c1]):
                    if MaxCalcSXS[c3][c1] < self.BroadSXS[1][c2][c3][c1]:
                        MaxCalcSXS[c3][c1] = self.BroadSXS[1][c2][c3][c1]
        #p = figure()
        for c1 in range(self.CalcSXSCase):
            calcxasX = np.zeros([self.CalcSXSCount[1][c1]])
            calcxasY = np.zeros([self.CalcSXSCount[1][c1]])
            calcxanesX = np.zeros([self.CalcSXSCount[2][c1]])
            calcxanesY = np.zeros([self.CalcSXSCount[2][c1]])

            for c2 in range(self.CalcSXSCount[1][c1]): # Calculated XAS spectra
                calcxasX[c2] = self.BroadSXS[0][c2][1][c1]
                calcxasY[c2] = self.BroadSXS[1][c2][1][c1] / (MaxCalcSXS[1][c1])

            for c2 in range(self.CalcSXSCount[2][c1]): # Calculated XANES spectra
                calcxanesX[c2] = self.BroadSXS[0][c2][2][c1]
                calcxanesY[c2] = self.BroadSXS[1][c2][2][c1] / (MaxCalcSXS[2][c1])
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

        MaxCalcSXS = np.zeros([3,self.maxSites]) # Find the maximum value in the spectra to normalize it for plotting.
        for c1 in range(self.CalcSXSCase):
            for c3 in range(3):
                for c2 in range(self.CalcSXSCount[c3][c1]):
                    if MaxCalcSXS[c3][c1] < self.CalcSXS[1][c2][c3][c1]:
                        MaxCalcSXS[c3][c1] = self.CalcSXS[1][c2][c3][c1]
        for c1 in range(self.CalcSXSCase): # Since this is np array you can use : to get all data points
            calcxesX = np.zeros([self.CalcSXSCount[0][c1]])
            calcxesY = np.zeros([self.CalcSXSCount[0][c1]])

            for c2 in range(self.CalcSXSCount[0][c1]): # Calculated XES spectra
                calcxesX[c2] = self.CalcSXS[0][c2][0][c1]
                calcxesY[c2] = self.CalcSXS[1][c2][0][c1] / (MaxCalcSXS[0][c1])
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
            for c2 in range(self.SumSXSCount[c3]):
                if MaxBroadSXS[c3] < self.SumSXS[1][c2][c3]:
                    MaxBroadSXS[c3] = self.SumSXS[1][c2][c3]
        #p = figure()
        sumxesX = np.zeros([self.SumSXSCount[0]])
        sumxesY = np.zeros([self.SumSXSCount[0]])
        sumxasX = np.zeros([self.SumSXSCount[1]])
        sumxasY = np.zeros([self.SumSXSCount[1]])
        sumxanesX = np.zeros([self.SumSXSCount[2]])
        sumxanesY = np.zeros([self.SumSXSCount[2]])
        for c2 in range(self.SumSXSCount[0]): # Calculated XES spectra
            sumxesX[c2] = self.SumSXS[0][c2][0]
            sumxesY[c2] = self.SumSXS[1][c2][0] / MaxBroadSXS[0]

        for c2 in range(self.SumSXSCount[1]): # Calculated XAS spectra
            sumxasX[c2] = self.SumSXS[0][c2][1]
            sumxasY[c2] = self.SumSXS[1][c2][1] / MaxBroadSXS[1]

        for c2 in range(self.SumSXSCount[2]): # Calculated XANES spectra
            sumxanesX[c2] = self.SumSXS[0][c2][2]
            sumxanesY[c2] = self.SumSXS[1][c2][2] / MaxBroadSXS[2]

        p.line(sumxesX,sumxesY,line_color="limegreen",legend_label="Broadened XES/XANES") # XES plot
        p.line(sumxasX,sumxasY,line_color="blue",legend_label="Broadened XAS") # XAS plot
        p.line(sumxanesX,sumxanesY,line_color="limegreen") # XANES plot
        #show(p)
        return
    
    def plotÆngus(self,p):
        """
        Plot the final calculated and broadened data.
        The bokeh figure needs to be created and configured outside of the function. This simply adds the XANES and XES to a figure.

        Parameters
        ----------
        p : figure()
            The bokeh figure needs to be created outside of the function.
        """
        MaxBroadSXS = np.zeros([3])
        for c3 in range(3): # Find the maximum value for normalization
            for c2 in range(self.SumSXSCount[c3]):
                if MaxBroadSXS[c3] < self.SumSXS[1][c2][c3]:
                    MaxBroadSXS[c3] = self.SumSXS[1][c2][c3]
        #p = figure()
        sumxesX = np.zeros([self.SumSXSCount[0]])
        sumxesY = np.zeros([self.SumSXSCount[0]])
        sumxanesX = np.zeros([self.SumSXSCount[2]])
        sumxanesY = np.zeros([self.SumSXSCount[2]])
        for c2 in range(self.SumSXSCount[0]): # Calculated XES spectra
            sumxesX[c2] = self.SumSXS[0][c2][0]
            sumxesY[c2] = self.SumSXS[1][c2][0] / MaxBroadSXS[0]

        for c2 in range(self.SumSXSCount[2]): # Calculated XANES spectra
            sumxanesX[c2] = self.SumSXS[0][c2][2]
            sumxanesY[c2] = self.SumSXS[1][c2][2] / MaxBroadSXS[2]

        p.line(sumxesX,sumxesY,line_color="limegreen",legend_label="Broadened XES/XANES") # XES plot
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
            for c2 in range(self.SumSXSCount[c3]):
                if MaxBroadSXS[c3] < self.SumSXS[1][c2][c3]:
                    MaxBroadSXS[c3] = self.SumSXS[1][c2][c3]
        #p = figure()
        sumxasX = np.zeros([self.SumSXSCount[1]])
        sumxasY = np.zeros([self.SumSXSCount[1]])
        sumxanesX = np.zeros([self.SumSXSCount[2]])
        sumxanesY = np.zeros([self.SumSXSCount[2]])

        for c2 in range(self.SumSXSCount[1]): # Calculated XAS spectra
            sumxasX[c2] = self.SumSXS[0][c2][1]
            sumxasY[c2] = self.SumSXS[1][c2][1] / MaxBroadSXS[1]

        for c2 in range(self.SumSXSCount[2]): # Calculated XANES spectra
            sumxanesX[c2] = self.SumSXS[0][c2][2]
            sumxanesY[c2] = self.SumSXS[1][c2][2] / MaxBroadSXS[2]

        p.line(sumxasX,sumxasY,line_color="blue",legend_label="Broadened XAS") # XAS plot
        p.line(sumxanesX,sumxanesY,line_color="limegreen",legend_label="Broadened XANES") # XANES plot
        return
    
    def plotÆngusXANES(self,p):
        """
        Plot the final calculated and broadened data for XANES
        The bokeh figure needs to be created and configured outside of the function. This simply adds the XANES to a figure.

        Parameters
        ----------
        p : figure()
            The bokeh figure needs to be created outside of the function.
        """
        MaxBroadSXS = np.zeros([3])
        for c3 in range(3): # Find the maximum value for normalization
            for c2 in range(self.SumSXSCount[c3]):
                if MaxBroadSXS[c3] < self.SumSXS[1][c2][c3]:
                    MaxBroadSXS[c3] = self.SumSXS[1][c2][c3]
        #p = figure()
        sumxanesX = np.zeros([self.SumSXSCount[2]])
        sumxanesY = np.zeros([self.SumSXSCount[2]])

        for c2 in range(self.SumSXSCount[2]): # Calculated XANES spectra
            sumxanesX[c2] = self.SumSXS[0][c2][2]
            sumxanesY[c2] = self.SumSXS[1][c2][2] / MaxBroadSXS[2]
        
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
            for c2 in range(self.SumSXSCount[c3]):
                if MaxBroadSXS[c3] < self.SumSXS[1][c2][c3]:
                    MaxBroadSXS[c3] = self.SumSXS[1][c2][c3]
        
        sumxesX = np.zeros([self.SumSXSCount[0]])
        sumxesY = np.zeros([self.SumSXSCount[0]])
        for c2 in range(self.SumSXSCount[0]): # Calculated XES spectra
            sumxesX[c2] = self.SumSXS[0][c2][0]
            sumxesY[c2] = self.SumSXS[1][c2][0] / MaxBroadSXS[0]

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
            for c1 in range(self.SumSXSCount[0]):
                writer.writerow([self.SumSXS[0][c1][0],self.SumSXS[1][c1][0]])

        with open(f"{filename}_{element}_XAS.csv", 'w', newline='') as f:
            writer = csv.writer(f,delimiter=",")
            writer.writerow(["Energy",element+"_XAS"])
            for c1 in range(self.SumSXSCount[1]):
                writer.writerow([self.SumSXS[0][c1][1],self.SumSXS[1][c1][1]])

        with open(f"{filename}_{element}_XANES.csv", 'w', newline='') as f:
            writer = csv.writer(f,delimiter=",")
            writer.writerow(["Energy",element+"_XANES"])
            for c1 in range(self.SumSXSCount[2]):
                writer.writerow([self.SumSXS[0][c1][2],self.SumSXS[1][c1][2]])
        if individual is True:
            for c2 in range(self.CalcSXSCase):
                with open(f"{filename}_{element}"+str(c2+1)+"_XES.csv", 'w', newline='') as f:
                    writer = csv.writer(f,delimiter=",")
                    writer.writerow(["Energy",element+str(c2+1)+"_XES"])
                    for c1 in range(self.BroadSXSCount[0][c2]):
                        writer.writerow([self.BroadSXS[0][c1][0][c2],self.BroadSXS[6][c1][0][c2]])

                with open(f"{filename}_{element}"+str(c2+1)+"_XAS.csv", 'w', newline='') as f:
                    writer = csv.writer(f,delimiter=",")
                    writer.writerow(["Energy",element+str(c2+1)+"_XAS"])
                    for c1 in range(self.BroadSXSCount[1][c2]):
                        writer.writerow([self.BroadSXS[0][c1][1][c2],self.BroadSXS[6][c1][1][c2]])

                with open(f"{filename}_{element}"+str(c2+1)+"_XANES.csv", 'w', newline='') as f:
                    writer = csv.writer(f,delimiter=",")
                    writer.writerow(["Energy",element+str(c2+1)+"_XANES"])
                    for c1 in range(self.BroadSXSCount[2][c2]):
                        writer.writerow([self.BroadSXS[0][c1][2][c2],self.BroadSXS[6][c1][2][c2]])


        print(f"Successfully wrote DataFrame to {filename}.csv")
