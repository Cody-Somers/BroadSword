# BroadSword
Converting the BroadSword program written by Teak Boyko from the Canadian Light Source in Saskatoon, SK, CA.
The program has been transcribed into python so that it can be compatible with jupyter notebook.
Execution should be < 30s in the majority of cases. However when certain conditions do not align it can take > 2min. The program will warn the user of such an event.

Go to the [github](https://github.com/Cody-Somers/BroadSword/tree/main) to find an example program to better understand the input file requirements.

## Installation

Install the package from PyPi with the pip package manager. This is the recommended way to obtain a copy for your local machine and will install all required dependencies.

```
    $ pip install BroadSword
```

You will also need [Jupyter Notebook](https://github.com/jupyter) together with python 3 on your local machine.

## Example Program

```
# Specify the base directory for the location of the data files
# Or put the path name directly into the functions below
basedir = '.'
# basedir = "/Users/cas003/Downloads/Beamtime/DataFiles"

## Setup necessary inputs
from BroadSword.BroadSword import *
from bokeh.io import output_notebook
output_notebook(hide_banner=True)

# Create an instance of the class
broad = Broaden()

# Load the experimental and calculations
broad.loadExp(basedir,XES="N_test_XES.txt",XANES="N_test_XAS.txt",GS_fermi=0.44996547,headerlines=[2,2])
broad.loadCalc(basedir,XES="N1_emis.txspec",XAS="N1_abs.txspec",GS_bindingEnergy=27.176237,XANES="N1_half.txspec",ES_fermi=0.45062079,sites=1,edge="L2")
broad.loadCalc(basedir,XES="N2_emis.txspec",XAS="N2_abs.txspec",GS_bindingEnergy=27.177975,XANES="N2_half.txspec",ES_fermi=0.45091878)
broad.loadCalc(basedir,XES="N3_emis.txspec",XAS="N3_abs.txspec",GS_bindingEnergy=27.122234,XANES="N3_half.txspec",ES_fermi=0.45090808)
broad.loadCalc(basedir,XES="N4_emis.txspec",XAS="N4_abs.txspec",GS_bindingEnergy=27.177070,XANES="N4_half.txspec",ES_fermi=0.45088602)

# broad.loadCalc(basedir,XES="N1_emis.txspec",XAS="N1_abs.txspec",GS_bindingEnergy=27.177070) # Minimum required inputs to broaden a spectra.

# Initialize the broadening parameters
broad.initResolution(corelifetime=0.15,specResolution=1200,monoResolution=5000,disorder=0.5,XESscaling=0.5,XASscaling=0.5)
# Optionally you can scale specific bands in XEN. Use printBands() to determine where the bands are located.
# Then add the new argument XESbandScaling into initResolution()
# broad.printBands()
# XESbandScaling=[[0.1,0.2,0.2,0.4],[0.2,0.2,0.4,0.2],[0.3,0.2,0.1,0.5],[0.3,0.5,0.4,0.2]])

# Shift the spectra until the calculation aligns with the experimental
broad.Shift(XESshift=19.2,XASshift=20.2,separate=False)
# Optionally you can shift specific bands in XES.
# Add the new argument into Shift()
# XESbandshift=[[30,33,30,20],[15,19.2,19.2,19.2],[30,33,30,20],[15,19.2,19.2,19.2]])

# Broaden the spectra
broad.broaden(separate=False)

# Export the broadened calculated spectra
# broad.export(filename="GeP2N4",element="N",individual=False)
```

### Functions
Below are the functions with their input criteria. If needed the docstrings will appear in Jupyter notebook using "shift+tab"

```
def loadExp(self, basedir, XES, XANES, GS_fermi, headerlines=[0,0]):
# Loads the measured experimental data. Fermi energy is from the calculated ground state

def loadCalc(self, basedir, XES, XAS, GS_bindingEnergy, XANES=0, ES_fermi=0,  edge="K", sites=1, headerlines=[0,0,0]):
# Loads the calculated data. The header lines are an array describing the number of header lines in the [XES, XAS, XANES] respectively.
# Fermis is the energy from the calculated excited state. Binding is from the ground state.
# Specifying the edge and number of sites are only required if they differ from the K edge and you have a different number of atoms between different inequivalent atoms.

def printBands(self):
# Prints out the location of the bands

def initResolution(self, corelifetime, specResolution, monoResolution, disorder, XESscaling, XASscaling, XESbandScaling=0)
# Specifies the broadening parameters based on instrument, general disorder, and lifetime broadening.
# An optional variable to scale the bands individually is available

def Shift(self,XESshift, XASshift, XESbandshift=0, separate=False):
# Shifts the calculated spectra based on user input.
# An optional variable to shift the bands individually is available
# separate describes whether or not to create a separate plot for XES and XAS

def broaden(self, separate=True)
# Broadens the calculated spectra. The library path will need to specified as described in the Installation segment
# separate describes whether or not to create a separate plot for XES and XAS

def export(self, filename, element, individual=False)
# Exports the broadened data as a .csv file.
# Set the filename and then the elemental edge, as well as if the broadened spectra for each individual inequivalent site should be export.
# Ex: filename=C3N3, element=N, ouput would be C3N3_N_XES.csv
```
### Post Script

Shifting only takes ~1s to plot. Comment out the broad.broaden() function and shift the unbroadened spectra first until it is in the proper position. Then include broad.broaden() in the notebook since this can take ~30s to compute.
