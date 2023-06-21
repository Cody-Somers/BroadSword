# BroadSword
Converting the BroadSword program written by Teak Boyko from the Canadian Light Source in Saskatoon, SK, CA.
The program has been transcribed into python so that it can be compatible with jupyter notebook.
It still makes use of C functions as the runtime in pure Python is too long to be practically usable.

Go to the [github](https://github.com/Cody-Somers/BroadSword/tree/main) to find an example program to better understand the input file requirements.

## Installation

Install the package from PyPi with the pip package manager. This is the recommended way to obtain a copy for your local machine and will install all required dependencies.

```
    $ pip install BroadSword
```
After installing BroadSword the following lines may appear in the terminal.
```
Collecting BroadSword
  Using cached BroadSword-0.0.15-py3-none-any.whl (21 kB)
Requirement already satisfied: numpy in ./opt/anaconda3/lib/python3.9/site-packages (from BroadSword) (1.20.3)
.
.
.
Installing collected packages: BroadSword
Successfully installed BroadSword-0.0.15
```
Take note of location that the package was installed to, as this will be needed when running the program. In this case "opt/anaconda3/lib/python3.9/site-packages"

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
broad.loadExp(basedir,"N_test_XES.txt","N_test_XAS.txt",fermi=0.44996547)
broad.loadCalc(basedir,"N1_emis.txspec","N1_abs.txspec","N1_half.txspec",fermis=0.45062079,binds=27.176237)
broad.loadCalc(".","N2_emis.txspec","N2_abs.txspec","N2_half.txspec",fermis=0.45091878,binds=27.177975)
broad.loadCalc(".","N3_emis.txspec","N3_abs.txspec","N3_half.txspec",fermis=0.45090808,binds=27.122234,sites=1.245)
broad.loadCalc(".","N4_emis.txspec","N4_abs.txspec","N4_half.txspec",fermis=0.45088602,binds=27.177070,edge="L2")

# Initialize the broadening parameters
broad.initResolution(corelifetime=0.15,specResolution=1200,monoResolution=5000,disorder=0.5,XESscaling=0.5,XASscaling=0.5)
# Optionally you can scale specific bands in XEN. Use printBands() to determine where the bands are located.
# Then add the new argument XESbandScaling into initResolution()
# broad.printBands()
# XESbandScaling=[[0.1,0.2,0.2,0.4],[0.2,0.2,0.4,0.2],[0.3,0.2,0.1,0.5],[0.3,0.5,0.4,0.2]])

# Shift the spectra until the calculation aligns with the experimental
broad.Shift(XESshift=19.2,XASshift=20.2)
# Optionally you can shift specific bands in XES.
# Add the new argument into Shift()
# XESbandshift=[[30,33,30,20],[15,19.2,19.2,19.2],[30,33,30,20],[15,19.2,19.2,19.2]])

# Broaden the spectra
broad.broaden("/Users/cas003/opt/anaconda3/lib/python3.9/site-packages/BroadSword/")

# Export the broadened calculated spectra
# broad.export("Nitrogen")
```

### Functions
Below are the functions with their input criteria. If needed the docstrings will appear in Jupyter notebook using "shift+tab"

```
def loadExp(self, basedir, XES, XANES, fermi):
# Loads the measured experimental data (no headers allowed). Fermi energy is from the calculated ground state

def loadCalc(self, basedir, XES, XAS, XANES, fermis, binds, edge="K", sites=1):
# Loads the calculated data (no headers allowed). Fermis is the energy from the calculated excited state. Binding is from the ground state.
# Specifying the edge and number of sites are only required if they differ from the K edge and you have a different number of atoms between different inequivalent atoms.

def printBands(self):
# Prints out the location of the bands

def initResolution(self, corelifetime, specResolution, monoResolution, disorder, XESscaling, XASscaling, XESbandScaling=0)
# Specifies the broadening parameters based on instrument, general disorder, and lifetime broadening.
# An optional variable to scale the bands individually is available

def Shift(self,XESshift, XASshift, XESbandshift=0):
# Shifts the calculated spectra based on user input.
# An optional variable to shift the bands individually is available

def broaden(self, libpath="./")
# Broadens the calculated spectra. The library path will need to specified as described in the Installation segment

def export(self, filename)
# Exports the broadened data as a .csv file.
```
### Comments

Shifting only takes ~1s to plot. Comment out the broad.broaden() function and shift the unbroadened spectra first until it is in the proper position. Then include broad.broaden() in the notebook since this can take ~30s to compute.
