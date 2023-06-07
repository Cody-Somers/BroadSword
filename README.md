# BroadSword
Converting the BroadSword program written by Teak Boyko from the Canadian Light Source in Saskatoon, SK, CA.
The program has been transcribed into python so that it can be compatible with jupyter notebook.

## Installation

Install the package from PyPi with the pip package manager. This is the recommended way to obtain a copy for your local machine and will install all required dependencies.

```
    $ pip install BroadSword
```

You will also need [Jupyter Notebook](https://github.com/jupyter) together with python 3 on your local machine.

## Example Program

```
# Specify the base directory for the scans
# e.g. put it in the scan directory or specify a full path name
basedir = '.'
#basedir = "/Users/cas003/Downloads/Beamtime/DataFiles"

## Setup necessary inputs
from BroadSword.BroadSword import *
from bokeh.io import output_notebook
output_notebook(hide_banner=True)

broad = Broaden()
broad.loadExp(".","N_test_XES.txt","N_test_XAS.txt",0.44996547)
broad.loadCalc(".","N1_emis.txspec","N1_abs.txspec","N1_half.txspec",0.45062079,27.176237)
broad.loadCalc(".","N2_emis.txspec","N2_abs.txspec","N2_half.txspec",0.45091878,27.177975)
broad.loadCalc(".","N3_emis.txspec","N3_abs.txspec","N3_half.txspec",0.45090808,27.122234)
broad.loadCalc(".","N4_emis.txspec","N4_abs.txspec","N4_half.txspec",0.45088602,27.177070)
broad.FindBands()
broad.Shift(19.2,20.2)
broad.initResolution(0.15,1200,5000,0.5,0.5,0.5)
#broad.broaden()
broad.broaden("/Users/cas003/opt/anaconda3/lib/python3.9/site-packages/BroadSword/")
```

### Functions

loadExp(base directory, measured XES spectrum, measured XANES spectrum, calculated ground state fermi energy)

loadCalc(base directory, calculated XES spectrum, calculated XAS spectrum, calculated XANES spectrum, calculated excited state fermi energy, calculated ground state binding energy)

FindBands() Will be depreciated

Shift(XAS shift, XES shift)

initResolution(XES corehole lifetime, spectrometer resolution, monochromator resolution, disorder, XES corehole lifetime scaling, XAS corehole lifetime scaling)

broaden(Path to .so or .dylib file)
