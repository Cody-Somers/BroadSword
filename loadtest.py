from BroadSword import *

broad = Broaden()
broad.loadExp(".","expxes.txt","expxanes.txt")
broad.loadCalc(".","actualxes.txt","actualxas.txt","actualxanes.txt")
broad.loadCalc(".","xes.txt","xas.txt","xanes.txt",8)
broad.FindBands()
broad.initParam(0.449965,[0.45062079,0.45091878],[27.17623,27.1222],["K","K"]) # Should put some of these into the loadCalc file. Easier to keep track of
broad.Shift(19.2,20.2)
broad.initResolution(0.15,1200,5000,0.5,0.5,0.5)
broad.broaden()