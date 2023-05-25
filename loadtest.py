from BroadSword import *

broad = Broaden()
broad.loadExp(".","expxes.txt","expxanes.txt")
broad.loadCalc(".","actualxes.txt","actualxas.txt","actualxanes.txt")
broad.loadCalc(".","xes.txt","xas.txt","xanes.txt",8)
broad.FindBands()
broad.initParam(0.449965,[0.45062079,0.45091878],[27.17623,27.1222],["K","K"])
broad.Shift(0,0)
broad.Shift(19.2,20.2)
