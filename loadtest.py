from BroadSword import *

broad = Broaden()
broad.loadExp(".","expxes.txt","expxanes.txt")
broad.loadCalc(".","actualxes.txt","actualxas.txt","actualxanes.txt")
broad.loadCalc(".","xes.txt","xas.txt","xanes.txt",8)
broad.FindBands()
broad.initParam(0.45,[0.46,0.47],[27.2,27.3],["K","K"])
broad.Shift(5,6,10)
