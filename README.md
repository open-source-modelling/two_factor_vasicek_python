# Two factor Vasicek model
A simple model for calculating the nominal interest rates. Used to add inflation to the simulation of interest rates. The model has two sources of randomnes (Two correlated Brownian motions). 

## Problem
When modeling the nominal rate, both the real rate of return and the inflation should be considered. The correlation between them means that one should use a multifactor model as opposed to two independent models. Additionaly, there is a robust body of literature showing that both real rates and the inflation are mean-reverting.

## Solution
The simplest model for modeling real rates and inflation together is the multifactor Vasicek model https://en.wikipedia.org/wiki/Vasicek_model. The Vasicek model is a short rate model describing the evolution of rates. Both the real rate process and the inflation rate process are assumed to follow a Vasicek model. The movement of the two curves is given by a two dimensional correlated Brownian motion.
### Input
TBD
### Output
TBD
## Getting started
``` python
import numpy as np
import pandas as pd
import datetime as dt

from Vasicek import BrownianMotion

from Pricing import ZeroCouponBond

from IPython.display import display
import matplotlib.pyplot as plt

brownian = BrownianMotion()
interest_rate_simulation = brownian.simulate_Vasicek_Two_Factor()

display(interest_rate_simulation)

interest_rate_simulation.plot(figsize = (15,9), grid = True)
plt.legend()
plt.show()

# Defining a zero curve for the example
Dates = [[2010,1,1], [2011,1,1], [2013,1,1], [2015,1,1], [2017,1,1], [2020,1,1], [2030,1,1]]
curveDates = []
for date in Dates:
    curveDates.append(dt.date(date[0],date[1],date[2]))

zeroRates = np.array([1.0, 1.9, 2.6, 3.1, 3.5, 4.0, 4.3])/100

plt.figure(figsize = (15,9))
plt.plot(curveDates,zeroRates)
plt.title('Yield Curve for ' + str(curveDates[0]))
plt.xlabel('Date')
plt.ylabel('Rate')
plt.show()

zero_coupon_bond = ZeroCouponBond(1)
zero_coupon_bond.price()
print(zero_coupon_bond._price)
```
