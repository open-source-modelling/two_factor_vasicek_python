# Two factor Vasicek model
A simple model for calculating the nominal interest rates. Used to add inflation to the simulation of interest rates. The model has two sources of randomnes (Two correlated Brownian motions). 

## Problem
When modeling the nominal rate, both the real rate of return and the inflation should be considered. The correlation between them means that one should use a multifactor model as opposed to two independent models. Additionaly, there is a robust body of literature showing that both real rates and the inflation are mean-reverting.

## Solution
The simplest model for modeling real rates and inflation together is the multifactor Vasicek model https://en.wikipedia.org/wiki/Vasicek_model. The Vasicek model is a short rate model describing the evolution of rates. Both the real rate process and the inflation rate process are assumed to follow a Vasicek model. The movement of the two curves is given by a two dimensional correlated Brownian motion.
### Input
Vasicek model simulator:
 - r0 ... Starting annualized real rate and inflation rate. ex. if the annualized real rate is 1.4% and inflation is 6%, then r0 = [0.014, 0.06]   
 - a ... mean reversion speed for the real and inflation process. ex. if the reversion factor is 0.8 for real rates and 1 for inflation, a = [0.8, 1]         
 - b ... long term mean level for the real and inflation process. ex. if the long term real rate is 1% and long term inflation is 1.5%, b = [0.01, 0.015]    
 - sigma ... instantaneous volatility of the real and inlfation process. ex. volatility of the real rate process is 5% and inflation process is 4%, sigma = [0.05, 0.04] 
 - rho ... correlation between the stochastic noise that generates the two processess. ex. if the calculated correlation coefficient is 0.t, rho = 0.6            
 - T ... modelling time horizon. ex. if time horizon is 25 years, T = 25               
 - dt ... time increments. ex. time increments are 6 months, dt = 0.5             

Vasicek model pricing:
TBD

### Output
Vasicek model simulator:
 - interest_rate_simulation is a pandas dataframe with one sample path generated by the model. One for the real rate process and the other for the nominal rates (real rate + inflation rate)

Vasicek model pricing:
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

# Vasicek model simulator
r0 = [0.014, 0.06]   # Starting annual real rate and annual inflation rate
a = [0.8, 1]         # mean reversion speed for real rate and inflation
b = [0.01, 0.015]    # long term trend for real rate and inflation
sigma = [0.05, 0.04] # annualized volatility of real rate and inflation process
rho = 0.6            # correlation
T = 25               # time horizon
dt = 0.5             # time increment

brownian = BrownianMotion()
interest_rate_simulation = brownian.simulate_Vasicek_Two_Factor(r0, a, b, sigma, rho, T, dt)

display(interest_rate_simulation)

interest_rate_simulation.plot(figsize = (15,9), grid = True)
plt.legend()
plt.show()

# Vasicek model pricing

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
