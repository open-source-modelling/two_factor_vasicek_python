import pandas as pd
import numpy as np
from scipy import integrate

from Vasicek import BrownianMotion

class Swaption(object):

    def __init__(self,
                 type: str,
                 maturity: float = 1,
                 exercise_date: float = 0.5,
                 notional: float = 10**6,
                 fixed_rate: float = 0.1,
                 floating_leg_frequency: float = 0.5,
                 payer: bool = True):

        receiver = not payer
        self._maturity = maturity
        self._exercise_date = exercise_date
        self._notional = notional
        self._fixed_rate = fixed_rate
        self._floating_leg_frequency = floating_leg_frequency
        self._is_payer = payer
        self._is_receiver = receiver
        self._type = type


class ZeroCouponBond():

    def __init__(self,
                 maturity):

        self._T = maturity

    def price_Vasicek_Two_Factor(self, r0, a, b, sigma, rho, T, dt, nScen):

        interest_rate_simulation = pd.DataFrame()
        brownian_motion = BrownianMotion()
        for i in range(nScen):
            interest_rate_simulation = pd.concat([interest_rate_simulation,
            brownian_motion.simulate_Vasicek_Two_Factor(r0, a, b, sigma, rho, T, dt)['Real Interest Rate']],axis = 1)
        integral = interest_rate_simulation.apply(integrate.trapz)
        self._price = np.mean(np.exp(-integral))
        return self._price
