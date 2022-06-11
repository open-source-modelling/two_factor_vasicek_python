import numpy as np
import pandas as pd
from typing import Any
from typing import List

class BrownianMotion():

    def __init__(self, x0: float = 0) -> None:

        self.x0 = float(x0)

    # This method generates a Weiner process (more commonly known as a Brownian Motion)
    def generate_weiner_process(self, T: int = 1, dt: float = 0.001, rho: float = None) -> Any:
        # GENERATE_WIENER_PROCESS calculates the XXX
        # W = generate_weiner_process(self, T, dt, rho)
        #
        # Arguments:
        #   self =
        #   T = 
        #   dt =
        #   rho = 
        #
        # Returns:
        #   W = 
        #
        # Example:
        #       
        # For more information see SOURCE

        N = int(T / dt)

        if not rho:

            W = np.ones(N) * self.x0

            for iter in range(1, N):

                W[iter] = W[iter-1] + np.random.normal(scale = dt)

            return W

        if rho:

            W_1 = np.ones(N) * self.x0
            W_2 = np.ones(N) * self.x0

            for iter in range(1, N):

                Z1 = np.random.normal(scale = dt)
                Z2 = np.random.normal(scale = dt)
                Z3 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

                W_1[iter] = W_1[iter-1] + Z1
                W_2[iter] = W_2[iter-1] + Z3

            return [W_1, W_2]

    # This simulates a temporal series of stock prices using the Black Scholes log normal model and the generated Weiner process
    def simulate_Black_Scholes(self, S0: int = 100, mu: float = 0.05, sigma: float = 0.3, T: int = 52, dt: float = 0.1, rho: float = None) -> pd.DataFrame:
        # SIMULATE_BLACK_SHOLES calculates the XXX
        # stock_price_simulation = simulate_Black_Scholes(self, S0, mu, sigma, T, dt, rho)
        #
        # Arguments:
        #   self =
        #   S0 = 
        #   mu =
        #   sigma = 
        #   T = 
        #   dt = 
        #   rho = 
        #
        # Returns:
        #   stock_price_simulation = 
        #
        # Example:
        #       
        # For more information see SOURCE
        
        N = int(T / dt)

        time, delta_t = np.linspace(0, T, num = N, retstep = True)

        stock_variation = (mu - (sigma**2/2)) * time

        weiner_process = sigma * self.generate_weiner_process(T, dt)

        S = S0*(np.exp(stock_variation + weiner_process))

        dict = {'Time' : time, 'Stock Price' : S}

        stock_price_simulation = pd.DataFrame.from_dict(data = dict)
        stock_price_simulation.set_index('Time', inplace = True)

        return stock_price_simulation

    # This simulates a temporal series of interest rates using the One Factor Vasicek mean reverting model and the generated Weiner process
    def simulate_Vasicek_One_Factor(self, r0: float = 0.1, a: float = 1.0, b: float = 0.1, sigma: float = 0.2, T: int = 52, dt = 0.1) -> pd.DataFrame:
        # SIMULATE_VASICEK_ONE_FACTOR calculates the XXX
        # interest_rate_simulation = simulate_Vasicek_One_Factor(self, r0, a, b, sigma, T, dt)
        #
        # Arguments:
        #   self =
        #   r0 = 
        #   a =
        #   b = 
        #   sigma = 
        #   T = 
        #   dt = 
        #
        # Returns:
        #   interest_rate_simulation = 
        #
        # Example:
        #       
        # For more information see SOURCE
        
        N = int(T / dt)

        time, delta_t = np.linspace(0, T, num = N, retstep = True)

        weiner_process = self.generate_weiner_process(T, dt)

        r = np.ones(N) * r0

        for t in range(1,N):
            r[t] = r[t-1] + a * (b - r[t-1]) * dt + sigma * (weiner_process[t] - weiner_process[t-1])

        dict = {'Time' : time, 'Interest Rate' : r}

        interest_rate_simulation = pd.DataFrame.from_dict(data = dict)
        interest_rate_simulation.set_index('Time', inplace = True)

        return interest_rate_simulation

    def simulate_Vasicek_Two_Factor(self, r0: List[float] = [0.1, 0.1], a: List[float] = [1.0, 1.0], b: List[float] = [0.1, 0.1], sigma: List[float] = [0.2, 0.2], rho: float = 0.5, T: int = 52, dt: float = 0.1) -> pd.DataFrame:
        # SIMULATE_VASICEK_TWO_FACTOR calculates the XXX
        # interest_rate_simulation = simulate_Vasicek_Two_Factor(self, r0, a, b, sigma, rho, T, dt)
        #
        # Arguments:
        #   self =
        #   r0 = 
        #   a =
        #   b = 
        #   sigma = 
        #   rho =
        #   T = 
        #   dt = 
        #
        # Returns:
        #   interest_rate_simulation = 
        #
        # Example:
        #       
        # For more information see SOURCE
        
        
            N = int(T / dt)

            time, delta_t = np.linspace(0, T, num = N, retstep = True)

            weiner_process = self.generate_weiner_process(T, dt, rho)

            weiner_process_e = weiner_process[0]
            weiner_process_s = weiner_process[1]

            r_e, s = np.ones(N) * r0[0], np.ones(N) * r0[1]

            a_e, a_s = a[0], a[1]

            b_e, b_s = b[0], b[1]

            sigma_e, sigma_s = sigma[0], sigma[1]

            for t in range(1,N):
                r_e[t] = r_e[t-1] + a_e * (b_e - r_e[t-1]) * dt + sigma_e * (weiner_process_e[t] - weiner_process_e[t-1])
                s[t] = s[t-1] + a_s * (b_s - s[t-1]) * dt + sigma_s * (weiner_process_s[t] - weiner_process_s[t-1])

            r_s = r_e - s

            dict = {'Time' : time, 'Foreign Interest Rate' : r_e, 'Domestic Interest Rate' : r_s}

            interest_rate_simulation = pd.DataFrame.from_dict(data = dict)
            interest_rate_simulation.set_index('Time', inplace = True)

            return interest_rate_simulation
