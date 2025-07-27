# This program calculates the implied volatility of a call or put option that matches the market-observed option price
# using the Black-Scholes model and the Newton-Raphson method for iterative approximation.

import numpy as np
import scipy.stats as st

# Initial Parameters and Market Quotes.
V_Market = 4 # Market Call/Put option price.
K = 120 # Strike price
Tau = 1 # Time to Maturity (Years)
R = 0.05 # Interest Rate
S_0 = 100 # Current Stock Price
SigmaInitial = 0.11 # Current Initial Implied Volatility
CP = "C" # C = Call : P = Put

def ImpliedVolatility(CP, S_0, K, SigmaInitial, Tau, R):
    Error = 1e10; # Initial Error
    OptPrice = lambda SigmaInitial: BS_CP_Option_Price(CP, S_0, K, SigmaInitial, Tau, R)
    Vega = lambda SigmaInitial: dV_Sigma(S_0, K, SigmaInitial, Tau, R)

    n = 1.0
    while Error > 10e-10:
        G       = OptPrice(SigmaInitial) - V_Market
        G_Prim  = Vega(SigmaInitial)
        Sigma_New = SigmaInitial - G / G_Prim

        #Error = abs(Sigma_New - SigmaInitial)
        Error = abs(G)
        SigmaInitial = Sigma_New;

        print('Iteration {0} with Error = {1}'.format(n, Error))

        n = n+1
    return SigmaInitial

def dV_Sigma(S_0, K, SigmaInitial, Tau, R):

    d2 = (np.log(S_0 / float(K)) + (R-0.5 * np.power(SigmaInitial, 2.0)) * Tau) / float(SigmaInitial * np.sqrt(Tau))
    Value = K * np.exp(-R * Tau) * st.norm.pdf(d2) * np.sqrt(Tau)
    return Value

def BS_CP_Option_Price(CP, S_0, K, SigmaInitial, Tau, R):
    #BS Call/Put Option Price
    d1 = (np.log(S_0 / float(K)) + (R + 0.5 * np.power(SigmaInitial, 2.0)) * Tau) / float(SigmaInitial * np.sqrt(Tau))
    d2 = d1 - SigmaInitial * np.sqrt(Tau)
    if str(CP).upper() == "C" or str(CP).lower() == "1":
        Value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-R * Tau)
    elif str(CP).upper() == "P" or str(CP).lower() == "-1":
        Value = st.norm.cdf(-d2) * K * np.exp(-R * Tau) - st.norm.cdf(-d1) * S_0
    return Value

Sigma_Imp = ImpliedVolatility(CP, S_0, K, SigmaInitial, Tau, R)
Message = ''' Implied Volatility for CallPrice = {}, Strike K = {}, Maturity T = {}, Interest Rate R = {} and
initial stock S_0 = {} equals Sigma_Imp = {:.7f}'''.format(V_Market, K, Tau, R, S_0, Sigma_Imp)

print(Message)

Val = BS_CP_Option_Price(CP, S_0, K, Sigma_Imp, Tau, R)
print('Option price for Imp Vol of {0} is equal to {1}'.format(Sigma_Imp, Val))