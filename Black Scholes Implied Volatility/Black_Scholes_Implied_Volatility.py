# This program calculates the implied volatility of a call or put option that matches the market-observed option price
# using the Black-Scholes model and the Newton-Raphson method for iterative approximation.

import numpy as np
import scipy.stats as st

V_Market = 4 # Market Call/Put option price.
K = 120 # Strike price
Tau = 1 # Time to Maturity (Years)
R = 0.05 # Interest Rate
S_0 = 100 # Current Stock Price
SigmaInitial = 0.11 # Current Initial Implied Volatility
CP = "C" # C = Call : P = Put

def ImpliedVolatility(CP, S_0, K, SigmaInitial, Tau, R, Max_Iterations = 100, Tolerance = 1e-6):
    """Calculate the Implied Volatility using Newton-Raphson."""
    if S_0 <= 0:
        raise ValueError(f"Invalid input: S_0={S_0}")
    elif K <= 0:
        raise ValueError(f"Invalid input: K={K}")
    elif Tau <= 0:
        raise ValueError(f"Invalid input: Tau={Tau}")
    elif SigmaInitial <= 0:
        raise ValueError(f"Invalid input: SigmaInitial={SigmaInitial}")

    for n in range(1, Max_Iterations + 1):
        D1, D2 = Calculate_D_Parameters(S_0, K, SigmaInitial, Tau, R)
        G = BS_CP_Option_Price(CP, S_0, K, SigmaInitial, Tau, R, D1, D2) - V_Market
        G_Prime = dV_Sigma(S_0, K, SigmaInitial, Tau, R, D1, D2)
        if abs(G_Prime) < Tolerance:
            raise ValueError("Vega too small to converge.")

        Sigma_New = SigmaInitial - G / G_Prime
        Error = abs(G)
        print(f"Iteration {n} with Error = {Error}")
        if Error < Tolerance:
            return Sigma_New

        SigmaInitial = Sigma_New
    raise ValueError("Failed to converge.")

def Calculate_D_Parameters(S_0, K, SigmaInitial, Tau, R):
    """Calculate the D1 and D2 parameters."""
    if Tau <= 0:
        raise ValueError(f"Invalid Input: Tau={Tau}")
    D1 = (np.log(S_0 / K) + (R + 0.5 * SigmaInitial ** 2) * Tau) / (SigmaInitial * np.sqrt(Tau))
    D2 = D1 - SigmaInitial * np.sqrt(Tau)
    return D1, D2

def dV_Sigma(S_0, K, SigmaInitial, Tau, R, D1, D2):
    """Calculate Vega - The sensitivity to volatility changes."""
    return S_0 * st.norm.pdf(D1) * np.sqrt(Tau)

def BS_CP_Option_Price(CP, S_0, K, SigmaInitial, Tau, R, D1, D2):
    """Calculate the Black-Scholes option price for call or put."""
    if CP.upper() == "C":
        return st.norm.cdf(D1) * S_0 - st.norm.cdf(D2) * K * np.exp(-R * Tau)
    elif CP.upper() == "P":
        return st.norm.cdf(-D2) * K * np.exp(-R * Tau) - st.norm.cdf(-D1) * S_0
    raise ValueError("Invalid Option Type")

Sigma_Imp = ImpliedVolatility(CP, S_0, K, SigmaInitial, Tau, R)
print(f"Implied Volatility Calculation Results:  Market Price: ${V_Market} "
      f"Strike: ${K}  Time to Maturity: {Tau} years  Risk-free Rate: {R:.1%}  Current Stock Price: ${S_0} "
      f"Implied Volatility: {Sigma_Imp:.4f} ({Sigma_Imp:.2%})")

D1, D2 = Calculate_D_Parameters(S_0, K, Sigma_Imp, Tau, R)
Val = BS_CP_Option_Price(CP, S_0, K, Sigma_Imp, Tau, R, D1, D2)
print(f'Option price for Implied Vol of {Sigma_Imp:.4f} is equal to ${Val:.6f}')