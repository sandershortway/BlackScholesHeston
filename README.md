# Python scripts for BlackScholes, Heston models
I've created a Python scripts for various use cases for the Black-Scholes model and the more advanced Heston model.

## Black-Scholes-Merton model
We want to be able to calculate the price of an European call and an European put option, provided we know the parameters that the model should adhere to. Furthermore, if we enter all but the volatility of the stock and a market price for a European call option, we want to be able to calculate the volatility that would have yielded the correct option price using the formula. This volatility is called the *implied volatility*. Since there is no direct formula to solve for the implied volatility we have implemented the bisection root-finding algorithm.

## Heston model
We want to be able to calculate the price of a European call option using both numerical methods for the integrals and using Monte Carlo methods. Also, to be able to calibrate the model provided you enter marketprices of European call options. This will be done using the Differential Evolution algorithm. Also, it should be able to determine the price of some complex exotic options.

# To-do list
**The Black-Scholes Model**
- [x] Calculate price of European call option
- [x] Calculate price of European put option
- [x] Calculate price of European call option using simulation
- [x] Calculate price of European put option using simulation
- [x] Calculate price of Forward contract
- [x] Calculate implied volatility
- [x] Surface plot of price for various strikes and maturities
- [x] Surface plot of implied volatility for various strikes and maturities

**The Heston Model**
- [x] Check if Feller condition is satisfied
- [x] Generate sample paths of S(t)
- [x] Generate sample paths of v(t)
- [x] Calculate price of European call option using numerical estimation of integrals
- [x] Calculate price of European call option using Monte Carlo simulation
- [x] Calculate price of European put option using put-call parity
- [x] Calibrating the Heston model to market data
- [x] Pricing of various exotic options using Monte Carlo simulation

**Brownian Motion**
- [x] Generate sample path of Brownian motion
- [x] Generate sample path of correlated Brownian motion
- [x] Generate sample path of Brownian motion with drift
- [x] Generate sample path of Geometric Brownian motion

**Historical Volatility**
- [x] Generate stock price graph
- [x] Determine the correlation between upwards movements and volatility
- [x] Generate historical volatility graph

**Differential Evolution**
- [x] Evaluate minimization objective function for particular set of parameters
- [x] Generate test data for testing algorithm

# Contact
Email: sanderkorteweg@gmail.com