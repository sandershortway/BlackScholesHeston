# Python scripts for BlackScholes, Heston models
I've created a Python scripts for various use cases for the Black-Scholes model and the more advanced Heston model.

## Black-Scholes-Merton model
We want to be able to calculate the price of an European call and an European put option, provided we know the parameters that the model should adhere to. Furthermore, if we enter all but the volatility of the stock and a market price for a European call option, we want to be able to calculate the volatility that would have yielded the correct option price using the formula. This volatility is called the *implied volatility*. Since there is no direct formula to solve for the implied volatility we have implemented the bisection root-finding algorithm.

## Heston model
We want to be able to calculate the price of a European call option using both numerical methods for the integrals and using Monte Carlo methods. Also, to be able to calibrate the model provided you enter marketprices of European call options. Also, it should be able to determine the price of some complex exotic options.

# To-do list
**The Black-Scholes Model**
- [x] Calculate price of European call option
- [x] Calculate price of European put option
- [ ] Calculate price of European call option using simulation
- [ ] Calculate price of Forward contract
- [x] Calculate implied volatility
- [ ] Implement systemic way to import marketdata by reading datalist.txt
- [x] Surface plot of price for various strikes and maturities
- [x] Surface plot of implied volatility for various strikes and maturities

**The Heston Model**
- [ ] Check if Feller condition is satisfied
- [ ] Generate sample paths for various parameters
- [ ] Calculate price of European call option using numerical estimation of integrals
- [ ] Calculate price of European call option using Monte Carlo simulation
- [ ] Calculate price of European put option using put-call parity
- [ ] Calibrating the Heston model to market data
- [ ] Pricing of various exotic options using Monte Carlo simulation

**Brownian Motion**
- [x] Generate sample path of Brownian motion
- [x] Generate sample path of correlated Brownian motion
- [x] Generate sample path of Brownian motion with drift
- [x] Generate sample path of Geometric Brownian motion

# Information on the data
The data that will be relevant is the maturity time in days and strike price of various European call options for a particular stock. The data is currently in .csv format which I was able to download from barchart.com. I cleaned up the data by removing rows and columns which were not relevant. The model does not do this for us (yet). The first column contains the strike price and the second column the market price for a European call option.

# Contact
Email: sanderkorteweg@gmail.com