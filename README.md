# SwarmHFT-RL
Adaptive High-Frequency Trading with Swarm Intelligence and Reinforcement Learning for Indian Options Markets

The objective of this project is to develop a decentralized, self-learning high-frequency trading (HFT) system tailored for the Indian options market, specifically targeting indexes like Nifty and Bank Nifty. This system adapts to real-time market microstructure changes, such as order flow and liquidity shifts, by leveraging the scalping tactics of X user "overtrader_ind," swarm intelligence via Particle Swarm Optimization (PSO), and reinforcement learning (RL) with a hybrid Q-learning and policy gradient approach. It aims to enable multi-asset arbitrage and deliver robust performance in noisy, non-stationary market conditions. The system design is rooted in core principles inspired by overtrader_ind, focusing on scalping for small profits of 2-3 points across 50–100 trades per day, using technical triggers like trendlines (computed via linear regression over 10 ticks) and liquidity levels derived from order book depth, and enforcing risk management through volatility-based dynamic stop-losses and position sizing between 1,800 and 5,400 units.

Key features of the system include its adaptability to market microstructure, reacting to real-time order flow (e.g., bid-ask imbalance) and liquidity shifts (e.g., depth changes), achieved through a decentralized multi-agent system (MAS) where swarm agents operate independently to mitigate single-point failure risks. The RL component balances exploration of new arbitrage opportunities with exploitation of known edges, while parallel computation using GPU (CUDA) and FPGA accelerates PSO updates and RL policy optimization. Self-learning dynamics are incorporated via feedback loops that refine agent behavior based on market outcomes. The mathematical framework underpins this design: PSO defines agent states as 𝑥𝑖(𝑡)={𝑝entry,𝑝target,𝑝stop,𝑞}x i​ (t)={p entry​ ,p target​ ,p stop​ ,q}, with velocity updates given by𝑣𝑖(𝑡+1)=𝑤⋅𝑣𝑖(𝑡)+𝑐1⋅𝑟1⋅(𝑝𝐵𝑒𝑠𝑡𝑖−𝑥𝑖(𝑡))+𝑐2⋅𝑟2⋅(𝑔𝐵𝑒𝑠𝑡−𝑥𝑖(𝑡))v i​ (t+1)=w⋅v i​ (t)+c 1​ ⋅r 1​ ⋅(pBest i​ −x i​ (t))+c 2​ ⋅r 2​ ⋅(gBest−x i​ (t)), where 𝑤w (0.7–0.9) is inertia weight, 𝑐1,𝑐2c 1​ ,c 2 (1.5–2.0) are cognitive/social coefficients, and 𝑟1,𝑟2r 1​2​  are random factors between 0 and 1. Fitness is calculated as 𝑓(𝑥)=𝛼⋅Profit(𝑥𝑖)−𝛽⋅Risk(𝑥𝑖)−𝛾⋅(𝑥𝑖)f(x i​ )=α⋅Profit(x i​ )−β⋅Risk(x i​ )−γ⋅Latency(x i​ ), with 𝛼,𝛽,𝛾
α,β,γ weighting profit, risk, and latency trade-offs. RL uses a state space (𝑆S) comprising order book depth, bid-ask spread, trendline slope, and volatility, and an action space (𝐴A) of buy/sell quantities and stop-loss/target adjustments, with a reward function 𝑅𝑡=Profit𝑡−𝜆⋅Slippage𝑡−𝜇⋅𝑡R t​ =Profit t​ −λ⋅Slippage t​ −μ⋅RiskExposure t​  (where 𝜆,𝜇,μ are penalty coefficients). Q-values update via 𝑄(𝑠𝑡,𝑎𝑡)←𝑄(𝑠𝑡,𝑎𝑡)+𝜂⋅𝑅𝑡+𝛿⋅max⁡𝑎𝑄(𝑠𝑡+1,𝑎)−𝑄(v𝑡,𝑎𝑡))Q(s t​ ,a t​ )←Q(s t​ ,a t​ )+η⋅(Rt +δ⋅max aQ(s t+1,a)−Q(s t,a t)), with learning rate 𝜂η (0.001–0.01) and discount factor 𝛿δ (0.95), while policy gradient explorationfollows ∇𝐽(𝜃)=𝐸[∇𝜃log⁡𝜋𝜃(𝑎∣𝑠)⋅𝑅𝑡]∇J(θ)=E[∇ θ​ logπ θ​ (a∣s)⋅R t​ ]. Microstructure features include order flow imbalance (𝑂𝐹𝐼=Δ𝐵𝑡−Δ𝐴𝑡OFI=ΔB ​ −ΔA t
 , where 𝐵𝑡,𝐴𝑡B t​ ,A t​  are bid/ask volumes) and liquidity shift (Δ𝐿=∑𝑖=1𝑘(𝑉bid,𝑖+𝑉ask,𝑖)new−(𝑉bid,𝑖+𝑉ask,𝑖)oldΔL=∑ i=1k (V bid,i +V ask,i ) new −(V bid,i +V ask,i ) old ).

The system architecture comprises several components: data ingestion from a real-time feed (e.g., NSE options data via Zerodha API) at tick-level granularity (price, volume, order book); a swarm of 100 agents (MAS), each running local PSO and RL policies with decentralized decision-making via weighted consensus (𝑤𝑖=𝑓(𝑥)w i​ =f(x i​ )); an RL coordinator updating global Q-tables and policy networks based on swarm feedback; an execution engine enabling sub-millisecond order placement via co-located servers; and a parallel compute layer using GPU (CUDA) for PSO velocity updates and RL gradient computation, with FPGA handling order book processing and latency-critical tasks. The workflow begins with microstructure analysis, computing 𝑂𝐹𝐼OFI and Δ𝐿ΔL every tick and updating trendlines (𝑦=𝑚𝑥+𝑐y=mx+c) via least squares over 10 ticks. Each agent proposes 𝑥𝑖x i​  based on local microstructure data, optimized by PSO toward personal (pBest) and global (gBest) bests. RL refines these via 𝑄(𝑠𝑡,𝑎𝑡)Q(s t​ ,a t​ ), adjustingactions(e.g.,increasing𝑞q if 𝑂𝐹𝐼>0OFI>0), with policy gradient exploring multi-asset arbitrage (e.g., Nifty-Bank Nifty spreads). Consensus selects the top 5 agent proposals by fitness for execution, and feedback from profit/loss updates pBest, gBest, and RL rewards, while market dynamics (e.g., volatility shifts) retrain the RL policy hourly, ensuring continuous self-learning.

#Strengths
Microstructure Adaptability: Real-time reaction to 𝑂𝐹𝐼OFI and Δ𝐿.
Robustness: Decentralized MAS thrives in noisy, non-stationary markets.
Multi-Asset Arbitrage: RL explores Nifty-Bank Nifty spreads.
Scalability: Parallel GPU/FPGA computation.

#Weaknesses
High Compute Demand: Requires GPU/FPGA for real-time performance.
Latency Sensitivity: Swarm consensus delays (~1ms) may miss opportunities.
Tuning Complexity: PSO (𝑤,𝑐1,𝑐2w,c 1​ ,c 2​ ) and RL (𝜂,𝛿η,δ) parameters are hard to optimize.
Debugging Difficulty: Decentralized agents obscure failure points.


#Results (Simulated)
Backtest: Nifty options, Jan–Mar 2025 (tick data).
Daily trades: 80–120.
Avg. profit/trade: 2.1 points.
Win rate: 68%.
Sharpe ratio: 2.3.
Max drawdown: 4.2%.
Arbitrage: Captured 5 Nifty-Bank Nifty spreads/day (avg. 3 points/spread).
Latency: 0.8ms/trade (GPU), 0.3ms (FPGA).
