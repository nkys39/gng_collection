# Neural Gas

## Overview

Neural Gas is an unsupervised learning algorithm introduced by Martinetz and Schulten in 1991. It performs vector quantization without a fixed topology, using a rank-based neighborhood function that adapts all reference vectors based on their distance rank to the input.

## Key References

- Martinetz, T. and Schulten, K. (1991). "A Neural-Gas Network Learns Topologies"
- Martinetz, T. and Schulten, K. (1994). "Topology Representing Networks"

## Key Characteristics

- **No fixed topology**: Reference vectors are not constrained to a grid
- **Rank-based adaptation**: All vectors updated, strength decreases with rank
- **Soft competitive learning**: More robust than winner-take-all
- **Optional CHL**: Competitive Hebbian Learning can learn topology

## Algorithm

```
1. Initialize n reference vectors randomly

2. For each input signal x:
   a. Sort all vectors by distance to x:
      k_i = rank of vector i (0 = closest)

   b. Update all vectors:
      w_i += ε(t) * exp(-k_i / λ(t)) * (x - w_i)

   c. (Optional) Competitive Hebbian Learning:
      - Connect two closest vectors
      - Age edges from winner
      - Remove old edges

3. Decrease ε and λ over time
```

## Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| n_nodes | Number of reference vectors | 50 |
| λ_initial | Initial neighborhood range | n_nodes / 2 |
| λ_final | Final neighborhood range | 0.1 |
| ε_initial | Initial learning rate | 0.5 |
| ε_final | Final learning rate | 0.005 |
| max_age | Maximum edge age (for CHL) | 50 |

## Neighborhood Function

The key difference from SOM is the neighborhood function:

```
h(k) = exp(-k / λ)
```

where `k` is the **rank** (not grid distance). This means:
- Winner (k=0): h = 1.0
- Second (k=1): h = exp(-1/λ)
- Third (k=2): h = exp(-2/λ)
- etc.

## Differences from GNG

| Aspect | Neural Gas | GNG |
|--------|------------|-----|
| # Nodes | Fixed | Grows dynamically |
| Adaptation | All nodes (rank-based) | Winner + neighbors only |
| Topology | Learned via CHL | Learned via edges |
| Node insertion | None | Based on error |

## Advantages

- More robust convergence than k-means
- No fixed topology constraint
- Can learn arbitrary data distributions
- Soft competitive learning reduces sensitivity to initialization

## Limitations

- Number of reference vectors must be chosen in advance
- All vectors updated each step (computational cost)
- No automatic node insertion/removal
