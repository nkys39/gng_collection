# GCS (Growing Cell Structures)

## Overview

Growing Cell Structures (GCS) is a self-organizing neural network introduced by Bernd Fritzke in 1994. It maintains a k-dimensional simplicial complex (triangular mesh in 2D) that grows dynamically to represent the input distribution. GCS is a precursor to the Growing Neural Gas (GNG) algorithm.

## Key References

- Fritzke, B. (1994). "Growing cell structures - a self-organizing network for unsupervised and supervised learning", Neural Networks, 7(9), 1441-1460
- Fritzke, B. (1993). "Growing Cell Structures—A Self-Organizing Network for Unsupervised and Supervised Learning", Technical Report TR-93-026, ICSI Berkeley

## Key Characteristics

- **Simplicial complex**: Maintains a triangular mesh (in 2D)
- **Dynamic growth**: Nodes are inserted to reduce error
- **Connected structure**: Always maintains a connected mesh
- **Local adaptation**: Winner and direct neighbors are updated

## Algorithm

```
1. Initialize with a k-simplex (triangle in 2D)

2. For each input signal x:
   a. Find winner: s = argmin_i ||x - w_i||

   b. Update error: E_s += ||x - w_s||

   c. Update winner: w_s += ε_b * (x - w_s)

   d. Update neighbors: w_n += ε_n * (x - w_n)

3. Every λ iterations:
   a. Find node q with maximum error
   b. Find neighbor f of q with maximum error
   c. Insert new node r between q and f:
      - w_r = (w_q + w_f) / 2
      - Remove edge (q, f)
      - Add edges (q, r) and (f, r)
      - Connect r to common neighbors of q and f
   d. Decay errors: E_q *= (1-α), E_f *= (1-α)

4. Decay all errors: E_i *= (1-β)
```

## Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| max_nodes | Maximum number of nodes | 100 |
| λ | Node insertion interval | 100 |
| ε_b | Winner learning rate | 0.1 |
| ε_n | Neighbor learning rate | 0.01 |
| α | Error decay on insertion | 0.5 |
| β | Global error decay | 0.005 |

## Differences from GNG

| Aspect | GCS | GNG |
|--------|-----|-----|
| Structure | Simplicial complex (mesh) | Arbitrary graph |
| Initialization | k-simplex (triangle) | 2 random nodes |
| Edge removal | None (mesh structure) | Age-based |
| Node removal | None | Isolated nodes |
| Topology | Always connected mesh | Can have multiple components |

## Advantages

- Always maintains a valid mesh structure
- Suitable for surface reconstruction
- Stable topology (no edge aging)
- Constant parameters (no time-dependent decay)

## Limitations

- Cannot represent disconnected regions
- Mesh structure may not fit all data distributions
- No mechanism to remove obsolete nodes
- Less flexible topology than GNG
