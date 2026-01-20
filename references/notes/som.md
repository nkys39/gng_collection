# SOM (Self-Organizing Map)

## Overview

Self-Organizing Map (SOM), also known as Kohonen Map, is an unsupervised neural network algorithm introduced by Teuvo Kohonen in the 1980s. It uses competitive learning to produce a low-dimensional (typically 2D) discretized representation of the input space.

## Key References

- Kohonen, T. (1982). "Self-organized formation of topologically correct feature maps"
- Kohonen, T. (2001). "Self-Organizing Maps" (3rd ed.), Springer

## Key Characteristics

- **Fixed topology**: Neurons are arranged in a predefined grid (usually 2D)
- **Neighborhood function**: Based on grid distance, not data space distance
- **Winner-take-all**: Best Matching Unit (BMU) is selected for each input
- **Topology preservation**: Nearby neurons respond to similar inputs

## Algorithm

```
1. Initialize weight vectors randomly

2. For each input signal x:
   a. Find Best Matching Unit (BMU):
      BMU = argmin_i ||x - w_i||

   b. Update BMU and neighbors:
      w_i += η(t) * h(i, BMU, t) * (x - w_i)

      where:
      - η(t) = learning rate (decreases over time)
      - h(i, BMU, t) = neighborhood function (Gaussian on grid distance)

3. Decrease learning rate and neighborhood radius over time
```

## Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| grid_height | Height of neuron grid | 10 |
| grid_width | Width of neuron grid | 10 |
| σ_initial | Initial neighborhood radius | grid_size / 2 |
| σ_final | Final neighborhood radius | 0.5 |
| η_initial | Initial learning rate | 0.5 |
| η_final | Final learning rate | 0.01 |

## Differences from GNG

| Aspect | SOM | GNG |
|--------|-----|-----|
| Topology | Fixed grid | Dynamic (learned) |
| # Nodes | Fixed | Grows dynamically |
| Neighborhood | Grid-based | Edge-based |
| Edge learning | None (predefined) | Competitive Hebbian |

## Advantages

- Simple and well-understood
- Good for visualization (fixed 2D grid)
- Preserves topology of input space

## Limitations

- Fixed topology may not match data structure
- Number of neurons must be chosen in advance
- Poor adaptation to non-stationary distributions
