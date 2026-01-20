# GNG-U (Growing Neural Gas with Utility)

## Overview

GNG-U is an extension of the standard GNG algorithm proposed by Fritzke (1997) to handle non-stationary distributions. It introduces a **utility measure** for each node that tracks how useful the node is for reducing the overall network error.

## Key References

- Fritzke, B. (1997). "Some Competitive Learning Methods"
- Fritzke, B. (1999). "Be Busy and Unique — or Be History—The Utility Criterion for Removing Units in Self-Organizing Networks" in KI-99: Advances in Artificial Intelligence, Lecture Notes in Computer Science, vol 1701.

## Algorithm Differences from Standard GNG

### Additional Node Property
Each node has a **utility** value `U` in addition to the error `E`.

### Utility Update
After finding the winner (s1) and second winner (s2):
```
U_s1 += ||x - w_s2||^2 - ||x - w_s1||^2
```
or equivalently:
```
U_s1 += error_s2 - error_s1
```

This represents how much the total network error would increase if the winner node were removed (the second winner would then become the winner).

### Utility Decay
All node utilities are decayed:
```
U_i *= beta  (for all nodes i)
```
where `beta` is the same decay factor used for error (typically 0.995-0.9995).

### Node Removal Criterion
After inserting a new node, check if any node should be removed:
```
if max_error / min_utility > k:
    remove the node with minimum utility
```
where `k` is the utility threshold parameter (recommended value: 1.3).

## Complete Algorithm (GNG-U)

```
0. Initialize: Create two nodes with random positions, utility=0, error=0

1. Generate input signal x from distribution P(x)

2. Find winner s1 and second winner s2 (two nearest nodes)

3. Update error of winner:
   E_s1 += ||x - w_s1||^2

4. Update utility of winner:
   U_s1 += ||x - w_s2||^2 - ||x - w_s1||^2

5. Move winner and neighbors toward input:
   w_s1 += eps_b * (x - w_s1)
   w_n += eps_n * (x - w_n)  for all neighbors n of s1

6. Age edges from s1, create/reset edge (s1, s2)

7. Remove edges with age > max_age
   Remove nodes with no edges

8. Every lambda iterations:
   a. Find node q with maximum error
   b. Find neighbor f of q with maximum error
   c. Insert new node r between q and f
   d. Update edges: remove (q,f), add (q,r), (f,r)
   e. Update errors: E_q *= alpha, E_f *= alpha, E_r = (E_q + E_f) / 2
   f. Set utility of new node: U_r = (U_q + U_f) / 2

   g. [GNG-U specific] Check utility criterion:
      - Find node with minimum utility (min_u)
      - If max_error / U_min_u > k:
          Remove node with minimum utility

9. Decay all errors and utilities:
   E_i *= beta
   U_i *= beta

10. If stopping criterion not met, go to step 1
```

## Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| k | Utility threshold for node removal | 1.3 |
| beta | Error/utility decay rate | 0.995-0.9995 |
| (other params same as GNG) | | |

## Use Cases

- **Non-stationary distributions**: When the input distribution changes over time
- **Dynamic tracking**: Following moving objects or changing data patterns
- **Resource-constrained learning**: Automatically removing unnecessary nodes

## Implementation Notes

1. Utility should be initialized to a small positive value (e.g., 1.0) to avoid division by zero
2. The utility check is performed after node insertion (every lambda iterations)
3. Only one node is removed per utility check (the one with minimum utility)
4. New nodes inherit averaged utility from their parent nodes
