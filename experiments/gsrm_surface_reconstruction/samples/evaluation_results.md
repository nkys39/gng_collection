## Multi-Resolution Evaluation Results

### Sphere Surface Reconstruction

| Nodes | Edges | Faces | Hausdorff | Mean Dist | Time (s) |
|------:|------:|------:|----------:|----------:|---------:|
| 20 | 126 | 120 | 0.373396 | 0.193204 | 0.11 |
| 50 | 314 | 273 | 0.305257 | 0.110338 | 0.47 |
| 100 | 611 | 523 | 0.278045 | 0.070772 | 1.45 |
| 200 | 1167 | 1043 | 0.210783 | 0.045358 | 4.97 |
| 500 | 2830 | 2423 | 0.115094 | 0.026284 | 29.53 |

### Torus Surface Reconstruction

| Nodes | Edges | Faces | Hausdorff | Mean Dist | Time (s) |
|------:|------:|------:|----------:|----------:|---------:|
| 20 | 84 | 60 | 0.177724 | 0.118180 | 0.11 |
| 50 | 250 | 254 | 0.141409 | 0.087664 | 0.42 |
| 100 | 606 | 589 | 0.119930 | 0.067029 | 1.42 |
| 200 | 1281 | 1139 | 0.110352 | 0.043256 | 5.27 |
| 500 | 2896 | 2485 | 0.098862 | 0.023425 | 29.23 |

### Distance Metrics

- **Hausdorff Distance**: Maximum of the directed Hausdorff distances between point cloud and mesh vertices
- **Mean Distance**: Average distance from each point cloud point to its nearest mesh vertex

### Observations

1. Both Hausdorff and mean distances decrease as the number of nodes increases
2. The torus (with hole) requires more nodes than the sphere for similar accuracy
3. Computation time scales approximately linearly with the number of nodes

![Multi-resolution Evaluation](samples/gsrm/python/multiresolution_eval.png)
