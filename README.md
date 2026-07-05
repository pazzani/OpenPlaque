# OpenPlaque Default Boundary Refinement

Updates `src/openplaque/boundary.py` to use tuned default parameters.

Default params:

```python
{'remove_small': True, 'min_component_voxels': 80, 'trim_lumen_adjacent': False, 'lumen_distance_voxels': 0, 'erode_core': False, 'erosion_iterations': 0, 'low_hu_threshold': None, 'high_hu_threshold': None}
```

Includes standalone Colab notebook:

```text
notebooks/07_UCLA_TPV_Tuned_Default_Refinement.ipynb
```
