"""Observable RGB-window morphology on the existing native segmentation rule.

Non-differentiable CPU audit only. Pixels are not micrometres; connected RGB
regions are not certified crystallographic grains.
"""
from __future__ import annotations
import numpy as np
from skimage.measure import label, regionprops
from .local_descriptors import segment_native, validate_settings

SCHEMA = 'native_rgb_hard_mechanics_v2'
NAMES = ['log_area_weighted_diameter_px', 'log_region_density_per_px2',
         'coarse_area_fraction', 'area_weighted_aspect']
LAYOUT_NAME = 'centroid_density_cv'


def describe_hard(rgb, settings=None, *, spatial_block=64, include_layout=False):
    s = validate_settings(settings or {})
    if type(spatial_block) is not int or spatial_block < 1:
        raise ValueError('Spatial block must be a positive integer')
    regions, quality = segment_native(rgb, s)
    visible = label(regions, background=0, connectivity=1)
    props = [p for p in regionprops(visible) if p.area >= s['min_region_area_px']]
    width = 5 if include_layout else 4
    quality = {**quality, 'retained_region_count': len(props), 'is_physical_grain_measurement': False}
    if len(props) < s['min_regions']:
        return np.full(width, np.nan), False, quality
    area = np.asarray([p.area for p in props], float)
    diameter = np.sqrt(4*area/np.pi)
    aspect = []
    for p in props:
        eigen = np.clip(p.inertia_tensor_eigvals, 0, None)+1/12
        aspect.append(np.sqrt(eigen.max()/eigen.min()))
    values = [np.log(np.average(diameter, weights=area)), np.log(len(props)/visible.size),
              area[diameter >= s['coarse_diameter_px']].sum()/area.sum(), np.average(aspect, weights=area)]
    if include_layout:
        h, w = visible.shape
        counts = np.zeros(((h+spatial_block-1)//spatial_block, (w+spatial_block-1)//spatial_block))
        for p in props:
            y, x = p.centroid
            counts[int(y)//spatial_block, int(x)//spatial_block] += 1
        heights = np.minimum(spatial_block, h-np.arange(counts.shape[0])*spatial_block)
        widths = np.minimum(spatial_block, w-np.arange(counts.shape[1])*spatial_block)
        densities = counts/(heights[:, None]*widths[None, :])
        values.append(float(densities.std()/densities.mean()))
    values = np.asarray(values, float)
    return values, bool(np.isfinite(values).all()), quality
