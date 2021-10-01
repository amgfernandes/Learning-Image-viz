
# %% 
'''Based on 
https://ilovesymposia.com/2019/10/24/introducing-napari-a-fast-n-dimensional-image-viewer-in-python/
'''


# %%

import numpy as np
from skimage import data, filters
import napari

from skimage import segmentation
from skimage import morphology

import os
from dask import array as da
 # %%
blobs_raw = np.stack([
    data.binary_blobs(length=256, n_dim=3, volume_fraction=f)
    for f in np.linspace(0.05, 0.5, 10)
])

blobs = filters.gaussian(blobs_raw, sigma=(0, 2, 2, 2))
print(blobs.shape)
(10, 256, 256, 256)


# %%

viewer = napari.view_image(blobs)

# %%

coins = data.coins()

viewer = napari.view_image(coins, name='coins')

edges = filters.sobel(coins)

edges_layer = viewer.add_image(edges, colormap='magenta', blending='additive')

pts_layer = viewer.add_points(size=5)
pts_layer.mode = 'add'
# annotate the background and all the coins, in that order

coordinates = pts_layer.data
coordinates_int = np.round(coordinates).astype(int)

markers_raw = np.zeros_like(coins)
markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(coordinates))
# raw markers might be in a little watershed "well".
markers = morphology.dilation(markers_raw, morphology.disk(5))

segments = segmentation.watershed(edges, markers=markers)

labels_layer = viewer.add_labels(segments - 1)  # make background 0

# %%

def threshold(image, t):
    arr = da.from_array(image, chunks=image.shape)
    return arr > t

all_thresholds = da.stack([threshold(coins, t) for t in np.arange(255)])

viewer = napari.view_image(coins, name='coins')
viewer.add_image(all_thresholds,
    name='thresholded', colormap='magenta', blending='additive'
)
# %%
