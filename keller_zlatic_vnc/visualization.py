""" Tools to help with visualizing Keller Zlatic VNC data. """

from typing import Sequence, Tuple
import numpy as np

from janelia_core.visualization.custom_color_maps import make_red_green_c_map
from janelia_core.visualization.custom_color_maps import generate_two_param_hsv_map
from janelia_core.visualization.custom_color_maps import MultiParamCMap
from janelia_core.visualization.volume_visualization import visualize_rgb_max_project

def gen_coef_p_vl_cmap(coef_vls: np.ndarray, p_vls: np.ndarray, n_coef_clrs: int = 1024, n_p_vl_vls: int = 1024,
                       clim_percs: Sequence[float] = [.1, 99.9], max_p_vl: float = None) -> MultiParamCMap:
    """ Generates a MultiParamCMap for a range of coefficient values as colors and p-values as values of hsv colors.

    Args:
        coef_vls: The coefficient values that will be displayed

        p_vls: The p-values that will be displayed

        n_coef_clrs: The number of distint coefficient colors that should be in the colormap

        n_p_vl_vls: The number of distint p-value values (that is hsv values) that should be in the colormap

        max_p_vl: The p-value at and above which is shown as black. If not provided, will be the largest
        p-value in p_vls.

    Raises:
        ValueError: If n_coef_clrs or n_p_vl_vls is less than 2.
    """

    if n_coef_clrs < 2:
        raise(ValueError('n_coef_clrs must be greater than 1'))

    if n_p_vl_vls < 2:
        raise(ValueError('n_p_vl_vls must be greater than 1'))

    min_coef = np.percentile(coef_vls, clim_percs[0])
    max_coef = np.percentile(coef_vls, clim_percs[1])
    max_abs_coef = np.max(np.abs([min_coef, max_coef]))

    coef_step = 2*max_abs_coef/n_coef_clrs

    min_p_vl = np.min(p_vls)
    if max_p_vl is None:
        max_p_vl = np.max(p_vls)

    if max_p_vl < min_p_vl:
        # Handle this special, degenerate case here - in this case, everything gets mapped to black
        return MultiParamCMap(param_vl_ranges=[(-1, 1, .1), (min_p_vl, min_p_vl + .2, .1)], clrs=np.zeros([20, 3, 3]))

    p_vl_step = (max_p_vl - min_p_vl)/n_p_vl_vls

    coef_range = (-max_abs_coef, max_abs_coef+coef_step, coef_step)
    p_vl_range = (max_p_vl+p_vl_step, min_p_vl, -p_vl_step)

    return generate_two_param_hsv_map(clr_param_range=coef_range, vl_param_range=p_vl_range,
                                      p1_cmap=make_red_green_c_map(n_coef_clrs),
                                      clims=(-max_abs_coef, max_abs_coef), vllims=(max_p_vl, min_p_vl))


def visualize_coef_p_vl_max_projs(vol: np.ndarray, cmap: MultiParamCMap, cmap_coef_range: Sequence = None,
                                  cmap_p_vl_range: Sequence = None, title: str = None):
    """ Generates an image of max projections of rgb volumes of combined coefficient values and p-values.

    Args:
        vol: The volume to visulize.  Dimensions are [z, x, y, rgb]

        cmap: The colormap used to produce the original volume.

        cmap_coef_range: The range of values to generate colormap keys for.  Of the form (start, stop, step). If None,
        the range of values within saturation limits of the colormap will be used.

        cmap_p_vl_range: The range of values to generate p-value colormap keys for. Of the same form as cmap_coef_range.
        If None, the range of values within saturation limits of the colormap will be used.

        title: The title for the figure.

    """

    # Generate the cmap image
    if cmap_coef_range is None:
        cmap_coef_range = cmap.param_vl_ranges[0]
    if cmap_p_vl_range is None:
        cmap_p_vl_range = cmap.param_vl_ranges[1]

    coef_vls = np.sort(np.arange(*cmap_coef_range))
    p_vl_vls = np.sort(np.arange(*cmap_p_vl_range))

    n_coef_vls = len(coef_vls)
    n_p_vl_vls = len(p_vl_vls)

    param_grids = np.mgrid[0:n_coef_vls, 0:n_p_vl_vls]
    coef_vl_grid = coef_vls[param_grids[0]]
    p_vl_grid = p_vl_vls[param_grids[1]]

    cmap_im = cmap[coef_vl_grid, p_vl_grid]

    min_cmap_p_vl = np.min(p_vl_vls)
    max_cmap_p_vl = np.max(p_vl_vls)
    min_coef_vl = np.min(coef_vls)
    max_coef_vl = np.max(coef_vls)

    visualize_rgb_max_project(vol=vol, cmap_im=cmap_im,
                              cmap_extent=(min_cmap_p_vl, max_cmap_p_vl, min_coef_vl, max_coef_vl),
                              cmap_xlabel='$\log(p)$', cmap_ylabel='coef vl ($\Delta F / F$)',
                              title=title)
