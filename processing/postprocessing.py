from scipy.ndimage import label
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_closing, generate_binary_structure
from scipy.ndimage import binary_opening
from scipy.spatial import ConvexHull
from scipy.ndimage import iterate_structure
import warnings
try:
    import itk
except ImportError:
    itk = None  # STAPLE fusion falls back to probabilistic mean
try:
    from skimage.morphology import convex_hull_image
except ImportError:
    convex_hull_image = None
from scipy.ndimage import gaussian_filter


## Recommended Postprocessing Pipeline (per organ)
'''
Raw mask
  │
  ├── Step 1: Remove small components (<100 voxels)
  ├── Step 2: Morphological closing (radius=2, for tubular organs only)
  ├── Step 3: Hole filling
  ├── Step 4: Keep largest connected component (for solid organs)
  │            OR keep top-K components (for paired/multi-lobe organs)
  ├── Step 5: Boundary smoothing (optional, before shape radiomics)
  │
  └── Output: cleaned mask → QC → radiomics
'''

# currently, these are the only postprocessing functions being used

def _nonzero_bbox(mask: np.ndarray):
    """Return tuple of slices for the tight bbox of nonzero voxels, or None if empty.

    Optimization helper for keep_largest_component / fill_holes — running scipy
    over a tight crop is 10-50x faster than over a 512^3 volume, while
    producing byte-identical results (since voxels outside the bbox are zero).
    """
    if mask.ndim == 3:
        # Project along each axis: O(N) bool scans, only small 1D outputs.
        # Avoid np.argwhere/flatnonzero which allocate int64 arrays sized to the
        # number of nonzero voxels — pathological for dense masks.
        nz_any = mask != 0
        ax0_where = np.where(np.any(nz_any, axis=(1, 2)))[0]
        if ax0_where.size == 0:
            return None
        ax1_where = np.where(np.any(nz_any, axis=(0, 2)))[0]
        ax2_where = np.where(np.any(nz_any, axis=(0, 1)))[0]
        i0, i1 = int(ax0_where[0]), int(ax0_where[-1])
        j0, j1 = int(ax1_where[0]), int(ax1_where[-1])
        k0, k1 = int(ax2_where[0]), int(ax2_where[-1])
        return (slice(i0, i1 + 1), slice(j0, j1 + 1), slice(k0, k1 + 1))
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mn = coords.min(axis=0)
    mx = coords.max(axis=0) + 1
    return tuple(slice(int(mn[d]), int(mx[d])) for d in range(mask.ndim))


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    bbox = _nonzero_bbox(mask)
    if bbox is None:
        return mask
    sub = mask[bbox]
    labeled, num_features = label(sub)
    if num_features <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())[1:]  # skip background
    largest = sizes.argmax() + 1
    cleaned_sub = (labeled == largest).astype(mask.dtype)
    out = np.zeros_like(mask)
    out[bbox] = cleaned_sub
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    bbox = _nonzero_bbox(mask)
    if bbox is None:
        return binary_fill_holes(mask).astype(mask.dtype)
    sub = mask[bbox]
    # Pad with 1 zero on all sides so scipy's "boundary = outside" semantics
    # match the full-volume case (where surrounding zeros connect to outside).
    padded = np.pad(sub, 1, mode="constant", constant_values=0)
    filled = binary_fill_holes(padded)
    inner = filled[tuple(slice(1, -1) for _ in range(mask.ndim))]
    out = np.zeros_like(mask)
    out[bbox] = inner.astype(mask.dtype)
    return out


# currently the remaining are not used, but can be plugged in for specific cases

def remove_small_components(mask: np.ndarray, min_volume_voxels: int = 100) -> np.ndarray:
    labeled, num_features = label(mask)
    sizes = np.bincount(labeled.ravel())
    small_labels = np.where(sizes < min_volume_voxels)[0]
    mask_clean = mask.copy()
    for sl in small_labels:
        mask_clean[labeled == sl] = 0
    return mask_clean


def morphological_close(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    struct = generate_binary_structure(3, 1)  # 6-connectivity
    # Expand structuring element for larger radius
    struct = iterate_structure(struct, radius)
    return binary_closing(mask, structure=struct).astype(mask.dtype)


def morphological_open(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    struct = iterate_structure(generate_binary_structure(3, 1), radius)
    return binary_opening(mask, structure=struct).astype(mask.dtype)


def _fallback_convex_hull_2d(slice_mask: np.ndarray) -> np.ndarray:
    pts = np.column_stack(np.nonzero(slice_mask))
    if pts.shape[0] < 3:
        return slice_mask

    try:
        hull = ConvexHull(pts)
    except Exception:
        return slice_mask

    hull_pts = pts[hull.vertices]
    try:
        from skimage.draw import polygon
    except ImportError:
        # Fallback to bounding box if polygon filling is unavailable
        min_r, max_r = hull_pts[:, 0].min(), hull_pts[:, 0].max()
        min_c, max_c = hull_pts[:, 1].min(), hull_pts[:, 1].max()
        filled = np.zeros_like(slice_mask, dtype=bool)
        filled[min_r : max_r + 1, min_c : max_c + 1] = True
        return filled

    rr, cc = polygon(hull_pts[:, 0], hull_pts[:, 1], shape=slice_mask.shape)
    filled = np.zeros_like(slice_mask, dtype=bool)
    filled[rr, cc] = True
    return filled


def apply_convex_hull_2d_slicewise(mask: np.ndarray) -> np.ndarray:
    """Apply convex hull slice-by-slice (3D convex hull is too aggressive)."""
    result = np.zeros_like(mask)
    for z in range(mask.shape[2]):
        sl = mask[:, :, z]
        if sl.sum() > 0:
            if convex_hull_image is not None:
                result[:, :, z] = convex_hull_image(sl)
            else:
                warnings.warn(
                    "scikit-image convex_hull_image unavailable; using fallback bounding hull",
                    stacklevel=2,
                )
                result[:, :, z] = _fallback_convex_hull_2d(sl)
    return result


def threshold_probabilities(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (prob_map >= threshold).astype(np.uint8)


def staple_fusion(masks: list) -> np.ndarray:
    """
    STAPLE consensus from multiple binary masks.

    Tries backends in order: SimpleITK (preferred, already in cdw_radiomics),
    ITK (heavier), probabilistic mean (pure numpy fallback).
    """
    if len(masks) == 0:
        raise ValueError("At least one mask is required for STAPLE fusion")
    if len(masks) == 1:
        return masks[0].astype(np.uint8)

    # Try SimpleITK first (lighter, already installed in cdw_radiomics)
    # STAPLE requires uint16 input for 3D volumes (float32 not supported in 3D)
    try:
        import SimpleITK as sitk
        sitk_masks = [sitk.GetImageFromArray(m.astype(np.uint16)) for m in masks]
        result = sitk.STAPLE(sitk_masks)
        return (sitk.GetArrayFromImage(result) > 0.5).astype(np.uint8)
    except Exception:
        pass

    # Try ITK (heavier but more configurable)
    try:
        if itk is not None:
            itk_masks = [itk.GetImageFromArray(m.astype(np.float32)) for m in masks]
            ImageType = type(itk_masks[0])
            staple_filter = itk.STAPLEImageFilter[ImageType, ImageType].New()
            for idx, itk_mask in enumerate(itk_masks):
                staple_filter.SetInput(idx, itk_mask)
            staple_filter.Update()
            result = staple_filter.GetOutput()
            return (itk.GetArrayFromImage(result) > 0.5).astype(np.uint8)
    except Exception:
        pass

    # Fallback: probabilistic mean (equivalent to majority vote for binary)
    warnings.warn(
        "Neither SimpleITK nor ITK STAPLE available; using probabilistic mean",
        stacklevel=2,
    )
    stacked = np.stack(masks, axis=0)
    return (stacked.mean(axis=0) > 0.5).astype(np.uint8)


def majority_vote(masks: list) -> np.ndarray:
    stacked = np.stack(masks, axis=0)
    return (stacked.sum(axis=0) > len(masks) / 2).astype(np.uint8)


def smooth_boundary(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    smoothed = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return (smoothed > 0.5).astype(mask.dtype)
