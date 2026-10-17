"""Ocean-neighbor stencil reused from the validated 2026-09 physics pilot."""
import torch
import torch.nn.functional as F

STENCIL = 'ocean_neighbors_v1'


def shifted_neighbors(field):
    return (F.pad(field[..., :-1], (1, 0, 0, 0)),
            F.pad(field[..., 1:], (0, 1, 0, 0)),
            F.pad(field[..., :-1, :], (0, 0, 1, 0)),
            F.pad(field[..., 1:, :], (0, 0, 0, 1)))


def ocean_neighbor_support(mask):
    """Physics cells need at least one ocean neighbor on EACH horizontal axis."""
    left, right, up, down = shifted_neighbors(mask > 0)
    return ((mask > 0) & (left | right) & (up | down)).to(mask.dtype)


def ocean_neighbor_gradients(field, dx, dy, mask):
    """Central / one-sided / zero if two / one / no ocean neighbors per axis.

    Land is zeroed before arithmetic, so even NaN/Inf land fill cannot enter a
    valid derivative or its backward graph. Finite ocean arithmetic matches
    the pilot exactly. Cells missing either axis are excluded by support().
    """
    ocean = mask.to(field.device) > 0
    field = torch.where(ocean, field, torch.zeros_like(field))
    ml, mr, mu, md = shifted_neighbors(ocean)
    left, right, up, down = shifted_neighbors(field)

    def derivative(lo, hi, has_lo, has_hi, spacing):
        numerator = torch.where(has_lo & has_hi, (hi - lo) * .5,
                    torch.where(has_hi, hi - field,
                    torch.where(has_lo, field - lo, torch.zeros_like(field))))
        safe = torch.where(spacing.abs() > 1e-12, spacing, torch.ones_like(spacing))
        return torch.where(ocean, numerator / safe, torch.zeros_like(field))

    return derivative(left, right, ml, mr, dx), derivative(up, down, mu, md, dy)
