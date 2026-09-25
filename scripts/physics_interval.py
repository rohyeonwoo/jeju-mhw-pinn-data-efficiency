"""Endpoint/period-mean heat-budget loss; the original src is read-only."""
import train_hybrid_jeju_forecast as hybrid


def interval_physics_loss(pred_raw, Tm, target_raw, mask, batch, args,
                          lead_days, we_pred, rres_pred):
    if not args.zero_rres_physics:
        raise ValueError('This experiment excludes the free Rres closure')
    steps = max(int(lead_days), 1)
    rhs, _ = hybrid._linear_rhs_per_day(
        pred_raw, Tm, mask, batch, args, steps, we_pred)
    rhs_mean = rhs.reshape(steps, *pred_raw.shape).mean(dim=0)
    residual = (pred_raw - Tm) / steps - rhs_mean
    # Mean the raw signed residual first, then apply the legacy clip/scale.
    return hybrid.compute_physics_loss(
        residual, mask, args,
        tendency_per_day=hybrid.observed_tendency_per_day(Tm, target_raw, steps),
        rhs_per_day=rhs_mean)
