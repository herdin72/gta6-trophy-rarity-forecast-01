"""gta6-trophy-rarity-forecast-01: a forecaster for GTA VI trophy earn rates.

The task: for every trophy of Grand Theft Auto VI on PlayStation 5, forecast the
percent of Exophase-tracked players who will have earned it at the case's target
(2026-11-26T00:00:00Z for the final case, 6,510 minutes after the deadline), as
five quantiles (5%, 25%, 50%, 75%, 95%). Scored by mean pinball loss on the logit
scale; the entry must beat the house baselines to take a prize.

The model -- earners over players, separately.
---------------------------------------------
A trophy's earn rate is a ratio, and in launch week its two halves move very
differently:

    rate_i(t) = 100 * E_i(t) / P(t)

  - P(t), the players Exophase tracks, EXPLODES after launch. The house's own
    recording shows 638 players on launch night, then 3,868 at the first
    refresh (2026-11-19 04:30 UTC); the public page on 2026-11-20 showed the
    same pattern continuing (most rates far below launch night, because the
    denominator had grown many times over while earner counts had barely
    moved). New players dilute almost every rate.
  - E_i(t), the players who have earned trophy i, only ever grows, and grows
    at a trophy-specific pace: early story trophies (Chapter 1, fishing) pick
    up earners fast as the new cohort plays; late trophies (Chapter 4, the
    final mission, 100% completion, platinum) stay near zero for a week.

So this program fits the two counts separately, each as a power law in
log-time, x = ln(1 + hours since the first read):

    ln P(t)        = c  + d * x          (d >= 0: tracked players never shrink)
    ln (E_i(t)+.5) = a_i + b_i * x       (b_i >= 0: earners never shrink)

and forecasts rate_i(target) = 100 * E_i(target) / P(target). The median can
therefore FALL for a trophy whose earners grow slower than the player base
(dilution -- the fate of most trophies in week one) and RISE for early
trophies whose earners outrun it. A persistence forecast ("rates stay") gets
both directions wrong over a 4.5-day horizon, which is the whole edge over
the house baseline.

Fitting details:
  - Reads are deduplicated first: Exophase only changes at its own refreshes,
    so consecutive identical reads carry no information and would overweight
    the flat stretches. Only reads where the player count or some rate moved
    enter the fits.
  - Least squares with a mild recency weight (recent reads count double the
    oldest), because the launch-week regime changes fast.
  - Slopes are clamped (P: 0..4, E: 0..6) and the forecasts capped (P at most
    8x its last value, E at most 50x) so a steep two-point fit cannot run away
    over a 4.5-day extrapolation. A trophy at 100% in the last read stays
    100%; a trophy nobody has earned stays near zero.
  - When the history shows no movement at all (a single distinct read -- the
    practice cases are perfectly flat), the fits carry no signal, so the
    player-count slope falls back to a launch-week prior (1.5 in log-time,
    from the house's own first refresh, 638 -> 3,868 players, and the public
    page the day after launch), ramped in over a 12-hour horizon because
    player growth arrives in refresh jumps hours apart, not continuously.
    The fallback only fires on an uninformative history; on the final case,
    with two days of recorded growth, the fit runs the show.

The spread (logit scale) is the honest uncertainty, combined in quadrature:
  - a random-walk floor, 0.5 * sqrt(horizon in days), as the house baseline;
  - a launch-week volatility prior, 0.75 * sqrt(horizon in days): refreshes
    and dilution move launch-week rates by logits, not fractions of one,
    and a short or flat history cannot show that;
  - the fits' own residuals, scaled up with the extrapolation distance
    beyond the last read;
  - a refresh-jump term, a quarter of the largest single-read jump in
    ln P seen in the history (a refresh can dilute every rate at once).

Quantiles are the median shifted by the normal z-scores
(-1.645, -0.674, 0, 0.674, 1.645) times that spread, on the logit scale,
rounded half-up to 4 decimals, exactly like the house baseline's numbers.

Standard library only, no network, no randomness: two runs on the same case
write the same bytes. Runs in milliseconds on a case.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread of a rate's move, in logit units per square-root day (house baseline).
RW_SIGMA = 0.5
# Launch-week volatility prior, same units: refreshes and dilution move
# launch-week rates by whole logits, which a flat history cannot reveal.
VOL_SIGMA = 0.75
# Player-count slope fallback (log-time) when the history never moved:
# launch-week tracked players grow roughly like (1 + t)^1.5 from a few hours
# after launch (the house's own first refresh, 638 -> 3,868 players, and
# the public page a day after launch, ~15,000 players, both imply a
# log-time slope of about 1.5 over the segment a few hours past launch).
P_PRIOR_SLOPE = 1.5
# Caps so a steep fit cannot run away over a 4.5-day extrapolation.
P_GROWTH_CAP = 8.0      # fitted player count, multiple of the last read
P_PRIOR_CAP = 25.0      # fallback player count, multiple of the last read
E_GROWTH_CAP = 50.0     # earners, multiple of the last read's earners
# Slope clamps for the two log-time fits.
P_SLOPE_MAX = 4.0
E_SLOPE_MAX = 6.0
# Smoothing added to earner counts before the log (rates are printed to 2
# decimals, so small counts are quantised; +0.5 also keeps ln defined at 0).
E_SMOOTH = 0.5


def clamp_rate(x):
    """A rate into the 0.01..99.99 window the logit needs."""
    return min(99.99, max(0.01, x))


def logit_of(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic_of(logit_value):
    return 100.0 / (1.0 + math.exp(-logit_value))


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(read_at, first_at):
    """Hours since the first read, on the log scale the fits use."""
    hours = (read_at - first_at).total_seconds() / 3600.0
    return math.log(1.0 + max(0.0, hours))


def distinct_reads(history):
    """Drop consecutive reads that changed nothing (Exophase only moves at
    its own refreshes). Keeps the first read of each distinct page state."""
    kept = []
    last_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != last_key:
            kept.append(read)
            last_key = key
    return kept


def weighted_line(xs, ys, ws, slope_lo=None, slope_hi=None):
    """Weighted least squares for y = intercept + slope * x.

    Returns (intercept, slope, rms_residual). With fewer than two points, or
    no spread in x, the slope is 0 and the intercept is the weighted mean.
    When slope bounds are given, the slope is clamped to them and the
    intercept is refit at the clamped slope (clamping the slope without
    refitting would shift the level of the line).
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    w_sum = sum(ws)
    x_mean = sum(w * x for w, x in zip(ws, xs)) / w_sum
    y_mean = sum(w * y for w, y in zip(ws, ys)) / w_sum
    sxx = sum(w * (x - x_mean) ** 2 for w, x in zip(ws, xs))
    if n < 2 or sxx <= 1e-12:
        resid = math.sqrt(sum(w * (y - y_mean) ** 2 for w, y in zip(ws, ys)) / w_sum)
        return y_mean, 0.0, resid
    sxy = sum(w * (x - x_mean) * (y - y_mean) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    if slope_lo is not None:
        slope = max(slope, slope_lo)
    if slope_hi is not None:
        slope = min(slope, slope_hi)
    intercept = sum(w * (y - slope * x) for w, x, y in zip(ws, xs, ys)) / w_sum
    resid = math.sqrt(
        sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / w_sum
    )
    return intercept, slope, resid


def recency_weights(times):
    """Mild recency weight: the newest read counts double the oldest."""
    if len(times) < 2:
        return [1.0] * len(times)
    lo, hi = min(times), max(times)
    span = hi - lo
    if span <= 1e-12:
        return [1.0] * len(times)
    return [0.5 + 0.5 * (t - lo) / span for t in times]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    trophies = case["trophies"]
    reads = distinct_reads(case["history"])
    first_at = parse_time(reads[0]["at"])
    # Anchor on the CHRONOLOGICAL last read (the most recent page state),
    # not on the last distinct read: the last few reads are often duplicates
    # of an earlier state, and anchoring on the first of them would pretend
    # the forecast starts hours earlier than it does.
    last = case["history"][-1]
    last_at = parse_time(last["at"])
    target_at = parse_time(case["target"])
    x_target = log_time(target_at, first_at)
    x_last = log_time(last_at, first_at)
    horizon_days = case["horizon_minutes"] / 1440.0

    xs = [log_time(parse_time(r["at"]), first_at) for r in reads]
    ws = recency_weights(xs)

    # --- player-count fit: ln P against log-time -------------------------
    p_last = last["players_tracked"]
    p_ys = [math.log(max(1, r["players_tracked"])) for r in reads]
    p_int, p_slope, p_resid = weighted_line(xs, p_ys, ws, 0.0, P_SLOPE_MAX)
    p_informative = len(reads) >= 2 and len(set(p_ys)) > 1
    if p_informative:
        p_pred = math.exp(p_int + p_slope * x_target)
        p_pred = min(max(p_pred, p_last), p_last * P_GROWTH_CAP)
    else:
        # Flat history: fall back to the launch-week growth prior, anchored
        # on the last read so the forecast still starts from what was seen.
        # Player growth is lumpy (it jumps at Exophase refreshes, hours
        # apart), so the prior ramps in over 12 hours: over a 90-minute
        # horizon the page most likely does not refresh at all, while over
        # a day or more the onboarding surge is a near-certainty.
        ramp = min(1.0, case["horizon_minutes"] / 720.0)
        prior_slope = P_PRIOR_SLOPE * ramp
        p_pred = p_last * math.exp(prior_slope * (x_target - x_last))
        p_pred = min(p_pred, p_last * P_PRIOR_CAP)
        p_resid = 0.0

    # --- refresh-jump scale: the biggest single-read jump in ln P --------
    p_jumps = [
        abs(p_ys[k + 1] - p_ys[k]) for k in range(len(p_ys) - 1)
    ]
    jump_comp = 0.25 * max(p_jumps) if p_jumps else 0.0

    # --- spread parts shared by every trophy ------------------------------
    floor = RW_SIGMA * math.sqrt(horizon_days)
    vol_prior = VOL_SIGMA * math.sqrt(horizon_days)
    extrap = 1.0 + max(0.0, x_target - x_last)

    rows = []
    for i, trophy in enumerate(trophies):
        # Earner counts for this trophy, read by read (None where the read
        # lacked it). Earners are what the rate hides: rate * players / 100.
        earners = []
        for r in reads:
            rate = r["rates"][i]
            earners.append(None if rate is None else rate * r["players_tracked"] / 100.0)
        known = [(x, e) for x, e in zip(xs, earners) if e is not None]
        # Earners in the most recent read that lists this trophy.
        e_last = 0.0
        for r in reversed(case["history"]):
            rate = r["rates"][i]
            if rate is not None:
                e_last = rate * r["players_tracked"] / 100.0
                break

        # A trophy everyone already holds stays held.
        if e_last >= p_last and p_last > 0:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue

        # --- earner fit: ln(E + 0.5) against log-time --------------------
        if len(known) >= 2:
            kx = [x for x, _ in known]
            ky = [math.log(e + E_SMOOTH) for _, e in known]
            kw = recency_weights(kx)
            e_int, e_slope, e_resid = weighted_line(kx, ky, kw, 0.0, E_SLOPE_MAX)
        else:
            e_int = math.log(e_last + E_SMOOTH)
            e_slope, e_resid = 0.0, 0.0
        e_pred = math.exp(e_int + e_slope * x_target) - E_SMOOTH
        e_pred = min(max(e_pred, 0.0), E_GROWTH_CAP * max(e_last, E_SMOOTH))

        median = 100.0 * e_pred / p_pred if p_pred > 0 else 0.01
        median = clamp_rate(median)

        # --- spread: floor, volatility prior, fit residuals, refresh -----
        # A trophy nobody has ever earned is structurally stuck near zero:
        # halve its random-walk floor (its residual is zero already).
        ever_earned = any(e > 0 for e in earners if e is not None)
        mobility = 1.0 if ever_earned else 0.5
        resid = math.sqrt(e_resid ** 2 + p_resid ** 2) * extrap
        spread = math.sqrt(
            (mobility * floor) ** 2 + vol_prior ** 2 + resid ** 2 + jump_comp ** 2
        )
        centre = logit_of(median)
        quantiles = []
        for z in Z:
            q = logistic_of(centre + z * spread)
            quantiles.append(round4(min(99.99, max(0.01, q))))
        # Rounding is monotone, so the row stays non-decreasing; enforce it
        # explicitly all the same, in case a future edit breaks that.
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
