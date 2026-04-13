"""
generation.py  --  Realistic distributed energy resource (DER) profiles.

Three generation sources modelled with physically-grounded temporal dynamics:

1. **SolarProfile** -- PV array with diurnal envelope + Markov cloud model
2. **WindProfile**  -- Wind turbine with Ornstein-Uhlenbeck wind speed process
3. **SMRProfile**   -- Small Modular Reactor (baseload) with thermal inertia

Each class exposes  `get_power_kw(t_s) -> float`  for use by the simulator's
per-second energy balance step.

Literature basis:
  - Solar clearness-index Markov model: Aguiar & Collares-Pereira (1992)
  - Wind O-U process: Bianchi, De Battista & Mantz, "Wind Turbine Control
    Systems" (Springer, 2007)
  - SMR capacity factors: IAEA TECDOC-1936 (2021), NuScale Power Module specs
  - Weibull wind speed: IEC 61400-1 wind turbine classes
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
#  Solar PV Profile
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SolarProfile:
    """
    Photovoltaic generation with diurnal curve + stochastic cloud cover.

    Parameters
    ----------
    capacity_kw : float
        Nameplate peak capacity under Standard Test Conditions (STC).
    time_of_day_h : float
        Clock hour at simulation t=0 (e.g. 10.0 = 10 AM start).
    latitude_deg : float
        Site latitude (governs solar elevation envelope).
    cloud_transition_s : float
        Mean sojourn time in each cloud state (Markov chain).
    clear_cf_range : tuple
        (min, max) capacity factor in the "clear" state.
    cloudy_cf_range : tuple
        (min, max) capacity factor in the "cloudy" state.
    ramp_s : float
        Smoothing time constant for cloud transitions (seconds).
    """
    capacity_kw: float = 60.0
    time_of_day_h: float = 10.0      # sim starts at 10 AM — 4h sim captures solar peak + afternoon decline
    latitude_deg: float = 35.0       # mid-latitude site
    cloud_transition_s: float = 300.0 # ~5 min mean between changes
    clear_cf_range: Tuple[float, float] = (0.80, 1.00)
    cloudy_cf_range: Tuple[float, float] = (0.20, 0.50)
    ramp_s: float = 60.0             # 1-min smoothing for cloud transitions

    # Internal state
    _rng: random.Random = field(default=None, repr=False)
    _is_clear: bool = field(default=True, repr=False)
    _next_transition_s: float = field(default=0.0, repr=False)
    _target_cf: float = field(default=0.9, repr=False)
    _smooth_cf: float = field(default=0.9, repr=False)

    def __post_init__(self):
        if self._rng is None:
            self._rng = random.Random(42)
        self._is_clear = True
        self._target_cf = self._rng.uniform(*self.clear_cf_range)
        self._smooth_cf = self._target_cf
        self._next_transition_s = self._rng.expovariate(1.0 / self.cloud_transition_s)

    def seed(self, rng: random.Random):
        """Attach an external RNG for reproducibility."""
        self._rng = rng
        self._next_transition_s = self._rng.expovariate(1.0 / self.cloud_transition_s)

    def _solar_elevation_factor(self, t_s: int) -> float:
        """
        Simplified diurnal envelope based on solar hour angle.
        Returns 0 at night, peaks ~1.0 at solar noon.
        """
        hour = self.time_of_day_h + t_s / 3600.0
        # Solar hour angle relative to noon (hours from solar noon)
        hour_angle = (hour % 24.0) - 12.0
        # Approximate: cos-based daylight curve, zero below horizon
        # At latitude 35 deg, day length ~10-14 h; use ±7h as daylight window
        lat_rad = math.radians(self.latitude_deg)
        # Simplified declination (assume equinox for simplicity → decl ≈ 0)
        cos_zenith = math.sin(lat_rad) * 0.0 + math.cos(lat_rad) * math.cos(
            math.radians(hour_angle * 15.0)
        )
        # Clamp: sun below horizon → 0
        return max(0.0, cos_zenith)

    def _update_cloud_state(self, t_s: int) -> None:
        """Markov-chain cloud state transitions."""
        if t_s >= self._next_transition_s:
            self._is_clear = not self._is_clear
            if self._is_clear:
                self._target_cf = self._rng.uniform(*self.clear_cf_range)
            else:
                self._target_cf = self._rng.uniform(*self.cloudy_cf_range)
            self._next_transition_s = t_s + self._rng.expovariate(
                1.0 / self.cloud_transition_s
            )

    def get_power_kw(self, t_s: int) -> float:
        """Return solar generation (kW) at simulation second t_s."""
        self._update_cloud_state(t_s)

        # Exponential smoothing for cloud transitions
        alpha = min(1.0, 1.0 / self.ramp_s) if self.ramp_s > 0 else 1.0
        self._smooth_cf += alpha * (self._target_cf - self._smooth_cf)

        elevation = self._solar_elevation_factor(t_s)
        return max(0.0, self.capacity_kw * self._smooth_cf * elevation)


# ─────────────────────────────────────────────────────────────────────
#  Wind Turbine Profile
# ─────────────────────────────────────────────────────────────────────

@dataclass
class WindProfile:
    """
    Wind turbine generation using Ornstein-Uhlenbeck wind speed process.

    The O-U process gives mean-reverting, temporally-correlated wind speeds,
    unlike white-noise Gaussian sampling.  A standard IEC power curve converts
    wind speed to electrical output.

    Parameters
    ----------
    capacity_kw : float
        Rated electrical output at rated wind speed.
    mean_speed_ms : float
        Long-term mean wind speed (Weibull scale ≈ mean for k≈2).
    turbulence_intensity : float
        σ_v / mean_v — standard IEC definition.
    correlation_s : float
        O-U mean-reversion timescale τ (seconds).
    cut_in_ms : float
        Cut-in wind speed (m/s).
    rated_ms : float
        Rated wind speed — output reaches capacity_kw.
    cut_out_ms : float
        Cut-out (safety shutdown) wind speed.
    """
    capacity_kw: float = 40.0
    mean_speed_ms: float = 7.0
    turbulence_intensity: float = 0.15
    correlation_s: float = 30.0       # 30-s mean reversion
    cut_in_ms: float = 3.0
    rated_ms: float = 12.0
    cut_out_ms: float = 25.0

    # Internal state
    _rng: random.Random = field(default=None, repr=False)
    _v: float = field(default=0.0, repr=False)  # current wind speed

    def __post_init__(self):
        if self._rng is None:
            self._rng = random.Random(43)
        self._v = self.mean_speed_ms

    def seed(self, rng: random.Random):
        self._rng = rng
        self._v = self.mean_speed_ms + self._rng.gauss(0, self.mean_speed_ms * self.turbulence_intensity)

    def _step_ou(self, dt_s: float = 1.0) -> float:
        """
        Advance Ornstein-Uhlenbeck process by dt_s seconds.
        dv = θ(μ - v)dt + σ dW
        """
        theta = 1.0 / max(1.0, self.correlation_s)
        sigma = self.mean_speed_ms * self.turbulence_intensity * math.sqrt(2.0 * theta)
        drift = theta * (self.mean_speed_ms - self._v) * dt_s
        diffusion = sigma * math.sqrt(dt_s) * self._rng.gauss(0, 1)
        self._v = max(0.0, self._v + drift + diffusion)
        return self._v

    def _power_curve(self, v: float) -> float:
        """IEC-style cubic power curve with cut-in/rated/cut-out."""
        if v < self.cut_in_ms or v >= self.cut_out_ms:
            return 0.0
        if v >= self.rated_ms:
            return self.capacity_kw
        # Cubic interpolation between cut-in and rated
        frac = (v - self.cut_in_ms) / (self.rated_ms - self.cut_in_ms)
        return self.capacity_kw * (frac ** 3)

    def get_power_kw(self, t_s: int) -> float:
        """Return wind generation (kW) at simulation second t_s."""
        v = self._step_ou(dt_s=1.0)
        return self._power_curve(v)


# ─────────────────────────────────────────────────────────────────────
#  Small Modular Reactor (SMR) Profile
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SMRProfile:
    """
    Small Modular Reactor providing near-constant baseload.

    SMRs (e.g. NuScale VOYGR, GE-Hitachi BWRX-300) have high capacity factors
    (85-95%) and very slow thermal ramp rates.  Over a 40-minute simulation
    window, output is effectively constant with minor instrumentation noise.

    Parameters
    ----------
    capacity_kw : float
        Rated thermal-to-electric output.
    availability : float
        Long-term availability factor (0-1).
    noise_sigma_frac : float
        Fractional noise σ (instrument/measurement jitter).
    ramp_rate_frac_per_min : float
        Maximum output change per minute (fraction of rated).
    """
    capacity_kw: float = 50.0
    availability: float = 0.90
    noise_sigma_frac: float = 0.015   # 1.5% jitter
    ramp_rate_frac_per_min: float = 0.01  # 1% per minute max ramp

    # Internal state
    _rng: random.Random = field(default=None, repr=False)
    _current_output_frac: float = field(default=0.0, repr=False)

    def __post_init__(self):
        if self._rng is None:
            self._rng = random.Random(44)
        self._current_output_frac = self.availability

    def seed(self, rng: random.Random):
        self._rng = rng
        self._current_output_frac = self.availability

    def get_power_kw(self, t_s: int) -> float:
        """Return SMR generation (kW) at simulation second t_s."""
        # Very small noise around the availability-set dispatch point
        noise = self._rng.gauss(0, self.noise_sigma_frac)
        output_frac = self._current_output_frac + noise
        output_frac = max(0.0, min(1.0, output_frac))
        return self.capacity_kw * output_frac


# ─────────────────────────────────────────────────────────────────────
#  Combined Generation Mix
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GenerationMix:
    """
    Aggregate generation from solar + wind + SMR sources.
    Provides a single get_power(t_s) that returns per-source + total.
    """
    solar: SolarProfile = field(default_factory=SolarProfile)
    wind: WindProfile = field(default_factory=WindProfile)
    smr: SMRProfile = field(default_factory=SMRProfile)

    def seed(self, seed: int, node_idx: int = 0):
        """
        Seed all sub-profiles with deterministic but distinct RNGs.
        node_idx offsets ensure different nodes get different weather.
        """
        self.solar.seed(random.Random(seed + node_idx * 100 + 1))
        self.wind.seed(random.Random(seed + node_idx * 100 + 2))
        self.smr.seed(random.Random(seed + node_idx * 100 + 3))

    def get_power(self, t_s: int) -> Tuple[float, float, float]:
        """
        Returns (solar_kw, wind_kw, smr_kw) at simulation second t_s.
        """
        s = self.solar.get_power_kw(t_s)
        w = self.wind.get_power_kw(t_s)
        m = self.smr.get_power_kw(t_s)
        return (s, w, m)

    def get_total_kw(self, t_s: int) -> float:
        s, w, m = self.get_power(t_s)
        return s + w + m
