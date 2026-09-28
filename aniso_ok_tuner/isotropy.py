"""Iso vs aniso selection: force override or auto heuristics."""

from __future__ import annotations

from typing import Any


def resolve_isotropy_mode(cfg: dict[str, Any]) -> str:
    """Return ``auto`` | ``force_iso`` | ``force_aniso``.

    Preference:
    1. ``isotropy_mode`` if set
    2. legacy ``isotropic: true/false`` → force_iso / force_aniso
    3. default ``auto``
    """
    raw = cfg.get("isotropy_mode", None)
    if raw is not None:
        mode = str(raw).strip().lower()
        aliases = {
            "auto": "auto",
            "iso": "force_iso",
            "force_iso": "force_iso",
            "isotropic": "force_iso",
            "aniso": "force_aniso",
            "force_aniso": "force_aniso",
            "anisotropic": "force_aniso",
        }
        if mode not in aliases:
            raise ValueError(
                f"isotropy_mode={raw!r}; expected one of {sorted(set(aliases))}"
            )
        return aliases[mode]

    if "isotropic" in cfg:
        return "force_iso" if bool(cfg["isotropic"]) else "force_aniso"
    return "auto"


def select_isotropy(
    *,
    a_major: float,
    a_minor: float,
    nugget: float,
    sill_total: float,
    n_points: int = 10**9,
    n_lags_major: int | None = None,
    n_lags_minor: int | None = None,
    sill_major: float | None = None,
    sill_minor: float | None = None,
    nugget_major: float | None = None,
    nugget_minor: float | None = None,
    anisotropy_ratio_min: float = 1.3,
    nugget_frac_iso: float = 0.45,
    aniso_min_points: int = 80,
    min_directional_lags: int = 4,
    sill_ratio_max: float = 2.5,
) -> tuple[bool, str, dict[str, float]]:
    """Decide isotropic (True) vs anisotropic (False).

    Rules (first match wins):
    1. Too few samples: ``n_points < aniso_min_points`` → ISO
    2. Hard noise: max(omni, directional) nugget/sill ≥ ``nugget_frac_iso`` → ISO
    3. Sparse directional VG: major/minor lag bins < ``min_directional_lags`` → ISO
    4. Inconsistent directional sills: sill ratio > ``sill_ratio_max`` → ISO
    5. Weak anisotropy: ``a_major / a_minor < anisotropy_ratio_min`` → ISO
    6. Else → aniso
    """
    a_maj = float(max(a_major, 1e-12))
    a_min = float(max(a_minor, 1e-12))
    if a_maj < a_min:
        a_maj, a_min = a_min, a_maj
    a_ratio = a_maj / a_min
    sill = float(max(sill_total, 1e-12))
    nugget_omni = float(max(nugget, 0.0))
    nugget_frac = nugget_omni / sill

    # directional nugget fractions (noise along axes)
    nugget_fracs = [nugget_frac]
    for ng, sil in (
        (nugget_major, sill_major),
        (nugget_minor, sill_minor),
    ):
        if ng is None or sil is None:
            continue
        tot = float(max(ng, 0.0)) + float(max(sil, 0.0))
        if tot > 1e-12:
            nugget_fracs.append(float(max(ng, 0.0)) / tot)
    nugget_frac_used = float(max(nugget_fracs))

    n_lags_maj = int(n_lags_major) if n_lags_major is not None else 10**9
    n_lags_min = int(n_lags_minor) if n_lags_minor is not None else 10**9

    sill_ratio = 1.0
    if sill_major is not None and sill_minor is not None:
        s_hi = float(max(sill_major, sill_minor, 1e-12))
        s_lo = float(max(min(sill_major, sill_minor), 1e-12))
        sill_ratio = s_hi / s_lo

    metrics = {
        "n_points": float(n_points),
        "a_ratio": float(a_ratio),
        "nugget_frac": float(nugget_frac),
        "nugget_frac_used": float(nugget_frac_used),
        "n_lags_major": float(n_lags_maj if n_lags_major is not None else -1),
        "n_lags_minor": float(n_lags_min if n_lags_minor is not None else -1),
        "sill_ratio": float(sill_ratio),
        "anisotropy_ratio_min": float(anisotropy_ratio_min),
        "nugget_frac_iso": float(nugget_frac_iso),
        "aniso_min_points": float(aniso_min_points),
        "min_directional_lags": float(min_directional_lags),
        "sill_ratio_max": float(sill_ratio_max),
        "a_major": float(a_maj),
        "a_minor": float(a_min),
        "nugget": float(nugget_omni),
        "sill_total": float(sill),
    }

    if int(n_points) < int(aniso_min_points):
        return True, "few_points", metrics
    if nugget_frac_used >= float(nugget_frac_iso):
        return True, "high_nugget_frac", metrics
    if n_lags_major is not None and n_lags_maj < int(min_directional_lags):
        return True, "sparse_directional", metrics
    if n_lags_minor is not None and n_lags_min < int(min_directional_lags):
        return True, "sparse_directional", metrics
    if (
        sill_major is not None
        and sill_minor is not None
        and sill_ratio > float(sill_ratio_max)
    ):
        return True, "inconsistent_directional_sills", metrics
    if a_ratio < float(anisotropy_ratio_min):
        return True, "weak_anisotropy", metrics
    return False, "clear_anisotropy", metrics


def explain_isotropy_selection(
    *,
    isotropic: bool,
    reason: str,
    metrics: dict[str, float] | None = None,
    mode: str = "auto",
) -> str:
    """One short RU sentence for reports (why ISO/ANISO)."""
    m = metrics or {}
    n = int(m["n_points"]) if "n_points" in m else None
    a_ratio = m.get("a_ratio")
    nug = m.get("nugget_frac_used", m.get("nugget_frac"))
    lags_maj = m.get("n_lags_major")
    lags_min = m.get("n_lags_minor")
    sill_r = m.get("sill_ratio")
    n_min = m.get("aniso_min_points")
    nug_thr = m.get("nugget_frac_iso")
    lag_thr = m.get("min_directional_lags")
    sill_thr = m.get("sill_ratio_max")
    a_thr = m.get("anisotropy_ratio_min")

    if mode == "force_iso":
        return "ISO: принудительно (force_iso)."
    if mode == "force_aniso":
        return "ANISO: принудительно (force_aniso)."

    if reason == "few_points":
        return (
            f"ISO: мало точек (n={n} < {int(n_min) if n_min else '?'}), "
            f"directional MOI ненадёжен."
        )
    if reason == "high_nugget_frac":
        return (
            f"ISO: сильный шум (nugget/sill={nug:.2f} ≥ "
            f"{nug_thr:.2f})."
            if nug is not None and nug_thr is not None
            else "ISO: сильный шум (высокий nugget/sill)."
        )
    if reason == "sparse_directional":
        lj = (
            f"лаги major/minor={int(lags_maj)}/{int(lags_min)}"
            if lags_maj is not None and lags_min is not None and lags_maj >= 0
            else "мало лагов на осях"
        )
        thr = int(lag_thr) if lag_thr is not None else "?"
        return f"ISO: дырявая directional VG ({lj}, порог ≥{thr})."
    if reason == "inconsistent_directional_sills":
        return (
            f"ISO: sill major/minor несогласованы "
            f"(ratio={sill_r:.2f} > {sill_thr})."
            if sill_r is not None and sill_thr is not None
            else "ISO: несогласованные sill по осям."
        )
    if reason == "weak_anisotropy":
        return (
            f"ISO: слабая вытянутость (a_ratio={a_ratio:.2f} < {a_thr})."
            if a_ratio is not None and a_thr is not None
            else "ISO: слабая вытянутость."
        )
    if reason == "clear_anisotropy":
        return (
            f"ANISO: явная анизотропия (a_ratio={a_ratio:.2f}), "
            f"n={n}, шум/лаги ок."
            if a_ratio is not None and n is not None
            else "ANISO: явная анизотропия."
        )
    geom = "ISO" if isotropic else "ANISO"
    return f"{geom}: {reason}."
