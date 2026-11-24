"""Paired forward simulations of hip support under impaired capacity or mobility."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MECHANISMS = [
    "unperturbed",
    "abductor capacity",
    "passive extension resistance",
    "combined",
]
ANGLES = ["hipFlexionDeg", "hipAdductionDeg", "trunkLeanDeg"]
ERROR_DEG = np.array([5.4, 3.7, 3.0])
OUTPUT_PHASE = np.linspace(0, 1, 65)


def population(n, rng):
    # Draw a virtual anatomy and controller once, then cross it with both perturbations.
    def uniform(lo, hi):
        return rng.uniform(lo, hi, n)

    return {
        "maxAbductionMomentBwM": uniform(0.10, 0.17),
        "abductorArmM": uniform(0.040, 0.065),
        "trunkLeverM": uniform(0.22, 0.35),
        "effortScaleRad": uniform(0.09, 0.22),
        "availableFraction": uniform(0.55, 0.85),
        "addedStiffnessBwMPerRad": uniform(0.18, 0.68),
        "passiveOnsetDeg": uniform(-12, -2),
        "trackingGain": uniform(0.72, 1.30),
        "flexOffsetDeg": rng.normal(0, 3.0, n),
        "flexAmplitude": uniform(0.88, 1.12),
        "adductionOffsetDeg": rng.normal(0, 2.0, n),
        "trunkOffsetDeg": rng.normal(0, 1.6, n),
        "externalScale": uniform(0.85, 1.15),
    }


def task_curve(reference, task, internal_steps, hand_load_bw):
    phase = np.linspace(0, 1, internal_steps + 1)
    if task == "walk":
        data = np.asarray(reference["frames"])
        template = np.column_stack(
            [np.interp(phase, data[:, 0], data[:, k]) for k in range(1, 7)]
        )
        return 0.68, template
    # Shift onto the right leg, hold, then unload an external contralateral moment.
    # The target is deliberately near neutral and loads the frontal-plane system.
    wave = np.sin(np.pi * phase) ** 2
    flexion = 7.0 - 5.0 * wave
    adduction = 1.5 + 3.5 * wave
    trunk = 0.8 * wave
    grf = 0.50 + (0.50 + hand_load_bw) * wave
    flexion_torque = 0.01 + 0.009 * wave
    # 0.10 BW external mass, contralateral horizontal moment arm 0.40 m.
    abduction_torque = 0.020 + (0.060 + hand_load_bw * 0.40) * wave
    return 2.20, np.column_stack(
        [flexion, adduction, trunk, grf, flexion_torque, abduction_torque]
    )


def simulate(
    reference,
    p,
    task,
    groups=(0, 1, 2, 3),
    steps=256,
    hand_load_bw=0.10,
    controller="effort",
    *,
    apply_trunk_load_relief=True,
):
    duration, template = task_curve(reference, task, steps, hand_load_bw)
    dt = duration / steps
    size = len(p["abductorArmM"])
    group = np.repeat(groups, size)
    par = {k: np.tile(v, len(groups)) for k, v in p.items()}
    capacity = par["maxAbductionMomentBwM"] * np.where(
        np.isin(group, [1, 3]), par["availableFraction"], 1
    )
    extra_k = np.where(np.isin(group, [2, 3]), par["addedStiffnessBwMPerRad"], 0)
    lever = par["trunkLeverM"]
    b2 = par["effortScaleRad"] ** 2
    active_k = 0.42 * par["trackingGain"]
    active_d = 0.105 * np.sqrt(par["trackingGain"])
    shape = (len(group), 3)
    displacement = np.zeros(shape)
    speed = np.zeros(shape)
    # Effective rotary inertia divided by body weight, in metres times seconds squared.
    inertia = np.array([0.009, 0.010, 0.016])
    store = np.empty((len(group), len(OUTPUT_PHASE), 5))
    indices = {i: k for k, i in enumerate(np.rint(OUTPUT_PHASE * steps).astype(int))}
    for t, row in enumerate(template):
        flex, add, lean, grf, flexion_torque, abduction_torque = row
        flex = np.deg2rad(flex * par["flexAmplitude"] + par["flexOffsetDeg"])
        add = np.deg2rad(add + par["adductionOffsetDeg"])
        lean = np.deg2rad(lean + par["trunkOffsetDeg"])
        angle = flex + displacement[:, 0]
        passive = extra_k * np.maximum(0, np.deg2rad(par["passiveOnsetDeg"]) - angle)
        demand = abduction_torque * par["externalScale"]
        support = lever * grf

        # Minimize normalized squared abductor effort plus squared lateral trunk deviation.
        # Subtract the nominal body's optimum, since its dynamics are already in the reference.
        def optimal_lean(strength):
            if controller == "reserve":
                # A second policy leans only to preserve a 10% abductor torque reserve.
                return np.minimum(
                    np.maximum(demand - 0.9 * strength, 0) / np.maximum(support, 0.01),
                    0.35,
                )
            return (
                demand * support * b2 / (strength * strength + support * support * b2)
            )

        lean_target = optimal_lean(capacity) - optimal_lean(
            par["maxAbductionMomentBwM"]
        )
        # The paired sensitivity test can decouple the achieved trunk lean from
        # frontal-plane hip demand while preserving the controller and movement target.
        ext = demand - support * np.sin(displacement[:, 2]) if apply_trunk_load_relief else demand
        # Deviations obey I*d2q/dt2 = applied - nominal torque. Positive frontal q
        # denotes adduction; ext is the external adduction moment. The joint spring
        # remains available when abductor recruitment hits its capacity.
        command_abd = ext + active_k * displacement[:, 1] + active_d * speed[:, 1]
        delivered_abd = np.clip(command_abd, -0.10, capacity)
        sagittal_command = (
            flexion_torque - active_k * displacement[:, 0] - active_d * speed[:, 0]
        )
        sagittal_actuator = np.clip(sagittal_command, -0.19, 0.16)
        net = np.column_stack(
            [
                sagittal_actuator - flexion_torque + passive,
                ext - delivered_abd - 0.19 * displacement[:, 1] - 0.055 * speed[:, 1],
                0.72 * (lean_target - displacement[:, 2]) - 0.17 * speed[:, 2],
            ]
        )
        if t in indices:
            store[:, indices[t], :] = np.column_stack(
                [
                    np.rad2deg(angle),
                    np.rad2deg(add + displacement[:, 1]),
                    np.rad2deg(lean + displacement[:, 2]),
                    np.maximum(delivered_abd, 0) / par["abductorArmM"],
                    passive,
                ]
            )
        if t < steps:
            speed += dt * net / inertia
            displacement += dt * speed
    if not np.isfinite(store).all():
        raise FloatingPointError("Nonfinite forward solution")
    return store.reshape(len(groups), size, len(OUTPUT_PHASE), 5), duration


def observed(truth, rng, bias=None):
    # Per-session calibration offset shared across tasks, and independent low-frequency error.
    group, n, t, c = truth.shape
    if bias is None:
        bias = rng.normal(size=(group, n, 1, c)) * ERROR_DEG * 0.72
    modes = 4
    phi = np.linspace(0, 2 * np.pi, t)
    bases = np.vstack(
        [wave(h * phi) for h in range(1, modes // 2 + 1) for wave in (np.sin, np.cos)]
    )
    jitter = np.einsum("gnmc,mt->gntc", rng.normal(size=(group, n, modes, c)), bases)
    jitter *= ERROR_DEG * np.sqrt((1 - 0.72**2) / (modes / 2))
    return truth + bias + jitter, bias


def score(views, train, test, excursion=False, testing_views=None):
    # Participants and their paired counterfactuals stay together in one split.
    if excursion:
        features = np.ptp(views[0][:, :, :, 0], axis=2)[..., None]
    else:
        features = np.concatenate(
            [(a[:, :, ::8, :] / ERROR_DEG).reshape(2, a.shape[1], -1) for a in views],
            axis=-1,
        )
    testing_features = features
    if testing_views is not None:
        testing_features = np.concatenate(
            [
                (a[:, :, ::8, :] / ERROR_DEG).reshape(2, a.shape[1], -1)
                for a in testing_views
            ],
            axis=-1,
        )
    training = features[:, train].reshape(-1, features.shape[-1])
    testing = testing_features[:, test].reshape(-1, features.shape[-1])
    labels = np.repeat([0, 1], len(train))
    ix = cKDTree(training).query(testing, k=15)[1]
    predictions = labels[ix].mean(axis=1) > 0.5
    outcomes = (predictions == np.repeat([0, 1], len(test))).reshape(2, -1)
    return outcomes


def make_frames(trace):
    return [
        {
            "phase": round(float(phase), 4),
            **{
                k: round(float(v), 4)
                for k, v in zip([*ANGLES, "abductorForceBw", "passiveTorqueBwM"], row)
            },
        }
        for phase, row in zip(OUTPUT_PHASE, trace)
    ]


def run(n=1024, seed=20261122):
    if n < 64:
        raise ValueError("At least 64 virtual anatomies are required for the holdout.")
    rng = np.random.default_rng(seed)
    reference = json.loads((HERE / "reference.json").read_text())
    p = population(n, rng)
    ids = rng.permutation(n)
    train, test = ids[: n * 3 // 4], ids[n * 3 // 4 :]
    walk, walk_time = simulate(reference, p, "walk")
    challenge, challenge_time = simulate(reference, p, "challenge")
    truth = [a[1:3, :, :, :3] for a in (walk, challenge)]
    observed_walk, bias = observed(truth[0], rng)
    observed_challenge, _ = observed(truth[1], rng, bias)
    repeat_walk, _ = observed(truth[0], rng, bias)
    strategies = {
        "walk": score([observed_walk], train, test),
        "combined": score([observed_walk, observed_challenge], train, test),
        "repeatWalk": score([observed_walk, repeat_walk], train, test),
        "excursion": score([observed_walk], train, test, True),
        "noiseFreeWalk": score([truth[0]], train, test),
        "noiseFreeCombined": score(truth, train, test),
    }
    # One shared bootstrap resample retains pairing across tasks and mechanisms.
    bootstrap = rng.integers(len(test), size=(2000, len(test)))
    delta = strategies["combined"].mean(axis=0) - strategies["walk"].mean(axis=0)

    def difference(a):
        d = a[0] - a[1]
        raw = np.sqrt(np.mean(d * d, axis=(1, 2)))
        # Equivalent to the maximum waveform-RMS ratio across all three axes.
        scaled = np.max(np.sqrt(np.mean((d / ERROR_DEG) ** 2, axis=1)), axis=1)
        return raw, scaled

    wr, ws = difference(truth[0])
    cr, cs = difference(truth[1])
    compatible = ws <= 1
    eligible = test[compatible[test]]
    # Deliberately showcase the largest challenge dissimilarity among walking matches.
    pair_id = (
        int(eligible[np.argmax(cs[eligible])])
        if len(eligible)
        else int(test[np.argmin(ws[test])])
    )
    cases = []
    for g in [1, 2]:
        cases.append(
            {
                "label": ["Abductor capacity", "Extension resistance"][g - 1],
                "mechanism": MECHANISMS[g],
                "anatomyId": pair_id,
                "parameters": {
                    "abductorCapacityBwM": round(
                        float(
                            p["maxAbductionMomentBwM"][pair_id]
                            * (p["availableFraction"][pair_id] if g == 1 else 1)
                        ),
                        5,
                    ),
                    "abductorArmM": round(float(p["abductorArmM"][pair_id]), 5),
                    "addedStiffnessBwMPerRad": round(
                        float(p["addedStiffnessBwMPerRad"][pair_id] if g == 2 else 0), 5
                    ),
                },
                "walk": {
                    "durationS": walk_time,
                    "frames": make_frames(walk[g, pair_id]),
                },
                "challenge": {
                    "durationS": challenge_time,
                    "frames": make_frames(challenge[g, pair_id]),
                },
            }
        )

    def summary(arr):
        return {
            "fraction": float(arr.mean()),
            "ci95": np.quantile(
                arr.mean(axis=0)[bootstrap].mean(axis=1), [0.025, 0.975]
            ).tolist(),
        }

    check_p = {k: v[:16] for k, v in p.items()}
    refinement = []
    for task, original in zip(["walk", "challenge"], [walk, challenge]):
        fine, _ = simulate(reference, check_p, task, steps=512)
        refinement.append(
            float(np.abs(fine[:, :, :, :3] - original[:, :16, :, :3]).max())
        )
    # Freeze training on the effort policy. Change the dynamics/control rule solely
    # for held-out bodies, preserving identical sampled camera errors.
    test_p = {k: v[test] for k, v in p.items()}
    alternate = []
    for task in ("walk", "challenge"):
        trace, _ = simulate(
            reference, test_p, task, groups=(1, 2), controller="reserve"
        )
        alternate.append(trace[:, :, :, :3])
    held_noisy = []
    for original, other_truth, perturbed in zip(
        truth, alternate, [observed_walk, observed_challenge]
    ):
        shifted = perturbed.copy()
        shifted[:, test] = other_truth + (perturbed[:, test] - original[:, test])
        held_noisy.append(shifted)
    policy_shift = {
        "walk": summary(
            score([observed_walk], train, test, testing_views=held_noisy[:1])
        ),
        "combined": summary(
            score(
                [observed_walk, observed_challenge],
                train,
                test,
                testing_views=held_noisy,
            )
        ),
    }
    load_sweep = []
    for load in (0, 0.05, 0.10, 0.15):
        if load == 0.10:
            angles = truth[1][:, test]
        else:
            loaded, _ = simulate(
                reference, test_p, "challenge", groups=(1, 2), hand_load_bw=load
            )
            angles = loaded[:, :, :, :3]
        raw, scaled = difference(angles)
        load_sweep.append(
            {
                "contralateralMassBw": load,
                "medianPairRmsDeg": float(np.median(raw)),
                "walkAndChallengeAmbiguousFraction": float(
                    (compatible[test] & (scaled <= 1)).mean()
                ),
            }
        )
    clinical_difference = float(
        (
            np.ptp(walk[1, test, :, 0], axis=1) - np.ptp(walk[2, test, :, 0], axis=1)
        ).mean()
    )
    return {
        "label": "Hip mechanism observability",
        "source": reference["source"],
        "seed": seed,
        "assumptions": [
            "Reduced-order right-stance tracking dynamics referenced to one healthy adult.",
            "Crossed physical mechanisms represent hypothetical mechanical impairments.",
            "Contralateral loaded stance is a virtual task. Camera errors are simulated.",
            "Abductor force is a lumped torque divided by an effective moment arm.",
        ],
        "tasks": [
            {"id": "walk", "label": "Walking stance", "durationS": walk_time},
            {
                "id": "challenge",
                "label": "Loaded single-leg stance",
                "durationS": challenge_time,
            },
        ],
        "metrics": {
            "nAnatomies": n,
            "nTestAnatomies": len(test),
            "nForwardTrials": n * 4 * 2,
            **{k + "BalancedAccuracy": float(v.mean()) for k, v in strategies.items()},
            "accuracyIntervals": {k: summary(v) for k, v in strategies.items()},
            "combinedMinusWalkCi95": np.quantile(
                delta[bootstrap].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "walkAmbiguousFraction": float(compatible[test].mean()),
            "combinedAmbiguousFraction": float(
                (compatible[test] & (cs[test] <= 1)).mean()
            ),
            "noiseRmsDeg": dict(zip(ANGLES, ERROR_DEG.tolist())),
            "maxIntegrationChangeDeg": max(refinement),
            "integrationSteps": 256,
            "pairedWalkingRmsMedianDeg": float(np.median(wr[test])),
            "pairedChallengeRmsMedianDeg": float(np.median(cr[test])),
        },
        "publishedForwardSensitivity": reference["publishedForwardSensitivity"],
        "sensitivity": {
            "unseenReservePolicy": policy_shift,
            "loadSweep": load_sweep,
            "addedForwardTrials": len(test) * (2 * 2 + 3 * 2),
            "heldOutClinicalComparison": {
                "simulatedCapacityMinusResistanceDeg": clinical_difference,
                "clinicalGtMinusOaDeg": 11.3,
                "clinicalCi95Deg": [9.1, 13.5],
                "source": "https://doi.org/10.1016/j.gaitpost.2018.02.011",
            },
        },
        "pair": {
            "description": "Same virtual anatomy under two separate impairments. Selected for the largest challenge difference among held-out pairs within the walking error budget.",
            "anatomyId": pair_id,
            "walkRmsDeg": float(wr[pair_id]),
            "challengeRmsDeg": float(cr[pair_id]),
            "walkMaxAxisRmsInErrorUnits": float(ws[pair_id]),
            "challengeMaxAxisRmsInErrorUnits": float(cs[pair_id]),
            "cases": cases,
        },
        "groupSummary": [
            {
                "mechanism": label,
                "task": task,
                "meanHipExcursionDeg": float(
                    np.ptp(trace[i, test, :, 0], axis=1).mean()
                ),
                "meanPeakAbductorForceBw": float(
                    trace[i, test, :, 3].max(axis=1).mean()
                ),
            }
            for i, label in enumerate(MECHANISMS)
            for task, trace in [("walk", walk), ("challenge", challenge)]
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anatomies", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--output", type=Path, default=DATA / "counterfactual.json")
    args = parser.parse_args()
    result = run(args.anatomies, args.seed)
    args.output.write_text(
        json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(json.dumps(result["metrics"], indent=2))
