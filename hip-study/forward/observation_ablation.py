"""How much does observing lateral trunk lean add to observing stance-side hip angles?

Run with NumPy and SciPy from the repository root:
    python3 hip-study/forward/observation_ablation.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

import experiment as model

SEED = 20261122
N_BODIES = 1024
N_TRAIN = 768
N_BOOTSTRAP = 2000
NEW_COHORTS = 20
TASKS = {"walk": (0,), "loadedStance": (1,), "walkAndLoadedStance": (0, 1)}
# A constant zero dimension contributes zero distance to the original 15-NN rule.
CHANNELS = {"hipAngles": (1, 1, 0), "hipAnglesAndTrunkLean": (1, 1, 1), "trunkLean": (0, 0, 1)}


def classify(training_views, testing_views, train, test):
    outcomes = {}
    for task, indices in TASKS.items():
        scores = {}
        for name, mask in CHANNELS.items():
            train_signals = [training_views[i] * mask for i in indices]
            test_signals = [testing_views[i] * mask for i in indices]
            correct = model.score(train_signals, train, test, testing_views=test_signals)
            # Equal numbers per class, each body contributes both counterfactuals.
            scores[name] = correct.mean(axis=0)
        outcomes[task] = scores
    return outcomes


def cohort(reference, seed, test_reserve=False):
    rng = np.random.default_rng(seed)
    body = model.population(N_BODIES, rng)
    train, test = np.split(rng.permutation(N_BODIES), [N_TRAIN])
    tasks = ("walk", "challenge")
    truth = [model.simulate(reference, body, task, groups=(1, 2))[0][..., :3] for task in tasks]
    walk, bias = model.observed(truth[0], rng)
    loaded, _ = model.observed(truth[1], rng, bias)
    noisy = [walk, loaded]
    # Consume the repeat-walk draw to match the primary experiment's bootstrap indices.
    model.observed(truth[0], rng, bias)
    bootstrap = rng.integers(len(test), size=(N_BOOTSTRAP, len(test)))
    policies = {"effort": classify(noisy, noisy, train, test)}
    if test_reserve:
        held_bodies = {key: values[test] for key, values in body.items()}
        shifted_test = []
        for i, task in enumerate(tasks):
            other, _ = model.simulate(reference, held_bodies, task, groups=(1, 2), controller="reserve")
            shifted = noisy[i].copy()
            shifted[:, test] = other[..., :3] + (noisy[i][:, test] - truth[i][:, test])
            shifted_test.append(shifted)
        policies["unseenTorqueReserve"] = classify(noisy, shifted_test, train, test)
    return policies, bootstrap


def contrast(outcomes, bootstrap):
    lower, upper = np.quantile(outcomes[bootstrap].mean(axis=1), [0.025, 0.975])
    return {"difference": float(outcomes.mean()), "pairedBootstrap95": [float(lower), float(upper)]}


def describe(scores, bootstrap):
    result = {}
    for task, streams in scores.items():
        added_trunk = streams["hipAnglesAndTrunkLean"] - streams["hipAngles"]
        result[task] = {
            "balancedAccuracy": {name: float(v.mean()) for name, v in streams.items()},
            "addedTrunkLean": contrast(added_trunk, bootstrap),
        }
    result["walkAndLoadedStanceMinusLoadedStance"] = {
        name: contrast(scores["walkAndLoadedStance"][name] - scores["loadedStance"][name], bootstrap)
        for name in CHANNELS
    }
    return result


def distribution(values):
    quantiles = np.quantile(values, [0, 0.1, 0.5, 0.9, 1])
    return dict(zip(("min", "p10", "median", "p90", "max"), map(float, quantiles)))


def lever_sensitivity(reference):
    rng = np.random.default_rng(SEED)
    all_bodies = model.population(N_BODIES, rng)
    ids = rng.permutation(N_BODIES)[N_TRAIN:]
    bodies = {key: vals[ids] for key, vals in all_bodies.items()}
    load = 0.10
    _, target = model.task_curve(reference, "challenge", len(model.OUTPUT_PHASE) - 1, load)
    phase = int(np.argmax(target[:, 3]))
    ground_force = float(target[phase, 3])
    uncoupled_demand = target[phase, 5] * bodies["externalScale"]

    def paired_outputs(steps):
        args = dict(task="challenge", groups=(0, 1), hand_load_bw=load, steps=steps)
        coupled, _ = model.simulate(reference, bodies, **args)
        decoupled, _ = model.simulate(reference, bodies, **args, apply_trunk_load_relief=False)
        if not np.array_equal(coupled[0], decoupled[0]):
            raise ValueError("Unperturbed placebo changed under removal of incremental lean relief.")
        if not np.array_equal(coupled[..., 2], decoupled[..., 2]):
            raise ValueError("Decoupling changed the achieved trunk motion.")
        added_lean = coupled[1, :, phase, 2] - coupled[0, :, phase, 2]
        relief = ground_force * bodies["trunkLeverM"] * np.sin(np.deg2rad(added_lean))
        hip_adduction = decoupled[1, :, phase, 1] - coupled[1, :, phase, 1]
        abductor_force = decoupled[1, :, phase, 3] - coupled[1, :, phase, 3]
        return {
            "extraStanceSideLeanDeg": added_lean,
            "externalMomentReliefBwM": relief,
            "externalMomentReliefPercent": 100 * relief / uncoupled_demand,
            "reductionInHipAdductionDeg": hip_adduction,
            "reductionInAbductorGroupForceBw": abductor_force,
        }

    values = paired_outputs(256)
    fine = paired_outputs(512)
    return {
        "question": "How much does the achieved compensatory lean alter frontal-plane hip mechanics inside the model?",
        "task": "Contralateral loaded single-leg stance, reduced abductor torque capacity",
        "pairedVirtualBodies": len(ids), "handLoadBw": load, "phase": float(model.OUTPUT_PHASE[phase]),
        "verticalGroundForceBw": ground_force,
        "intervention": "Disable only the reduction of external adduction moment by the achieved compensatory trunk lean. Both runs use the same anatomy, external load, hip tracking, strength limit and effort-minimizing trunk controller.",
        "signConvention": "Positive relief or reduction means the coupled simulation has the lower value. Each contrast pairs the same virtual body at the same instant. Distribution percentiles summarize model heterogeneity.",
        "demandWithoutTrunkReliefBwM": distribution(uncoupled_demand),
        "pairedDifferences": {key: distribution(val) for key, val in values.items()},
        "bodiesWithSameGroupForceAtPeak": int(np.count_nonzero(
            np.isclose(values["reductionInAbductorGroupForceBw"], 0, atol=1e-12, rtol=0))),
        "controls": {
            "nominalBodyAllOutputsIdentical": True,
            "achievedTrunkMotionIdenticalAcrossSwitch": True,
            "maxAbsoluteChangeInEachEffectOnDoublingSteps": {
                key: float(np.max(np.abs(v - fine[key]))) for key, v in values.items()
            },
        },
        "interpretation": "The reduced-order equations assign a shorter frontal-plane load lever to stance-side lean. The hip may respond with less adduction even while the abductor group remains torque-limited. The switch isolates this equation term with prescribed ground loading. Tissue-specific force and clinical effects require independent measurements.",
    }


def study():
    reference = json.loads((model.HERE / "reference.json").read_text())
    policies, bootstrap = cohort(reference, SEED, test_reserve=True)
    draws = []
    for seed in range(SEED + 1, SEED + 1 + NEW_COHORTS):
        rerun, _ = cohort(reference, seed)
        added = {task: float((v["hipAnglesAndTrunkLean"] - v["hipAngles"]).mean())
                 for task, v in rerun["effort"].items()}
        draws.append({"seed": seed, "addedTrunkLean": added})
    replication = {}
    for task in TASKS:
        values = np.array([draw["addedTrunkLean"][task] for draw in draws])
        replication[task] = {
            "meanDifference": float(values.mean()),
            "minDifference": float(values.min()),
            "maxDifference": float(values.max()),
            "positiveCohorts": int(np.count_nonzero(values > 0)),
        }
    return {
        "question": "Does observing trunk lean improve identification of two imposed mechanical impairments?",
        "seed": SEED, "trainingBodies": N_TRAIN, "heldOutBodies": N_BODIES - N_TRAIN,
        "bootstrapResamples": N_BOOTSTRAP, "accuracyUnits": "fraction (1.0 = 100%)",
        "methods": [
            "Every virtual anatomy receives both perturbations: reduction in abductor torque capacity or additional passive hip extension resistance.",
            "The same simulated hip–trunk dynamics generate every observation. Only the channels supplied to the observer vary.",
            "The body split, simulated camera noise, nine phases per task, error scaling and 15-nearest-neighbor classifier are the original experiment's.",
            "Hip angles are flexion and adduction. Lateral trunk lean is the third channel. The camera-like trunk error has an assumed 3-degree RMS scale.",
            "The paired percentile bootstrap resamples held-out body IDs and keeps both perturbations, tasks and channel sets together, conditional on training.",
            "All classifiers train on the effort-minimizing controller. The alternate test controller maintains torque reserve. Its test bodies receive the same camera errors.",
            "The additional twenty consecutive seeds independently regenerate anatomies, split, training, and observation errors with the effort controller. Their ranges describe variation across runs.",
        ],
        "originalCohort": {name: describe(scores, bootstrap) for name, scores in policies.items()},
        "additionalCohorts": {"count": NEW_COHORTS, "summary": replication, "details": draws},
        "mechanicalLeverSensitivity": lever_sensitivity(reference),
        "interpretation": "The experiment tests the incremental utility of observing a signal inside a three-coordinate model. Both the trunk compensation rules and noise model are imposed. These results motivate measuring hip angles and trunk lean during loaded stance in people. Human diagnosis, tissue-specific pain and the added value of full-body forward dynamics require separate tests.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=model.DATA / "observation-ablation.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(study(), indent=2, allow_nan=False) + "\n")
