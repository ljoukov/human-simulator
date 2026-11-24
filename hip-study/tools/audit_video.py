"""Run an offline markerless gait audit against a matching motion-capture file.

Inputs in Patients/HOA001/M0/gait/ are gait03.60948372.avi and gait03.c3d
from https://doi.org/10.6084/m9.figshare.14420645.v2 (Bertaux et al., CC BY 4.0).
The full float16 v1 pose model is linked in MODEL_URL below.
Install mediapipe==0.10.32, opencv-python-headless==4.12.0.88, c3d==0.6.0,
numpy==2.2.6. Example:
    python audit_video.py VIDEO.avi MOTION.c3d pose_landmarker_full.task --out video-audit.json
Only derived numeric results are written. No source images or facial landmarks are saved.
"""

import argparse
import hashlib
import json
from pathlib import Path

import c3d
import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
HASHES = {
    "avi": "5cea55df105c6620a462f0ca531094c0b8fdbe3b2425e32e52efa6b1a0caba4e",
    "c3d": "3cbb12f2b060075be74b51858bc07967659b6000d615683437aa98609a904d39",
    "model": "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1",
}
# Corresponding Vicon Plug-in Gait joint centers and MediaPipe landmarks.
C3D_JOINTS = ["LCLO", "RCLO", "LFEP", "RFEP", "LTIO", "RTIO", "LFEO", "RFEO"]
MP_IDS = [11, 12, 23, 24, 27, 28, 25, 26]
JOINT_NAMES = [
    "leftShoulder",
    "rightShoulder",
    "leftHip",
    "rightHip",
    "leftAnkle",
    "rightAnkle",
    "leftKnee",
    "rightKnee",
]
# Two shoulders, two hips, two ankles. Both knees are held out everywhere.
CALIBRATION = slice(0, 6)
RIGHT_HIP, RIGHT_KNEE = 3, 7
CONFIDENCE = 0.5


def pose_samples(video, model_file):
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lo, hi = round(0.25 * width), round(0.80 * width)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_file), delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    rows, times = [], []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as estimator:
        while capture.grab():
            frame = round(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if frame % 2:
                continue
            ok, bgr = capture.retrieve()
            if not ok:
                raise ValueError(f"Could not decode frame {frame}")
            rgb = cv2.cvtColor(bgr[:, lo:hi], cv2.COLOR_BGR2RGB)
            t = frame / fps
            result = estimator.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(t * 1000)
            )
            row = np.full((len(MP_IDS), 4), np.nan)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                for i, key in enumerate(MP_IDS):
                    p = landmarks[key]
                    row[i] = [
                        lo + p.x * (hi - lo),
                        p.y * height,
                        p.visibility,
                        p.presence,
                    ]
            rows.append(row)
            times.append(t)
    n_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return (
        np.array(times),
        np.array(rows),
        {
            "containerRateHz": fps,
            "analysisRateHz": fps / 2,
            "decodedFrameCount": n_frames,
            "sampleCount": len(times),
            "width": width,
            "height": height,
            "roiXPx": [lo, hi],
        },
    )


def reference_samples(path):
    with path.open("rb") as stream:
        reader = c3d.Reader(stream)
        labels = [s.strip() for s in reader.point_labels]
        indices = [labels.index(k) for k in C3D_JOINTS]
        times = (
            np.arange(reader.first_frame, reader.last_frame + 1) - 1
        ) / reader.point_rate
        p = np.stack([values[indices] for _, values, _ in reader.read_frames()])
        p[p[:, :, 3] < 0, :3] = np.nan
        events = reader.get("EVENT:TIMES").float_array
        event_times = (
            events[1] + 60 * events[0]
            if events.shape[0] == 2
            else events[:, 1] + 60 * events[:, 0]
        )
        strikes = [
            float(t)
            for t, e, s in zip(
                event_times,
                reader.get("EVENT:LABELS").string_array,
                reader.get("EVENT:CONTEXTS").string_array,
            )
            if e.strip() == "Foot Strike" and s.strip() == "Right"
        ]
        return times, p[:, :, :3] / 1000, strikes[1:3]


def project(matrix, points):
    """Project [N, K, 3] points in metres into the decoded pixel grid."""
    h = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
    projected = h @ matrix.T
    return projected[..., :2] / projected[..., 2:]


def fit_camera(points, pixels):
    """Robust projective DLT. The calibration support spans the length of the walkway."""
    center = points.mean(axis=0)
    spread = points.std(axis=0)
    xyz = np.column_stack(((points - center) / spread, np.ones(len(points))))
    uv_center = pixels.mean(axis=0)
    uv_spread = pixels.std(axis=0)
    uv = (pixels - uv_center) / uv_spread
    rows = np.zeros((len(xyz), 2, 12))
    rows[:, 0, :4] = xyz
    rows[:, 1, 4:8] = xyz
    rows[:, :, 8:] = -uv[:, :, None] * xyz[:, None, :]
    weights = np.ones(len(points))
    for _ in range(4):
        design = (rows * np.sqrt(weights)[:, None, None]).reshape(-1, 12)
        _, _, vt = np.linalg.svd(design, full_matrices=False)
        matrix = vt[-1].reshape(3, 4)
        residual = np.linalg.norm(
            ((xyz @ matrix.T)[:, :2] / (xyz @ matrix.T)[:, 2:] - uv) * uv_spread, axis=1
        )
        weights = np.minimum(1, 5 / np.maximum(residual, 1e-10))
    # Denormalize both sides of the projective transform.
    world = np.eye(4)
    world[:3, :3] = np.diag(1 / spread)
    world[:3, 3] = -center / spread
    image = np.eye(3)
    image[:2, :2] = np.diag(uv_spread)
    image[:2, 2] = uv_center
    return image @ matrix @ world


def inclination(pixels):
    # x right, y down in both arrays. Zero for a vertical downward thigh.
    segment = pixels[:, RIGHT_KNEE] - pixels[:, RIGHT_HIP]
    return np.degrees(np.arctan2(segment[:, 0], segment[:, 1]))


def audit(t, pose, video_meta, t_ref, world, strikes):
    start, end = strikes
    dt = float(np.median(np.diff(t_ref)))
    vis = (pose[:, :, 2] >= CONFIDENCE) & (pose[:, :, 3] >= CONFIDENCE)
    vis &= (
        (pose[:, :, 0] >= 0)
        & (pose[:, :, 0] < video_meta["width"])
        & (pose[:, :, 1] >= 0)
        & (pose[:, :, 1] < video_meta["height"])
    )
    # Use the C3D time as master. Choose only video samples on the 50 Hz grid.
    # A ±0.4 s guard around the held-out stride and record boundaries keeps
    # both samples of every calibration correspondence outside the test stride
    # for every candidate timing offset. No per-landmark time warping.
    guard = 0.4
    calibration_times = t[
        (t >= t_ref[0] + guard)
        & (t <= t_ref[-1] - guard)
        & ((t < start - guard) | (t > end + guard))
    ]
    video_rate = video_meta["analysisRateHz"]
    ref_idx = np.rint((calibration_times - t_ref[0]) / dt).astype(int)
    train_world = world[ref_idx, CALIBRATION]

    def fit_shift(shift_frames):
        idx = np.rint(calibration_times * video_rate).astype(int) + shift_frames
        mask = vis[idx, CALIBRATION] & np.isfinite(train_world).all(axis=2)
        xy = pose[idx, CALIBRATION, :2][mask]
        xyz = train_world[mask]
        matrix = fit_camera(xyz, xy)
        error = np.linalg.norm(project(matrix, xyz) - xy, axis=1)
        return shift_frames, matrix, float(np.median(error)), len(xy)

    zero = fit_shift(0)
    max_shift = round(guard * video_rate)
    fits = [fit_shift(k) for k in range(-max_shift, max_shift + 1)]
    best = min(fits, key=lambda row: row[2])

    stride_t = t[(t >= start) & (t <= end)]
    reference = world[np.rint((stride_t - t_ref[0]) / dt).astype(int)]

    def evaluate(fit):
        shift, matrix, train_residual, n_train = fit
        idx = np.rint(stride_t * video_rate).astype(int) + shift
        estimated = inclination(pose[idx, :, :2])
        projected = project(matrix, reference)
        target = inclination(projected)
        good = vis[idx, RIGHT_HIP] & vis[idx, RIGHT_KNEE] & np.isfinite(target)
        err = estimated[good] - target[good]
        if np.sum(good) < 3:
            raise ValueError("Insufficient evaluable right hip–knee pairs")
        train_idx = np.rint(calibration_times * video_rate).astype(int) + shift
        good_train = vis[train_idx, RIGHT_HIP] & vis[train_idx, RIGHT_KNEE]
        constant = float(np.median(inclination(pose[train_idx, :, :2])[good_train]))
        knee_pixel_error = np.linalg.norm(
            projected[good, RIGHT_KNEE] - pose[idx[good], RIGHT_KNEE, :2], axis=1
        )

        def xy_list(points, valid):
            return [
                np.round(point, 2).tolist() if ok else None
                for point, ok in zip(points, valid)
            ]

        return {
            "startS": round(start, 3),
            "endS": round(end, 3),
            "nFrames": len(stride_t),
            "nValid": int(good.sum()),
            "maeDeg": round(float(np.abs(err).mean()), 3),
            "rmseDeg": round(float(np.mean(err**2) ** 0.5), 3),
            "pearsonR": round(
                float(np.corrcoef(estimated[good], target[good])[0, 1]), 4
            ),
            "biasDeg": round(float(err.mean()), 3),
            "peakToPeakVideoDeg": round(float(np.ptp(estimated[good])), 3),
            "peakToPeakReferenceDeg": round(float(np.ptp(target[good])), 3),
            "medianKneePositionErrorPx": round(float(np.median(knee_pixel_error)), 2),
            "rmsKneePositionErrorPx": round(
                float(np.sqrt(np.mean(knee_pixel_error**2))), 2
            ),
            "constantVideoBaselineDeg": round(constant, 3),
            "constantVideoBaselineMaeDeg": round(
                float(np.mean(np.abs(target[good] - constant))), 3
            ),
            "baselineMethod": "Median confident right-thigh tilt from the disjoint registration-period video frames, held constant throughout the test stride.",
            "videoOffsetS": round(shift / video_rate, 3),
            "nCalibrationCorrespondences": n_train,
            "calibrationMedianResidualPx": round(train_residual, 3),
            "timeline": [
                {
                    "phase": round(float((time - start) / (end - start)), 4),
                    "videoDeg": round(float(est), 3) if ok else None,
                    "referenceDeg": round(float(ref), 3) if np.isfinite(ref) else None,
                    "videoXY": xy_list(pose[idx[i], :, :2], vis[idx[i]]),
                    "referenceXY": xy_list(
                        projected[i], np.isfinite(projected[i]).all(axis=1)
                    ),
                }
                for i, (time, est, ref, ok) in enumerate(
                    zip(stride_t, estimated, target, good)
                )
            ],
        }

    stride = evaluate(best)
    nominal = evaluate(zero)
    nominal.pop("timeline")
    return {
        "schemaVersion": 1,
        "jointNames": JOINT_NAMES,
        "pixelAxes": "Original full 644 × 484 frame in pixels. x increases rightward and y downward. Missing points are null.",
        "title": "Monocular right thigh inclination compared with projected motion capture",
        "source": "Bertaux et al., Scientific Data 9, 399 (2022). CC BY 4.0.",
        "sourceUrl": "https://doi.org/10.6084/m9.figshare.14420645.v2",
        "paperUrl": "https://doi.org/10.1038/s41597-022-01483-3",
        "trial": "Patients/HOA001/M0/gait/gait03",
        "cameraFile": "gait03.60948372.avi",
        "participantCount": 1,
        "trialCount": 1,
        "strideCount": 1,
        "sourceHashes": HASHES,
        "estimator": {
            "name": "MediaPipe Pose Landmarker Full",
            "version": mp.__version__,
            "modelUrl": MODEL_URL,
            "opencvVersion": cv2.__version__,
            "mode": "VIDEO, default tracker, CPU inference",
            "visibilityAndPresenceMinimum": CONFIDENCE,
        },
        "video": {
            **video_meta,
            "confidentRightThighFramesFullVideo": int(
                (vis[:, RIGHT_KNEE] & vis[:, RIGHT_HIP]).sum()
            ),
        },
        "metric": "Signed pixel-plane angle from downward vertical of the right hip-to-knee segment. Positive points toward image right.",
        "cameraFit": "Direct linear projection with 5-pixel Huber reweighting, estimated from paired shoulders, hips and ankles outside the evaluated stride. All knees excluded during fitting. The image crop spans the central walking corridor.",
        "timing": {
            "convention": "Video time = C3D time + videoOffsetS. C3D time = (frame − 1) / 100. AVI time = zero-based frame / header rate.",
            "calibrationC3dTimesS": calibration_times.round(3).tolist(),
            "offsetSearchRangeS": [-guard, guard],
            "stepS": 1 / video_rate,
            "criterion": "Minimum median 2D calibration reprojection error; all candidate pairings outside the test stride.",
            "cameraCalibration": "Estimated on this recording. No independent physical camera calibration or hardware timecode used.",
        },
        "stride": stride,
        "nominalClockSensitivity": nominal,
        "limits": "One rear-view stride from one preoperative OA participant. Both landmark definitions and camera fit contribute to discrepancies. Image-plane tilt does not measure anatomical hip adduction, rotation, flexion, or tissue condition.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("c3d", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name, path in (("avi", args.video), ("c3d", args.c3d), ("model", args.model)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
            raise ValueError(f"Unexpected {name} file content")
    output = audit(*pose_samples(args.video, args.model), *reference_samples(args.c3d))
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
