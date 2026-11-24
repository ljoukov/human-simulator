"""Independently verify CSV force scale, orientation and time against all companion C3Ds.

python3 -m pip install c3d numpy
python3 hip-study/tools/audit_implants.py --cache /tmp/orthoload-hip
Run prepare_implants.py first. Outputs only numerical QA summaries.
"""
import argparse
import io
import json
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import urlencode

import c3d
import numpy as np

from prepare_implants import PLATE_CENTERS_MM, TRIALS


def source_file(cache, trial):
    existing = list(cache.glob(trial.lower() + "_*_mov.zip"))
    if len(existing) == 1:
        return existing[0]

    def lookup(params):
        url = "https://orthoload.com/wp-admin/admin-ajax.php?" + urlencode(params)
        return json.loads(subprocess.check_output(["curl","-fsSL",url]))["data"]

    ident = trial.lower() + "_mov"
    entry = next(row for row in lookup({"action":"orthoload_searchData","term":ident}) if row["fileId"] == ident)
    files = lookup({**entry,"action":"orthoload_get_data"})
    archive = next(row for row in files if row["type"] == "RAW_MOV")
    destination = cache / Path(archive["path"]).name
    subprocess.run(["curl","-fsSL","--retry","2","https://orthoload.com"+archive["path"],"-o",str(destination)],check=True)
    return destination


def audit(cache, trial):
    with zipfile.ZipFile(source_file(cache, trial)) as archive:
        fname = next(n for n in archive.namelist() if n.lower().endswith(".c3d"))
        reader = c3d.Reader(io.BytesIO(archive.read(fname)))
        plate_centers = reader.get("FORCE_PLATFORM:CORNERS").float_array.mean(axis=1)
        np.testing.assert_allclose(plate_centers, np.c_[PLATE_CENTERS_MM, [0.,0.]], atol=.1)
        analogs = np.concatenate([a for _, _, a in reader.read_frames()], axis=1).T
        original_labels = [s.strip().upper() for s in reader.analog_labels]

    with (cache/(trial.lower()+"_mov.csv")).open(encoding="latin1") as stream:
        for line in stream:
            if line.lower().startswith("time,frame,"):
                break
        labels = [s.strip().upper() for s in line.split(",")]
        next(stream)
        data = np.loadtxt(stream,delimiter=",")
    delta_t = np.diff(data[:,0])
    np.testing.assert_allclose(delta_t,1/reader.analog_rate,atol=1.1e-5)
    assert len(analogs) == len(data), "Original and CSV sampling grids differ"
    errors = []
    for idx, directions in enumerate([(-1, 1, -1), (1, -1, -1)],start=1):
        for axis, sign in zip("XYZ", directions):
            label = f"F{axis}{idx}"
            reference = sign * analogs[:,original_labels.index(label)]
            exported = data[:,labels.index(label)] * (1000. if axis == "Z" else 1.)
            present = np.isfinite(exported)
            if present.any():
                errors.append(np.max(np.abs(exported[present]-reference[present])))
    error = float(max(errors))
    if error > .52:
        raise ValueError(f"{trial}: force plate discrepancy {error:.2f} N")
    observed = data[:,[labels.index(c) for c in ["FX","FY","FZ"]]]
    norm_error = np.nanmax(np.abs(np.linalg.norm(observed,axis=1) - data[:,labels.index("FRES")]))
    if norm_error > 2.:
        raise ValueError(f"{trial}: telemetry norm discrepancy {norm_error:.3f} N")
    return {"id":trial,"maxPlateComponentErrorN":round(error,4), "maxTelemetryNormErrorN":round(float(norm_error),5)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",required=True,type=Path)
    arguments = parser.parse_args()
    print(json.dumps([audit(arguments.cache,t) for pair in TRIALS for t in pair],indent=2))
