# Human Simulator

An interactive 3D anatomy and biomechanics prototype for the arm and leg. Explore elbow and knee flexion, isolate individual structures, and view major nerve paths alongside reference muscle–tendon mechanics.

## Run the viewer

```sh
cd upper-limb
npm ci
npm run dev -- --port 5178
```

Open [the arm viewer](http://127.0.0.1:5178/) or [the leg viewer](http://127.0.0.1:5178/?region=legs). Both use the same web app. Run `npm run build` from `upper-limb` to generate the hosted bundle and the offline `dist/upper-limb-viewer.html` and `dist/lower-limb-viewer.html`.

## Models and validation

- [Upper limb](upper-limb/README.md) describes the elbow rig, tendon attachment and contact audits, and Arm26 reference data. [Animated arm GLB](upper-limb/assets/upper-limb.glb)
- [Lower limb](lower-limb/README.md) describes the knee rig, tissue audits, and Rajagopal2016 reference data. [Animated leg GLB](lower-limb/assets/lower-limb.glb)
- [SimTK review](docs/simtk-review.md) identifies measured datasets for checking the rigs and resources for adding movements.

The viewer plays baked anatomy motion and displays generic reference mechanics. The load control displays an illustrative gravitational torque. Nerves are shown as anatomical paths. See each limb’s documentation for the scope of the geometric checks and the remaining validation work.

The animated GLBs adapt Open3DModel anatomy under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The limb documentation includes source credits, adaptation details, and attribution for the OpenSim-derived data.
