# Whole-body skeletal display

The local `body.glb` pairs the 22 body frames from the Harba et al. GC4 OpenSim model with the public OpenSim Models display geometry. It includes the skull, jaw, spine, rib cage, scapulae, pelvis, both legs and feet, both arms, wrists, and finger bones. The rig attaches the skull, jaw, ribs, and spine to the modeled torso and the fingers to their hand frame. Gait playback applies the dataset's body transforms in metres and does not add degrees of freedom to those bones.

The exporter reads the named display geometry and the model's scale factors. It triangulates polygons for GLB and computes normals from the surface. To complete the pelvis it mirrors the supplied hemipelvis and includes OpenSim's reference sacrum. The shared geometry library supplies the reference surface shapes. The displayed right knee uses the anatomical bone shapes.

Run this command from the repository root to generate the local file, using Python 3 and curl.

```sh
python3 knee-loading/tools/export_body.py --fetch
```

With the source model and geometry available locally, pass the OpenSim model path and the directory of VTP files instead of `--fetch`. The exporter pins both public GitHub revisions. The reconstruction is stored locally because the model library does not provide a blanket redistribution license covering all referenced meshes. The associated example models retain their individual licenses.

[Model and study materials](https://github.com/gilserrancoli/ReducePoly_forMSK/tree/00b6f94ee8bd2f60ae5b6f07a114f1aade089cb8) · [OpenSim display geometries](https://github.com/opensim-org/opensim-models/tree/d9b05d470b1a481c222372c85b75772faf8f7792/Geometry) · [OpenSim model licensing](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53086437)
