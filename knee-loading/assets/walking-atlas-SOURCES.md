# Walking atlas

The display uses BodyParts3D 3.0, © The Database Center for Life Science, under CC Attribution-Share Alike 2.1 Japan. The walking atlas is an adaptation distributed under the same license. It contains 386 muscle and associated soft-tissue or facial structures, 242 skeletal structures and one external body surface. The named FMA structures and triangle counts are recorded on the three surface nodes in the GLB. [Publisher](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html) · [License](https://creativecommons.org/licenses/by-sa/2.1/jp/deed.en)

The surface coordinates come from the publisher’s September 2011, 99%-reduced OBJ release. The structure selection for the skeleton and the normalization transform follow the OrganUI skeleton used by the hip study. The source archive has SHA-256 `2125f6d761b14ead4ee64f71713f56f3f0f29982693ff20aac63474470cb92f4`. [Polygon meshes](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/20110915/BodyParts3D_3.0_obj_99.zip) · [English FMA labels](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/20110915/parts_list_e.txt) · [OrganUI provenance](../../hip-study/assets/body-SOURCES.md)

The preparation script simplifies the individual structures, converts axes to the walking frame, assigns display colors, calculates attachment blends and packs the result with meshoptimizer. The external surface has a smooth geometric patch across the anterior pelvis. This common reference anatomy is shared by all recordings. Muscle-surface color and fine longitudinal shading identify structures. They do not encode activation, force or subject-specific fiber direction.

During replay the atlas uses the shoulder, elbow, wrist, hip, knee and ankle locations and the orientations of the recorded segments. The femur and tibia each follow their own length. Both use the femur scale for width through the knee. This yields a distal femoral width of about 78 mm compared with 75 mm in the shared OpenSim geometry. The hands register the wrist-to-third-metacarpal axis and the radial axis given by the index and little proximal phalanges. Their twist follows the input hand quaternion. Forearm rotation blends from elbow to wrist. The rearfoot and toes use separate orientations. A fixed 17 mm adjustment in rearfoot-local coordinates aligns the reference sole to the model. These surface registrations are for visualization. The source calculations and knee measurements use the original model.

With Python 3, NumPy, SciPy, trimesh, fast-simplification and gltfpack 0.24.0 installed, run the following command from the repository root. It requires the two original release files linked above.

```sh
python3 knee-loading/tools/prepare_walking_atlas.py BodyParts3D_3.0_obj_99.zip parts_list_e.txt
```
