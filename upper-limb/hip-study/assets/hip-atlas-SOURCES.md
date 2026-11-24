# Hip atlas geometry

The 76 registered structures in this collection come from *Open3DModel – Lower limb – English labels*, by the Open3D Project, Jan Kooloos (RadboudUMC), Eungyeol Lee (LUMC) and collaborators. The geometry and this adaptation use Creative Commons Attribution-ShareAlike 4.0. [Publisher's atlas](https://anatomytool.org/content/open3dmodel-lower-limb-english-labels) · [Publisher credits](https://web.archive.org/web/20260728014822id_/https://anatomytool.org/open3dmodel-create) · [Original geometry archive](https://caskanatomy.info/open3dmodelfiles/lower-limb/lower-limb-glb.zip) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## Structures and coordinates

The extraction retains the right lower limb's neutral coordinates in metres, with Y superior and +Z anterior. The 14 display groups include the femur and pelvis, joint surfaces, hip abductors, fascia, superficial muscles and fascia, deep rotators, iliopsoas, adductors, proximal thigh, ligaments, nerves, lumbosacral context, bursae and vessels. Surfaces include the trochanteric bursae associated with gluteus medius, minimus and maximus, the subcutaneous trochanteric bursa, and seven other regional bursal structures. The original source identifies five ligaments or ligament regions as surface overlays on the capsule. Their original group, node ID, material ID and label remain in glTF node metadata.

The field extends from Y = 0.695 m in the proximal thigh to Y = 1.045 m above L5. At each boundary the extractor intersects triangles with a plane, interpolates the surface normals and texture coordinates, and triangulates each closed cut contour independently. The fascia lata retains its source sheet topology at the cut. No mesh smoothing, subdivision or decimation changes the anatomy between the section planes. The 155,155 triangles use 89,146 render vertices, including texture and normal seams.

The joint center comes from an algebraic least-squares sphere fitted to all 1,288 unique femoral-head cartilage vertices. The center is (−83.879, 860.086, −5.343) mm, the radius is 23.936 mm and the radial RMS residual is 0.485 mm. These are geometric measures of this atlas.

## Surface appearance

The asset preserves the original UV charts wherever the publisher supplies them and includes the publisher's normal maps for the hip bone, femur, sacrum, L5, coccyx, hip capsule, sacrotuberal ligament and posterior sacroiliac ligament. Muscle and connective tissue colour and relief maps are independently generated illustrations. No muscle texture imagery from the publisher is included. The gluteus medius and minimus each have one continuous source mesh, so a displayed muscle–tendon transition is a visual approximation. Bursa sizes and shapes are neutral anatomical illustrations. Load overlays and pose-dependent deformations come from the viewer.

Bone vertex colours encode ambient accessibility in linear RGB. Each vertex casts 96 deterministic cosine-weighted rays against its own bone from a 0.15 mm normal offset. Occlusion attenuates linearly over 70 mm. Computed intensities range from 0.684 to 1.000 for the hip bone, 0.433 to 1.000 for the sacrum and 0.663 to 1.000 for the femur.

Tissue motion uses attachment patches inferred from proximity to bone or anatomical section surfaces. Geodesic distance along the atlas surface and a smoothed spatial field determine how intervening vertices follow pelvic and femoral motion. The distal gluteal overlay uses distance along the surface from inferred attachment patches. [Deformation manifest](../../../hip-study/data/hip-deformation.json) · [Binding generator](../../../hip-study/tools/prepare_hip_deformation.py)

Optional muscle guides connect control points from the MRIInformed2023 and Davis2026 OpenSim models. The viewer aligns the hip center and neutral anatomical axes, then animates the controls according to their body assignments. Differences in source anatomy and unrepresented wrapping between control points can displace the guides from the displayed surfaces. [Reference coordinates and provenance](../../../hip-study/data/geometry.json)

## Reproduction

The motion lab applies a pose-dependent correction to the iliotibial tract using its interfaces with nearby tissues. The source mesh remains available in its original reference pose. [Contact model and validation](../../../hip-study/CONTACT.md)

Run this command from the repository root with Python and uv available.

```sh
uv run --no-project --with numpy --with scipy --with trimesh --with embreex --with DracoPy --with mapbox-earcut --with Pillow python hip-study/tools/prepare_hip_atlas.py
```

Use `--source path/to/lower-limb-glb.zip` for an offline rebuild. The extractor accepts the unzipped GLB as well. It checks the SHA-256 of the original GLB against `5ecb1ef4c07a2f6aa17021ee6f6d0bb0587595821467ec8c6a14f24ee1091173`. The output is `upper-limb/hip-study/assets/hip-atlas.glb`.
