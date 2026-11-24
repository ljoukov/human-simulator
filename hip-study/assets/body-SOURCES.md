# Full-body display geometry

BodyParts3D, © The Database Center for Life Science, licensed under [CC Attribution-Share Alike 2.1 Japan](https://creativecommons.org/licenses/by-sa/2.1/jp/deed.en). This license covers both GLB files and adaptations of their geometry. [BodyParts3D 3.0 dataset](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html) · [Mitsuhashi et al.](https://doi.org/10.1093/nar/gkn613)

**body.glb** is the unmodified 242-structure skeleton distributed by OrganUI, including skull bones, vertebrae, discs, ribs, bilateral arms and fingers, pelvis, legs and feet. OrganUI converted the 95%-reduced BodyParts3D 3.0 release from STL, welded vertices, simplified meshes, transformed their axes, recalculated normals, and applied display materials. The pinned file has SHA-256 c68b05d97d70c4d33363bf71d9885c191eb25f61a82dbba935a726ddd8e60ac8. [File and original attribution](https://github.com/organui/skeleton-3d/tree/120fc17278559057b37aff516eee89d263b604c6/public/models)

**body-soft-tissue.glb** contains 32 registered surfaces from the same BodyParts3D release. Both sides include gluteus maximus, gluteus medius, gluteus minimus, tensor fasciae latae, iliotibial tract, iliacus, psoas major, piriformis, quadratus femoris, obturator internus, pectineus, adductor longus, adductor magnus, rectus femoris, vastus lateralis and semitendinosus. The source surfaces share one coordinate system with the skeleton. We reduced each to approximately 2,200–5,500 triangles before applying the skeleton atlas coordinate transform. The preparation script lists the original FMA identifiers and triangle budget for every surface. [Pinned source meshes](https://github.com/Kevin-Mattheus-Moerman/BodyParts3D/tree/f0eeb6e843380cfe6b83797cf8c3e1af74de5e61/assets/BodyParts3D_data/stl)

With NumPy, trimesh and fast-simplification installed, regenerate the soft-tissue GLB with `python hip-study/tools/prepare_body_surfaces.py`. The full-body skeleton uses the original OrganUI archive linked above.

## Motion registration

Each segment connects its two clip landmarks. Bilateral hip and shoulder landmarks set the transverse axes. Heel-to-forefoot direction orients each foot. The pelvis, rib cage and head use separate transforms, and the lumbar and cervical stacks span their adjacent segments. The interpolation preserves the input positions at sampled phases.

Landmarks alone leave long-axis rotation underdetermined. The renderer uses the transverse pelvic or shoulder axis to resolve this degree of freedom. Atlas proportions are scaled to the available landmarks, and soft-tissue attachments blend neighbouring transforms. The displayed surfaces therefore provide a common anatomical reference across subjects. Clip metadata identifies the landmarks reconstructed from sparse observations.

Soft-tissue surfaces follow a geometric attachment approximation. The gluteal and adductor envelopes blend pelvic and femoral motion, the short rotators transition from medial pelvic attachments to lateral femoral insertions, and the superior psoas follows the lumbar spine. Distal surfaces that cross the knee blend with the proximal tibia. These weights approximate changes in surface position. Tissue strain, inflammation and individual muscle activation require separate measurements or models.

In the full-body view, teal identifies gluteus medius, gluteus minimus and tensor fasciae latae on the clip's selected side. Muscle color denotes anatomy and side. The abductor setting shows those three muscles bilaterally, exposing the minimus beneath the medius where the surfaces permit. The bone setting exposes the underlying skeleton.
