# Recorded-stride muscle and ligament paths

The optional 3D overlay shows 13 bilateral muscle–tendon paths and three right patellar ligament bundles from the GC4-JW model. The muscle groups are rectus femoris, vastus lateralis and medialis, long head of biceps femoris, semitendinosus, medial and lateral gastrocnemius, soleus, tibialis anterior, fibularis longus, the middle gluteus medius path, external oblique and erector spinae. The implanted right knee supplies lateral, central and medial ligament paths with separate patellar and tibial attachments. [Pinned GC4-JW model](https://github.com/gilserrancoli/ReducePoly_forMSK/blob/00b6f94ee8bd2f60ae5b6f07a114f1aade089cb8/MuscleTendonLengthParameterizationReduction/JW%20All%20Body%20Model-scaled_forvisualization_leftright_v42.osim) · [Source MIT license](HARBA-LICENSE.txt)

OpenSim evaluates each GeometryPath at 51 evenly spaced phases in each complete recorded stride, using the same inverse-kinematic coordinates as the skeletal replay. Paths include active conditional points, moving insertions and points along wrap surfaces. The right quadriceps use the source paths that terminate on the patella. Tube width is a display setting. The paths record modeled routing. The recordings supply movement and knee load. Muscle activity, the position of a muscle–tendon junction, tendon strain and ligament force cannot be read from the overlay.

The exporter resamples each curve at equal arclength, subtracts the pelvis translation and stores XYZ coordinates as little-endian signed 16-bit integers in 0.1 mm units. The viewer interpolates adjacent samples, adds the current pelvis translation and draws occluded portions faintly. The optional stride-loop transition blends between the last and first sampled routes.

Regenerate with the pinned repository checked out locally and the Python dependencies listed in `knee-loading/tools/prepare_whole_body.py`.

```sh
python3 knee-loading/tools/prepare_muscle_paths.py /path/to/ReducePoly_forMSK
```
