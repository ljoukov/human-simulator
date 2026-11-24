// Directional unilateral membrane contact for the hip atlas. The same source is
// compiled to a native shared library and to WebAssembly without a C runtime.
typedef unsigned int uint;
typedef struct {
    double x, y, z;
} Vec3;
typedef struct {
    uint a, b, c;
    double ymin, ymax, zmin, zmax, xmax;
} BandFace;
typedef struct {
    double y, z, a, b;
} ClipVertex;
typedef struct {
    uint face, tag;
    float a, b, gap, rawGap;
} Contact;
#define NV_MAX 50000
#define BAND_MAX 1024
#define FACE_MAX 2200
#define GRID 28
#define BIN_MAX 100000
#define CONTACT_MAX 350000
static Vec3 p[NV_MAX], output[BAND_MAX];
static BandFace faces[FACE_MAX];
static Contact contacts[CONTACT_MAX];
static double correction[BAND_MAX], bound[BAND_MAX], add[BAND_MAX];
// Metrics are metres unless noted: [0] maximum preprojection scoped deficit,
// [1] remaining feasible deficit, [2] active polygon points, [3] residual
// before cap allowance, [4..11] residual by layer, [12..19] scoped initial
// deficit by layer, [20] largest-residual band face, [21..29] geometric
// deficits before removing registered embedding of fixed corner material.
static double sinLut[1025], cosLut[1025], metrics[32];
static int head[GRID * GRID], next[BIN_MAX], binned[BIN_MAX], seen[FACE_MAX];
static uint shape[4];
static const float *rest, *metric, *displacementCap;
static const uint *faceMask, *obstacleTag;
static const uint *tris;
static char input[5000000] __attribute__((aligned(8)));
static double y0, z0, sy, sz;
static const double clearance = .00025, maxSlope = .28;
static uint contactCount;
double sin(double);
double cos(double);

char *contact_input(void) { return input; }
Vec3 *contact_output(void) { return output; }
double *contact_offsets(void) { return correction; }
double *contact_metrics(void) { return metrics; }
// Packed little-endian input: four uint counts, four float values per vertex
// (neutral XYZ and femur weight), oriented faces, band-material distances,
// per-vertex displacement caps, per-face layer masks, obstacle-face tags.
int contact_initialize(uint bytes) {
    const uint *h = (const uint *)input;
    if (bytes < 16 || bytes > sizeof(input))
        return -1;
    for (uint j = 0; j < 4; j++)
        shape[j] = h[j];
    uint n = shape[0], b = shape[1], f = shape[2], o = shape[3];
    if (!b || !f || n > NV_MAX || b > BAND_MAX || f > FACE_MAX || b > n || o > 100000)
        return -2;
    if (bytes != 16 + n * 16 + (f + o) * 12 + b * b * 4 + 4 * (b + f + o))
        return -3;
    rest = (const float *)(input + 16);
    tris = (const uint *)(rest + 4 * n);
    metric = (const float *)(tris + 3 * (f + o));
    displacementCap = metric + b * b;
    faceMask = (const uint *)(displacementCap + b);
    obstacleTag = faceMask + f;
    for (uint i = 0; i < 3 * (f + o); i++)
        if (tris[i] >= (i < 3 * f ? b : n))
            return -4;
    return 0;
}
static double max(double a, double b) { return a > b ? a : b; }
static double min(double a, double b) { return a < b ? a : b; }
static int bin(double v) { return v < 0 ? 0 : (v >= GRID ? GRID - 1 : (int)v); }
static Vec3 rotate(Vec3 v, Vec3 axis, double s, double c) {
    double d = (v.x * axis.x + v.y * axis.y + v.z * axis.z) * (1 - c);
    Vec3 r = {c * v.x + s * (axis.y * v.z - axis.z * v.y) + d * axis.x,
              c * v.y + s * (axis.z * v.x - axis.x * v.z) + d * axis.y,
              c * v.z + s * (axis.x * v.y - axis.y * v.x) + d * axis.z};
    return r;
}
static double edge(Vec3 a, Vec3 b, double y, double z) {
    return (b.y - a.y) * (z - a.z) - (b.z - a.z) * (y - a.y);
}
static int clip_half_plane(const ClipVertex *src, int n, ClipVertex *dst, Vec3 a, Vec3 b,
                           double orient) {
    int count = 0;
    if (!n)
        return 0;
    ClipVertex prev = src[n - 1];
    double e0 = orient * edge(a, b, prev.y, prev.z);
    for (int i = 0; i < n; i++) {
        ClipVertex cur = src[i];
        double e1 = orient * edge(a, b, cur.y, cur.z);
        if ((e0 < 0) != (e1 < 0)) {
            double u = e0 / (e0 - e1);
            ClipVertex v = {prev.y + u * (cur.y - prev.y), prev.z + u * (cur.z - prev.z),
                            prev.a + u * (cur.a - prev.a), prev.b + u * (cur.b - prev.b)};
            dst[count++] = v;
        }
        if (e1 >= 0)
            dst[count++] = cur;
        prev = cur;
        e0 = e1;
    }
    return count;
}
// Projected triangle clipping preserves barycentric coordinates on the band.
// The signed lateral gap is affine over each clipped polygon. Its maximum is
// attained at a polygon vertex, including cases of interior edge crossing.
static int contact_pair(uint id, Vec3 A, Vec3 B, Vec3 C, double winding, double depthSlopeY,
                        double depthSlopeZ, uint tag) {
    BandFace f = faces[id];
    Vec3 a = p[f.a], b = p[f.b], c = p[f.c];
    double gapA = a.x - (A.x + depthSlopeY * (a.y - A.y) + depthSlopeZ * (a.z - A.z)) + clearance;
    double gapB = b.x - (A.x + depthSlopeY * (b.y - A.y) + depthSlopeZ * (b.z - A.z)) + clearance;
    double gapC = c.x - (A.x + depthSlopeY * (c.y - A.y) + depthSlopeZ * (c.z - A.z)) + clearance;
    if (max(max(gapA, gapB), gapC) <= 0)
        return 0;
    ClipVertex src[8] = {{a.y, a.z, 1, 0}, {b.y, b.z, 0, 1}, {c.y, c.z, 0, 0}}, dst[8];
    int n = clip_half_plane(src, 3, dst, A, B, winding);
    n = clip_half_plane(dst, n, src, B, C, winding);
    n = clip_half_plane(src, n, dst, C, A, winding);
    for (int i = 0; i < n; i++) {
        double u = dst[i].a, v = dst[i].b, w = 1 - u - v;
        double rawGap = u * gapA + v * gapB + w * gapC;
        if (rawGap <= 0)
            continue;
        double gap = rawGap;
        // Portions of a triangle attached to bone, a section edge, or deep
        // TFL retain their registered embedding. Remove only the positive
        // contribution of exactly fixed corners from the unilateral condition.
        if (displacementCap[f.a] <= 1e-10)
            gap -= u * max(0., gapA);
        if (displacementCap[f.b] <= 1e-10)
            gap -= v * max(0., gapB);
        if (displacementCap[f.c] <= 1e-10)
            gap -= w * max(0., gapC);
        gap = max(0., gap);
        if (contactCount >= CONTACT_MAX)
            return -5;
        contacts[contactCount++] =
            (Contact){id, tag, (float)u, (float)v, (float)gap, (float)rawGap};
        metrics[0] = max(metrics[0], gap);
        // A slope-limited field satisfying the barycentric contact must also
        // satisfy these lower bounds at the three corners of the triangle.
        const uint corners[3] = {f.a, f.b, f.c};
        for (uint k = 0; k < 3; k++) {
            uint i = corners[k];
            const float *dist = metric + i * shape[1];
            double envelope = gap - maxSlope * (u * dist[f.a] + v * dist[f.b] + w * dist[f.c]);
            bound[i] = max(bound[i], envelope);
        }
    }
    return 0;
}
// This is the componentwise least outward displacement field majorizing the
// lower bounds with |d_i-d_j| <= L*r_ij. The in-sheet distance r increases
// slowly through thickness, coupling opposite laminae. It penalizes added
// membrane shear and avoids independent motion of the outer and deep sheets.
static void regularize(uint n) {
    for (uint i = 0; i < n; i++) {
        double v = bound[i];
        for (uint j = 0; j < n; j++)
            v = max(v, bound[j] - maxSlope * metric[i * n + j]);
        correction[i] = min(v, displacementCap[i]);
    }
}
// Unit-axis/angle represents the right hip's relative femoral rotation.
// Geometry in p is expressed in the half-rotated frame, where -X is outward.
// Translation and pelvis orientation are common rigid transforms applied later.
int contact_solve(double ax, double ay, double az, double angle) {
    uint n = shape[0], b = shape[1], nb = shape[2], no = shape[3];
    if (!n)
        return -6;
    Vec3 axis = {ax, ay, az};
    double sa = sin(angle / 1024.), ca = cos(angle / 1024.);
    sinLut[0] = sin(-.5 * angle);
    cosLut[0] = cos(-.5 * angle);
    for (uint i = 1; i <= 1024; i++) {
        sinLut[i] = sinLut[i - 1] * ca + cosLut[i - 1] * sa;
        cosLut[i] = cosLut[i - 1] * ca - sinLut[i - 1] * sa;
    }
    for (uint i = 0; i < n; i++) {
        const float *v = rest + 4 * i;
        double fw = v[3] * 1024;
        uint k = (uint)fw;
        double h = fw - k;
        if (k >= 1024) {
            k = 1023;
            h = 1;
        }
        p[i] = rotate((Vec3){v[0], v[1], v[2]}, axis, sinLut[k] + h * (sinLut[k + 1] - sinLut[k]),
                      cosLut[k] + h * (cosLut[k + 1] - cosLut[k]));
    }
    for (uint i = 0; i < 32; i++)
        metrics[i] = 0;
    for (uint i = 0; i < b; i++)
        bound[i] = 0;
    contactCount = 0;
    double ymax = -1e12, zmax = -1e12, xmax = -1e12;
    y0 = 1e12;
    z0 = 1e12;
    for (uint i = 0; i < b; i++) {
        y0 = min(y0, p[i].y);
        z0 = min(z0, p[i].z);
        xmax = max(xmax, p[i].x);
        ymax = max(ymax, p[i].y);
        zmax = max(zmax, p[i].z);
    }
    sy = GRID / max(1e-9, ymax - y0);
    sz = GRID / max(1e-9, zmax - z0);
    for (int i = 0; i < GRID * GRID; i++)
        head[i] = -1;
    int used = 0;
    for (uint i = 0; i < nb; i++) {
        const uint *t = tris + 3 * i;
        Vec3 a = p[t[0]], q = p[t[1]], c = p[t[2]];
        BandFace f = {t[0],
                      t[1],
                      t[2],
                      min(a.y, min(q.y, c.y)),
                      max(a.y, max(q.y, c.y)),
                      min(a.z, min(q.z, c.z)),
                      max(a.z, max(q.z, c.z)),
                      max(a.x, max(q.x, c.x))};
        faces[i] = f;
        seen[i] = -1;
        for (int y = bin((f.ymin - y0) * sy); y <= bin((f.ymax - y0) * sy); y++)
            for (int z = bin((f.zmin - z0) * sz); z <= bin((f.zmax - z0) * sz); z++) {
                if (used >= BIN_MAX)
                    return -7;
                int cell = y * GRID + z;
                binned[used] = i;
                next[used] = head[cell];
                head[cell] = used++;
            }
    }
    for (uint s = 0; s < no; s++) {
        const uint *t = tris + 3 * (s + nb);
        Vec3 a = p[t[0]], q = p[t[1]], c = p[t[2]];
        double xmin = min(a.x, min(q.x, c.x)) - clearance;
        if (xmin >= xmax)
            continue;
        double yl = min(a.y, min(q.y, c.y)), yh = max(a.y, max(q.y, c.y)),
               zl = min(a.z, min(q.z, c.z)), zh = max(a.z, max(q.z, c.z));
        if (yl > ymax || yh < y0 || zl > zmax || zh < z0)
            continue;
        double det = edge(a, q, c.y, c.z);
        if (det < 1e-15 && det > -1e-15)
            continue;
        double depthSlopeY = ((q.x - a.x) * (c.z - a.z) - (c.x - a.x) * (q.z - a.z)) / det;
        double depthSlopeZ = ((q.y - a.y) * (c.x - a.x) - (c.y - a.y) * (q.x - a.x)) / det;
        for (int y = bin((yl - y0) * sy); y <= bin((yh - y0) * sy); y++)
            for (int z = bin((zl - z0) * sz); z <= bin((zh - z0) * sz); z++)
                for (int h = head[y * GRID + z]; h >= 0; h = next[h]) {
                    uint id = binned[h];
                    if (seen[id] == (int)s)
                        continue;
                    seen[id] = s;
                    BandFace f = faces[id];
                    if (!(faceMask[id] & obstacleTag[s]) || f.xmax < xmin || f.ymin > yh ||
                        f.ymax < yl || f.zmin > zh || f.zmax < zl)
                        continue;
                    int status = contact_pair(id, a, q, c, det > 0 ? 1 : -1, depthSlopeY,
                                              depthSlopeZ, obstacleTag[s]);
                    if (status)
                        return status;
                }
    }
    for (uint i = 0; i < b; i++)
        bound[i] = min(bound[i], displacementCap[i]);
    regularize(b);
    for (uint i = 0; i < b; i++)
        add[i] = 0;
    for (uint k = 0; k < contactCount; k++) {
        Contact t = contacts[k];
        BandFace f = faces[t.face];
        double d = t.gap - t.a * correction[f.a] - t.b * correction[f.b] -
                   (1. - t.a - t.b) * correction[f.c];
        if (d > 0) {
            // A constant increment on these three vertices bounds the affine
            // violation at every extreme point of the clipped polygon.
            uint idx[3] = {f.a, f.b, f.c};
            double lambda[3] = {t.a, t.b, 1. - t.a - t.b};
            double h[3], maximum = 0.;
            for (uint j = 0; j < 3; j++) {
                h[j] = max(0., displacementCap[idx[j]] - correction[idx[j]]);
                maximum += lambda[j] * h[j];
            }
            // Find the smallest common increment before clipping at each pin.
            // Infeasible part is reported separately as an attachment residual.
            double step = d, need = min(d, maximum);
            for (uint j = 0; j < 3; j++) {
                double active = 0., gain = 0.;
                for (uint k = 0; k < 3; k++) {
                    gain += lambda[k] * min(h[k], step);
                    if (h[k] > step)
                        active += lambda[k];
                }
                if (gain >= need - 1e-13 || active < 1e-10)
                    break;
                step += (need - gain) / active;
            }
            for (uint j = 0; j < 3; j++)
                add[idx[j]] = max(add[idx[j]], min(h[j], step));
        }
    }
    for (uint i = 0; i < b; i++)
        bound[i] = min(displacementCap[i], correction[i] + add[i]);
    regularize(b);
    for (uint k = 0; k < contactCount; k++) {
        Contact t = contacts[k];
        BandFace f = faces[t.face];
        double capAtPoint = t.a * displacementCap[f.a] + t.b * displacementCap[f.b] +
                            (1. - t.a - t.b) * displacementCap[f.c];
        double residual = t.gap - t.a * correction[f.a] - t.b * correction[f.b] -
                          (1. - t.a - t.b) * correction[f.c];
        metrics[1] = max(metrics[1], residual - max(0., t.gap - capAtPoint));
        metrics[3] = max(metrics[3], residual);
        uint layer = (uint)__builtin_ctz(t.tag);
        metrics[4 + layer] = max(metrics[4 + layer], residual);
        metrics[12 + layer] = max(metrics[12 + layer], t.gap);
        if (residual >= metrics[3])
            metrics[20] = t.face;
        double originalResidual = residual + (t.rawGap - t.gap);
        metrics[21] = max(metrics[21], originalResidual);
        metrics[22 + layer] = max(metrics[22 + layer], originalResidual);
    }
    double sh = sin(.5 * angle), ch = cos(.5 * angle);
    for (uint i = 0; i < b; i++) {
        Vec3 x = p[i];
        x.x -= correction[i];
        output[i] = rotate(x, axis, sh, ch);
    }
    metrics[2] = contactCount;
    return 0;
}
