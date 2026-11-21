import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np


def digest(data):
    return hashlib.blake2b(data, digest_size=12).hexdigest()


def summarize(path):
    data = path.read_bytes()
    length = struct.unpack_from('<I', data, 12)[0]
    doc = json.loads(data[20:20 + length])
    payload = memoryview(data)[28 + length:]
    views = doc.pop('bufferViews')
    doc.pop('buffers')
    types = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
    formats = {5121: 'u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}

    def values(view, offset, dtype, count):
        v = views[view]
        assert 'byteStride' not in v
        return np.frombuffer(payload, dtype, count, v.get('byteOffset', 0) + offset)

    columns = []
    for a in doc['accessors']:
        kind = a['componentType']
        dtype, width = formats[kind], types[a['type']]
        size = a['count'] * width
        view, offset = a.pop('bufferView', None), a.pop('byteOffset', 0)
        array = (np.zeros(size, dtype) if view is None else values(view, offset, dtype, size).copy()).reshape(-1, width)
        sparse = a.pop('sparse', None)
        if sparse:
            s, v, count = sparse['indices'], sparse['values'], sparse['count']
            indices = values(s['bufferView'], s.get('byteOffset', 0), formats[s['componentType']], count)
            array[indices] = values(v['bufferView'], v.get('byteOffset', 0), dtype, count * width).reshape(-1, width)
        a.pop('min', None)
        a.pop('max', None)
        if kind == 5126:
            assert np.isfinite(array).all()
            scaled = array.astype(np.float64) * 10000
            # Equal bins on any grid bound every scalar difference below one bin.
            columns.append([digest(np.floor(scaled + i / 12).astype('<i4')) for i in range(12)])
        else:
            columns.append([digest(array)])
    # Expose the numerical animation and vertex arrays above. All other structure must agree.
    return {'structure': digest(json.dumps(doc, sort_keys=True).encode()), 'columns': columns}


if __name__ == '__main__':
    candidate = summarize(Path(sys.argv[1]))
    if len(sys.argv) == 2:
        print(json.dumps(candidate, separators=(',', ':')))
    else:
        import gzip

        with gzip.open(sys.argv[2], 'rt') as reference_file:
            reference = json.load(reference_file)
        assert candidate['structure'] == reference['structure'], 'GLB structure differs'
        pairs = list(zip(candidate['columns'], reference['columns'], strict=True))
        mismatches = [i for i, (a, b) in enumerate(pairs) if not any(x == y for x, y in zip(a, b, strict=True))]
        print(f'{len(pairs) - len(mismatches)}/{len(pairs)} evaluated GLB accessors agree')
        assert not mismatches, f'GLB accessors with differences above comparison resolution: {mismatches}'
