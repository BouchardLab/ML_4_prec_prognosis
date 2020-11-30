import numpy as np

from activ.readfile import TrackTBIFile

def test_write():
    path = 'test_tbi_file.h5'

    N = 250
    p = 235
    q = 175

    bm = np.random.random((N, p))
    oc = np.random.random((N, q))

    bm_feats = np.array(['bm_%d' % i for i in range(p)])
    oc_feats = np.array(['oc_%d' % i for i in range(q)])
    pids = np.array(['patient_%d' % i for i in range(N)])

    TrackTBIFile.write(path, bm, oc, bm_feats, oc_feats, pids)

    bm_feat_type = np.array(['bm_type_%d' % (i%5) for i in range(p)])
    oc_feat_type = np.array(['oc_type_%d' % (i%5) for i in range(q)])

    TrackTBIFile.write_feat_types(path, {'nmf': bm_feat_type}, {'nmf': oc_feat_type})

    bm_k = 12
    oc_k = 8
    bm_w = np.random.random((N, bm_k))
    oc_w = np.random.random((N, oc_k))

    bm_h = np.random.random((bm_k, p))
    oc_h = np.random.random((oc_k, q))

    TrackTBIFile.write_nmf(path, bm_w, oc_w, bm_h, oc_h)

    cca_k = 6
    bm_cv = np.random.random((N, cca_k))
    oc_cv = np.random.random((N, cca_k))

    bm_l = np.random.random((p, cca_k))
    oc_l = np.random.random((q, cca_k))

    TrackTBIFile.write_cca(path, bm_cv, oc_cv, bm_l, oc_l)

    tbfile = TrackTBIFile(path)
