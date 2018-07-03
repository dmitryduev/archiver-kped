import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree
import time


def build_quad_hashes(positions):
    """

    :param positions:
    :return:
    """
    hashes = []
    hashed_quads = []

    # iterate over all len(positions) choose 4 combinations of stars:
    for quad in combinations(positions, 4):
        # print('quad:', quad)
        # matrix of pairwise distances in the quad:
        distances = squareform(pdist(quad))
        # print('distances:', distances)
        max_distance_index = np.unravel_index(np.argmax(distances), distances.shape)
        # print('max_distance_index:', max_distance_index)
        # # pairs themselves:
        # pairs = list(combinations(quad, 2))
        # print('pairs', pairs)

        # get the the far-most points:
        # AB = pairs[int(np.argmax(distances))]
        AB = [quad[max_distance_index[0]], quad[max_distance_index[1]]]
        # print('AB', AB)

        # compute projections:
        Ax, Ay, Bx, By = AB[0][0], AB[0][1], AB[1][0], AB[1][1]
        ABx = Bx - Ax
        ABy = By - Ay
        scale = (ABx * ABx) + (ABy * ABy)
        invscale = 1.0 / scale
        costheta = (ABy + ABx) * invscale
        sintheta = (ABy - ABx) * invscale

        # build hash:
        hash = []

        CD = (_p for _p in quad if _p not in AB)
        # print(CD)

        CDxy = []
        for D in CD:
            Dx, Dy = D[0], D[1]
            ADx = Dx - Ax
            ADy = Dy - Ay
            x = ADx * costheta + ADy * sintheta
            y = -ADx * sintheta + ADy * costheta
            # print(x, y)
            CDxy.append((x, y))
        # sort by x-projection value so that Cx < Dx
        # CDxy = sorted(CDxy)

        # add to the kd-tree if Cx + Dx < 1:
        if CDxy[0][0] + CDxy[1][0] < 1:
            hashes.append(CDxy[0] + CDxy[1])
            hashed_quads.append(quad)

    return hashes, hashed_quads


if __name__ == '__main__':

    # p = [(1004.2994, 502.0677), (426.6062, 135.6895), (94.5041, 331.7511), (539.3708, 287.2571), (580.3152, 480.9749)]
    # # print(list(combinations(p, 4)))
    #
    # hashes, hashed_quads = build_quad_hashes(p)
    # print('hashes:\n', hashes)
    # print('hashed quads:\n', hashed_quads)
    # print('number of valid hashes:', len(hashes))

    # detected:
    with open('/Users/dmitryduev/_caltech/python/archiver-kped/pix_det.txt', 'r') as f:
        f_lines = f.readlines()
    pix = [(float(_l.split()[0][1:-1]), float(_l.split()[1][0:-1])) for _l in f_lines]
    # print(pix)
    tic = time.time()
    hashes_det, hashed_quads_det = build_quad_hashes(pix)
    print(f'Building hashes for {len(pix)} detected sources took {time.time()-tic} seconds.')
    # print('hashes:\n', hashes_det)
    # print('hashed quads:\n', hashed_quads_det)
    print('number of valid hashes for detected sources:', len(hashes_det))

    # reference:
    with open('/Users/dmitryduev/_caltech/python/archiver-kped/pix_ref.txt', 'r') as f:
        f_lines = f.readlines()
    pix = [(float(_l.split()[0][1:-1]), float(_l.split()[1][0:-1])) for _l in f_lines]
    # print(pix)
    tic = time.time()
    hashes_ref, hashed_quads_ref = build_quad_hashes(pix[:70])
    print(f'Building hashes for {len(hashes_ref)} reference sources took {time.time()-tic} seconds.')
    # print('hashes:\n', hashes_ref)
    # print('hashed quads:\n', hashed_quads_ref)
    print('number of valid hashes for reference sources:', len(hashes_ref))

    # build tree for reference hashes:

    tree = KDTree(hashes_ref)
    # get the nearest + distance to it
    # dist, ind = tree.query(hashes_det, k=1)
    # print(np.min())
    results = []
    for ih, h in enumerate(hashes_det):
        dist, ind = tree.query([h], k=1)
        print(ih, dist, ind)
        results.append([ih, dist[0][0], ind[0][0]])

    results = np.array(results)
    print(np.min(results[:, 1]), np.median(results[:, 1]), np.max(results[:, 1]))
    print(sorted(results[:, 1])[:10], sorted(results[:, 1])[-10:])
