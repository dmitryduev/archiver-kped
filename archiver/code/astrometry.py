import inspect

import sewpy
import multiprocessing
import numpy as np
import os
import datetime
from skimage import exposure, img_as_float
from copy import deepcopy
from scipy.optimize import leastsq
from astropy.io import fits
from astropy import wcs
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve_fft
from collections import OrderedDict
from numba import jit
from penquins import Kowalski
import json
import matplotlib
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree
import time

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import aplpy
import image_registration
from skimage.feature import register_translation

np.set_printoptions(16)


# load secrets:
with open('/Users/dmitryduev/_caltech/python/archiver-kped/archiver/secrets.json') as sjson:
    secrets = json.load(sjson)


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


def generate_image(xy, mag, xy_ast=None, mag_ast=None, exp=None, nx=2048, ny=2048, psf=None):
    """

    :param xy:
    :param mag:
    :param xy_ast:
    :param mag_ast:
    :param exp: exposure in seconds to 'normalize' streak
    :param nx:
    :param ny:
    :param psf:
    :return:
    """

    if isinstance(xy, list):
        xy = np.array(xy)
    if isinstance(mag, list):
        mag = np.array(mag)

    image = np.zeros((ny, nx))

    # let us assume that a 6 mag star would have a flux of 10^9 counts
    flux_0 = 1e9
    # scale other stars wrt that:
    flux = flux_0 * 10 ** (0.4 * (6 - mag))
    # print(flux)

    # add stars to image
    for k, (i, j) in enumerate(xy):
        if i < nx and j < ny:
            image[int(j), int(i)] = flux[k]

    if exp is None:
        exp = 1.0

    if psf is None:
        # Convolve with a gaussian
        image = gaussian_filter(image, 7 * nx / 2e3)
    else:
        # convolve with a (model) psf
        # fftn, ifftn = image_registration.fft_tools.fast_ffts.get_ffts(nthreads=4, use_numpy_fft=False)
        # image = convolve_fft(image, psf, fftn=fftn, ifftn=ifftn)
        image = convolve_fft(image, psf)

    return image


def make_image(target, window_size, _model_psf, pix_stars, mag_stars, num_pix=1024, fov_size=264):
    """

    :return:
    """
    ''' set up WCS '''
    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)
    w._naxis1 = int(num_pix * (window_size[0] * 180.0 / np.pi * 3600) / fov_size)
    w._naxis2 = int(num_pix * (window_size[1] * 180.0 / np.pi * 3600) / fov_size)
    w.naxis1 = w._naxis1
    w.naxis2 = w._naxis2

    if w.naxis1 > 20000 or w.naxis2 > 20000:
        print('image too big to plot')
        return

    # Set up a tangential projection
    w.wcs.equinox = 2000.0
    # position of the tangential point on the detector [pix]
    w.wcs.crpix = np.array([w.naxis1 // 2, w.naxis2 // 2])
    # sky coordinates of the tangential point
    w.wcs.crval = [target.ra.deg, target.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    ''' create a [fake] simulated image '''
    # tic = _time()
    sim_image = generate_image(xy=pix_stars, mag=mag_stars, nx=w.naxis1, ny=w.naxis2, psf=_model_psf)

    return sim_image


def plot_field(target, window_size, _model_psf, grid_stars=None,
               pix_stars=None, mag_stars=None, a_priori_mapping=False,
               num_pix=1024, fov_size=264, _highlight_brighter_than_mag=None,
               _display_magnitude_labels=False, scale_bar=False, scale_bar_size=20,
               _display_plot=False, _save_plot=False, path='./', name='field'):
    """

    :return:
    """
    ''' set up WCS '''
    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)
    w._naxis1 = int(num_pix * (window_size[0] * 180.0 / np.pi * 3600) / fov_size)
    w._naxis2 = int(num_pix * (window_size[1] * 180.0 / np.pi * 3600) / fov_size)

    # make even:
    if w._naxis1 % 2:
        w._naxis1 += 1
    if w._naxis2 % 2:
        w._naxis2 += 1

    w.naxis1 = w._naxis1
    w.naxis2 = w._naxis2
    print(w.naxis1, w.naxis2)

    if w.naxis1 > 20000 or w.naxis2 > 20000:
        print('image too big to plot')
        return

    # Set up a tangential projection
    w.wcs.equinox = 2000.0
    # position of the tangential point on the detector [pix]
    w.wcs.crpix = np.array([w.naxis1 // 2, w.naxis2 // 2])
    # sky coordinates of the tangential point
    w.wcs.crval = [target.ra.deg, target.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # linear mapping detector :-> focal plane [deg/pix]
    # with RA inverted to correspond to previews
    if a_priori_mapping:
        # a priori mapping derived from first principles:
        w.wcs.cd = np.array([[-(fov_size / num_pix) / 3600. * 0.999, (fov_size / num_pix) / 3600. * 0.002],
                             [(fov_size / num_pix) / 3600. * 0.002, (fov_size / num_pix) / 3600. * 0.999]])
    else:
        w.wcs.cd = np.array([[-7.128e-05, -6.729e-07],
                             [ 5.967e-07,  7.121e-05]])

    print(w.wcs.cd)

    # set up quadratic distortions [xy->uv and uv->xy]
    # w.sip = wcs.Sip(a=np.array([-1.7628536101583434e-06, 5.2721963537675933e-08, -1.2395119995283236e-06]),
    #                 b=np.array([2.5686775443756446e-05, -6.4405711579912514e-06, 3.6239787339845234e-05]),
    #                 ap=np.array([-7.8730574242135546e-05, 1.6739809945514789e-06,
    #                              -1.9638469711488499e-08, 5.6147572815095856e-06,
    #                              1.1562096854108367e-06]),
    #                 bp=np.array([1.6917947345178044e-03,
    #                              -2.6065393907218176e-05, 6.4954883952398105e-06,
    #                              -4.5911421583810606e-04, -3.5974854928856988e-05]),
    #                 crpix=w.wcs.crpix)

    ''' create a [fake] simulated image '''
    # apply linear transformation only:
    if pix_stars is None:
        pix_stars = np.array(w.wcs_world2pix(grid_stars['RA'], grid_stars['Dec'], 0)).T
    # apply linear + SIP:
    # pix_stars = np.array(w.all_world2pix(grid_stars['RA'], grid_stars['Dec'], 0)).T
    if mag_stars is None:
        mag_stars = np.array(grid_stars['mag'])
    # print(pix_stars)
    # print(mag_stars)

    # tic = _time()
    sim_image = generate_image(xy=pix_stars, mag=mag_stars,
                               nx=w.naxis1, ny=w.naxis2, psf=_model_psf)
    # print(_time() - tic)

    # tic = _time()
    # convert simulated image to fits hdu:
    hdu = fits.PrimaryHDU(sim_image, header=w.to_header())

    ''' plot! '''
    # plt.close('all')
    # plot empty grid defined by wcs:
    # fig = aplpy.FITSFigure(w)
    # plot fake image:
    fig = aplpy.FITSFigure(hdu)

    # fig.set_theme('publication')

    fig.add_grid()

    fig.grid.show()
    fig.grid.set_color('gray')
    fig.grid.set_alpha(0.8)

    ''' display field '''
    # fig.show_colorscale(cmap='viridis')
    fig.show_colorscale(cmap='magma', stretch='sqrt')
    # fig.show_grayscale()
    # fig.show_markers(grid_stars[cat]['_RAJ2000'], grid_stars[cat]['_DEJ2000'],
    #                  layer='marker_set_1', edgecolor='white',
    #                  facecolor='white', marker='o', s=30, alpha=0.7)

    # highlight stars bright enough to serve as tip-tilt guide stars:
    if _highlight_brighter_than_mag is not None and grid_stars is not None:
        mask_bright = mag_stars <= float(_highlight_brighter_than_mag)
        if np.max(mask_bright) == 1:
            fig.show_markers(grid_stars[mask_bright]['RA'], grid_stars[mask_bright]['Dec'],
                             layer='marker_set_2', edgecolor=plt.cm.Oranges(0.9),
                             facecolor=plt.cm.Oranges(0.8), marker='+', s=50, alpha=0.9, linewidths=1)

    # show labels with magnitudes
    if _display_magnitude_labels and grid_stars is not None:
        for star in grid_stars:
            fig.add_label(star['RA'], star['Dec'], '{:.1f}'.format(star['mag']),
                          color=plt.cm.Oranges(0.4), horizontalalignment='right')

    # add scale bar
    if scale_bar:
        fig.add_scalebar(length=scale_bar_size * u.arcsecond)
        fig.scalebar.set_alpha(0.7)
        fig.scalebar.set_color('white')
        fig.scalebar.set_label('{:d}\"'.format(scale_bar_size))

    # remove frame
    fig.frame.set_linewidth(0)
    # print(_time() - tic)

    if _display_plot:
        plt.show()

    if _save_plot:
        if not os.path.exists(path):
            os.makedirs(path)
        fig.save(os.path.join(path, '{:s}.png'.format(name)))
        fig.close()

    return pix_stars, mag_stars


def get_fits_header(fits_file):
    """
        Get fits-file header
    :param fits_file:
    :return:
    """
    # read fits:
    with fits.open(os.path.join(fits_file)) as hdulist:
        # header(s):

        header_0 = OrderedDict()
        for _entry in hdulist[0].header.cards:
            header_0[_entry[0]] = _entry[1:]

        header_1 = OrderedDict()
        for _entry in hdulist[1].header.cards:
            header_1[_entry[0]] = _entry[1:]

    return header_0, header_1


def load_fits(fin, return_header=False):
    with fits.open(fin) as _f:
        _scidata = np.nan_to_num(_f[0].data)
    # _header = get_fits_header(fin) if return_header else None

    # return _scidata, _header
    return _scidata


def export_fits(path, _data, _header=None):
    """
        Save fits file overwriting if exists
    :param path:
    :param _data:
    :param _header:
    :return:
    """
    if _header is not None:
        hdu = fits.PrimaryHDU(_data, header=_header)
    else:
        hdu = fits.PrimaryHDU(_data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(path, overwrite=True)


def scale_image(image, correction='local'):
    """

    :param image:
    :param correction: 'local', 'log', or 'global'
    :return:
    """
    # scale image for beautification:
    scidata = deepcopy(image)
    norm = np.max(np.max(scidata))
    mask = scidata <= 0
    scidata[mask] = 0
    scidata = np.uint16(scidata / norm * 65535)

    # add more contrast to the image:
    if correction == 'log':
        return exposure.adjust_log(img_as_float(scidata / norm) + 1, 1)
    elif correction == 'global':
        p_1, p_2 = np.percentile(scidata, (5, 100))
        # p_1, p_2 = np.percentile(scidata, (1, 20))
        return exposure.rescale_intensity(scidata, in_range=(p_1, p_2))
    elif correction == 'local':
        # perform local histogram equalization instead:
        return exposure.equalize_adapthist(scidata, clip_limit=0.03)
    else:
        raise Exception('Contrast correction option not recognized')


def residual(p, y, x):
    """
        Simultaneously solve for RA_tan, DEC_tan, M, and 2nd-order distortion params
    :param p:
    :param y:
    :param x:
    :return:
    """
    # convert (ra, dec)s to 3d
    r = np.vstack((np.cos(x[:, 0] * np.pi / 180.0) * np.cos(x[:, 1] * np.pi / 180.0),
                   np.sin(x[:, 0] * np.pi / 180.0) * np.cos(x[:, 1] * np.pi / 180.0),
                   np.sin(x[:, 1] * np.pi / 180.0))).T
    # print(r.shape)
    # print(r)

    # the same for the tangent point
    t = np.array((np.cos(p[0] * np.pi / 180.0) * np.cos(p[1] * np.pi / 180.0),
                  np.sin(p[0] * np.pi / 180.0) * np.cos(p[1] * np.pi / 180.0),
                  np.sin(p[1] * np.pi / 180.0)))
    # print(t)

    k = np.array([0, 0, 1])

    # u,v projections
    u = np.cross(t, k) / np.linalg.norm(np.cross(t, k))
    v = np.cross(u, t)
    # print(u, v)

    R = r / (np.dot(r, t)[:, None])

    # print(R)

    # native tangent-plane coordinates:
    UV = 180.0 / np.pi * np.vstack((np.dot(R, u), np.dot(R, v))).T

    # print(UV)

    M_m1 = np.matrix([[p[2], p[3]], [p[4], p[5]]])
    M = np.linalg.pinv(M_m1)
    # print(M)

    x_tan = M * np.array([[p[0]], [p[1]]])
    # x_tan = np.array([[0], [0]])
    # print(x_tan)

    ksieta = (M_m1 * UV.T).T
    y_C = []

    for i in range(ksieta.shape[0]):
        ksi_i = ksieta[i, 0]
        eta_i = ksieta[i, 1]
        x_i = ksi_i + x_tan[0]
        y_i = eta_i + x_tan[1]
        y_C.append([x_i, y_i])
    y_C = np.squeeze(np.array(y_C))

    # return np.linalg.norm(y.T - (M_m1 * UV.T + x_tan), axis=0)
    return np.linalg.norm(y.T - y_C.T, axis=0)


def residual_quadratic(p, y, x):
    """
        Simultaneously solve for RA_tan, DEC_tan, M, and 2nd-order distortion params
    :param p:
    :param y:
    :param x:
    :return:
    """
    # convert (ra, dec)s to 3d
    r = np.vstack((np.cos(x[:, 0] * np.pi / 180.0) * np.cos(x[:, 1] * np.pi / 180.0),
                   np.sin(x[:, 0] * np.pi / 180.0) * np.cos(x[:, 1] * np.pi / 180.0),
                   np.sin(x[:, 1] * np.pi / 180.0))).T
    # print(r.shape)
    # print(r)

    # the same for the tangent point
    t = np.array((np.cos(p[0] * np.pi / 180.0) * np.cos(p[1] * np.pi / 180.0),
                  np.sin(p[0] * np.pi / 180.0) * np.cos(p[1] * np.pi / 180.0),
                  np.sin(p[1] * np.pi / 180.0)))
    # print(t)

    k = np.array([0, 0, 1])

    # u,v projections
    u = np.cross(t, k) / np.linalg.norm(np.cross(t, k))
    v = np.cross(u, t)
    # print(u, v)

    R = r / (np.dot(r, t)[:, None])

    # print(R)

    # native tangent-plane coordinates:
    UV = 180.0 / np.pi * np.vstack((np.dot(R, u), np.dot(R, v))).T

    # print(UV)

    M_m1 = np.matrix([[p[2], p[3]], [p[4], p[5]]])
    M = np.linalg.pinv(M_m1)
    # print(M)

    x_tan = M * np.array([[p[0]], [p[1]]])
    # x_tan = np.array([[0], [0]])
    # print(x_tan)

    ksieta = (M_m1 * UV.T).T
    y_C = []
    A_01, A_02, A_11, A_10, A_20, B_01, B_02, B_11, B_10, B_20 = p[6:]

    for i in range(ksieta.shape[0]):
        ksi_i = ksieta[i, 0]
        eta_i = ksieta[i, 1]
        x_i = ksi_i + x_tan[0] + A_01 * eta_i + A_02 * eta_i ** 2 + A_11 * ksi_i * eta_i \
              + A_10 * ksi_i + A_20 * ksi_i ** 2
        y_i = eta_i + x_tan[1] + B_01 * eta_i + B_02 * eta_i ** 2 + B_11 * ksi_i * eta_i \
              + B_10 * ksi_i + B_20 * ksi_i ** 2
        y_C.append([x_i, y_i])
    y_C = np.squeeze(np.array(y_C))

    # return np.linalg.norm(y.T - (M_m1 * UV.T + x_tan), axis=0)
    return np.linalg.norm(y.T - y_C.T, axis=0)


def compute_detector_position(p, x, quadratic=False):
    # convert (ra, dec)s to 3d
    r = np.vstack((np.cos(x[:, 0] * np.pi / 180.0) * np.cos(x[:, 1] * np.pi / 180.0),
                   np.sin(x[:, 0] * np.pi / 180.0) * np.cos(x[:, 1] * np.pi / 180.0),
                   np.sin(x[:, 1] * np.pi / 180.0))).T
    # print(r.shape)
    # print(r)

    # the same for the tangent point
    t = np.array((np.cos(p[0] * np.pi / 180.0) * np.cos(p[1] * np.pi / 180.0),
                  np.sin(p[0] * np.pi / 180.0) * np.cos(p[1] * np.pi / 180.0),
                  np.sin(p[1] * np.pi / 180.0)))
    # print(t)

    k = np.array([0, 0, 1])

    # u,v projections
    u = np.cross(t, k) / np.linalg.norm(np.cross(t, k))
    v = np.cross(u, t)
    # print(u, v)

    R = r / (np.dot(r, t)[:, None])

    # print(R)

    # native tangent-plane coordinates:
    UV = 180.0 / np.pi * np.vstack((np.dot(R, u), np.dot(R, v))).T

    # print('UV: ', UV)

    M_m1 = np.matrix([[p[2], p[3]], [p[4], p[5]]])
    M = np.linalg.pinv(M_m1)
    # print(M)

    x_tan = M * np.array([[p[0]], [p[1]]])
    # x_tan = np.array([[0], [0]])

    ksieta = (M_m1 * UV.T).T
    y_C = []
    if quadratic:
        A_01, A_02, A_11, A_10, A_20, B_01, B_02, B_11, B_10, B_20 = p[6:]

    for i in range(ksieta.shape[0]):
        ksi_i = ksieta[i, 0]
        eta_i = ksieta[i, 1]
        if quadratic:
            x_i = ksi_i + x_tan[0] + A_01 * eta_i + A_02 * eta_i ** 2 + A_11 * ksi_i * eta_i \
                  + A_10 * ksi_i + A_20 * ksi_i ** 2
            y_i = eta_i + x_tan[1] + B_01 * eta_i + B_02 * eta_i ** 2 + B_11 * ksi_i * eta_i \
                  + B_10 * ksi_i + B_20 * ksi_i ** 2
        else:
            x_i = ksi_i + x_tan[0]
            y_i = eta_i + x_tan[1]

        y_C.append([x_i, y_i])

    y_C = np.squeeze(np.array(y_C))

    return y_C.T


def fit_bootstrap(_residual, p0, datax, datay, yerr_systematic=0.0, n_samp=100, _scaling=None, Nsigma=1.):
    # Fit first time
    _p = leastsq(_residual, p0, args=(datax, datay), full_output=True, ftol=1.49012e-13, xtol=1.49012e-13)

    pfit, perr = _p[0], _p[1]

    # Get the stdev of the residuals
    residuals = _residual(pfit, datax, datay)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res ** 2 + yerr_systematic ** 2)

    # n_samp random data sets are generated and fitted
    ps = []
    # print('lala')
    # print(datay)
    for ii in range(n_samp):
        randomDelta = np.random.normal(0., sigma_err_total, size=datax.shape)
        # print(ii)
        randomdataX = datax + randomDelta
        # print(randomDelta)
        # raw_input()

        _p = leastsq(_residual, p0, args=(randomdataX, datay), full_output=True,
                     ftol=1.49012e-13, xtol=1.49012e-13, diag=_scaling)
        randomfit, randomcov = _p[0], _p[1]

        ps.append(randomfit)

    ps = np.array(ps)
    mean_pfit = np.mean(ps, 0)

    # You can choose the confidence interval that you want for your
    # parameter estimates:
    # 1sigma corresponds to 68.3% confidence interval
    # 2sigma corresponds to 95.44% confidence interval

    err_pfit = Nsigma * np.std(ps, 0)

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit

    return pfit_bootstrap, perr_bootstrap


def get_config(_config_file):
    """
        Load config JSON file
    """
    ''' script absolute location '''
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

    if _config_file[0] not in ('/', '~'):
        if os.path.isfile(os.path.join(abs_path, _config_file)):
            config_path = os.path.join(abs_path, _config_file)
        else:
            raise IOError('Failed to find config file')
    else:
        if os.path.isfile(_config_file):
            config_path = _config_file
        else:
            raise IOError('Failed to find config file')

    with open(config_path) as cjson:
        config_data = json.load(cjson)
        # config must not be empty:
        if len(config_data) > 0:
            return config_data
        else:
            raise Exception('Failed to load config file')


def parse_observed_dat(_fin):
    with open(_fin) as _f:
        _f_lines = _f.readlines()

    _sources = dict()
    for _l in _f_lines:
        _tmp = _l.split()
        _sources[_tmp[0]] = list(map(float, _tmp[1:]))

    return _sources


def query_reference_catalog(_star_sc, _fov_size_ref_arcsec, retries=3):

    for ir in range(retries):
        try:
            print(f'Querying Kowalski, attempt {ir+1}')
            # query Kowalski for Gaia stars:
            with Kowalski(username=secrets['kowalski']['user'], password=secrets['kowalski']['password']) as kowalski:
                # if False:
                q = {"query_type": "cone_search",
                     "object_coordinates": {"radec": f"[({_star_sc.ra.deg}, {_star_sc.dec.deg})]",
                                            "cone_search_radius": str(_fov_size_ref_arcsec / 2),
                                            "cone_search_unit": "arcsec"},
                     "catalogs": {"Gaia_DR2": {"filter": {},
                                               "projection": {"_id": 0, "source_id": 1,
                                                              "ra": 1, "dec": 1, "ra_error": 1, "dec_error": 1,
                                                              "phot_g_mean_mag": 1}}}
                     }
                r = kowalski.query(query=q, timeout=10)
                key = list(r['result']['Gaia_DR2'].keys())[0]
                fov_stars = r['result']['Gaia_DR2'][key]

                return fov_stars
        except Exception as _e:
            print(_e)
            continue

    return None


def astrometry(_obs, _config):
    """

    :return:
    """

    _tmp = _obs.split('_')
    _sou_name = '_'.join(_tmp[0:-5])
    # mode num:
    _mode = _tmp[-5:-4][0]
    # code of the filter used:
    _filt = _tmp[-4:-3][0]
    # date and time of obs:
    _date = datetime.datetime.strptime(_tmp[-3] + _tmp[-2], '%Y%m%d%H%M%S.%f').strftime('%Y%m%d')

    # path to archive:
    _path_archive = os.path.join(_config['path']['path_archive'], _date)
    _path_registered = os.path.join(_path_archive, _obs, 'registration')
    # path to output:
    _path_out = os.path.join(_path_archive, _obs, 'astrometry')

    if not (os.path.exists(_path_out)):
        os.makedirs(_path_out)

    _fits_in = f'{_obs}_registered_sum.fits'

    # set up sextractor:
    # use master flat field image for filter as weight map:
    weight_image = os.path.join(_config['path']['path_archive'], 'calib', f'flat_{_filt}.fits')
    sex_config = _config['pipeline']['astrometry']['sextractor_settings']['config']
    sex_config['WEIGHT_IMAGE'] = weight_image

    sew = sewpy.SEW(params=_config['pipeline']['astrometry']['sextractor_settings']['params'],
                    config=sex_config,
                    sexpath=_config['pipeline']['astrometry']['sextractor_settings']['sexpath'])

    out = sew(os.path.join(_path_registered, _fits_in))
    # sort by raw flux
    out['table'].sort('FLUX_AUTO')
    # descending order: first is brightest
    out['table'].reverse()

    # remove vignetted stuff. use the g band master flat field image
    weight_image_g = os.path.join(_config['path']['path_archive'], 'calib', 'flat_g.fits')
    weights = load_fits(weight_image_g)
    rows_to_remove = []
    for ri, row in enumerate(out['table']):
        x, y = int(row['X_IMAGE']), int(row['Y_IMAGE'])
        # print(x, y, weights[x, y])
        if weights[x, y] < _config['pipeline']['astrometry']['vignetting_cutoff']:
            rows_to_remove.append(ri)

    out['table'].remove_rows(rows_to_remove)

    # detected sources:
    pix_det = np.vstack((out['table']['X_IMAGE'], out['table']['Y_IMAGE'])).T
    pix_det_err = np.vstack((out['table']['X2_IMAGE'], out['table']['Y2_IMAGE'], out['table']['XY_IMAGE'])).T
    mag_det = np.array(out['table']['FLUX_AUTO'])

    if _config['pipeline']['astrometry']['verbose']:
        print(out['table'])

    # save preview
    preview_img = load_fits(os.path.join(_path_registered, _fits_in))
    # print(preview_img.shape)
    # scale with local contrast optimization for preview:
    # preview_img = scale_image(preview_img, correction='local')
    # preview_img = scale_image(preview_img, correction='log')
    # preview_img = scale_image(preview_img, correction='global')

    plt.close('all')
    fig = plt.figure()
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot detected objects:
    for i, _ in enumerate(out['table']['XWIN_IMAGE']):
        ax.plot(out['table']['X_IMAGE'][i] - 1, out['table']['Y_IMAGE'][i] - 1,
                'o', markersize=out['table']['FWHM_IMAGE'][i] / 2,
                markeredgewidth=2.5, markerfacecolor='None', markeredgecolor=plt.cm.Greens(0.7),
                label='SExtracted')
        # ax.annotate(i, (out['table']['X_IMAGE'][i]+40, out['table']['Y_IMAGE'][i]+40),
        #             color=plt.cm.Blues(0.3), backgroundcolor='black')

    preview_img_zerod = preview_img
    preview_img_zerod[preview_img_zerod < np.median(np.median(preview_img_zerod)) * 0.5] = 0.01
    ax.imshow(np.sqrt(np.sqrt(preview_img_zerod)), cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    plt.grid(False)

    # save figure
    fname = '{:s}_registered_sex.png'.format(_obs)
    plt.savefig(os.path.join(_path_out, fname), dpi=300)

    ''' get stars from Gaia DR2 catalogue, create fake images, then cross correlate them '''
    # stars in the field (without mag cut-off):
    if _config['pipeline']['astrometry']['fov_center'] == 'telescope':
        # use whatever reported by telescope
        header = get_fits_header(os.path.join(_path_registered, f'{_obs}_registered_0000.fits'))[0]
        star_sc = SkyCoord(ra=header['TELRA'][0], dec=header['TELDEC'][0],
                           unit=(u.hourangle, u.deg), frame='icrs')
    elif _config['pipeline']['astrometry']['fov_center'] == 'starlist':
        # use Michael's starlist:
        observed = parse_observed_dat(os.path.join(_config['path']['path_archive'], 'observed.dat'))
        star_sc = SkyCoord(ra=observed[_sou_name][0], dec=observed[_sou_name][1], unit=(u.deg, u.deg), frame='icrs')

    if _config['pipeline']['astrometry']['verbose']:
        print('nominal FoV center', star_sc)
        # print(star_sc.ra.deg, star_sc.dec.deg)

    # search radius*2: " -> rad
    fov_size_ref_arcsec = _config['pipeline']['astrometry']['reference_win_size']
    fov_size_ref = fov_size_ref_arcsec * np.pi / 180.0 / 3600

    fov_stars = query_reference_catalog(_star_sc=star_sc, _fov_size_ref_arcsec=fov_size_ref_arcsec, retries=3)

    # convert to astropy table:
    fov_stars = np.array([[fov_star['source_id'], fov_star['ra'], fov_star['dec'],
                           fov_star['ra_error'], fov_star['dec_error'], fov_star['phot_g_mean_mag']]
                          for fov_star in fov_stars])
    fov_stars = Table(fov_stars, names=('source_id', 'RA', 'Dec', 'e_RA', 'e_Dec', 'mag'))
    if _config['pipeline']['astrometry']['verbose']:
        print(fov_stars)

    pix_ref, mag_ref = plot_field(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                                  grid_stars=fov_stars, num_pix=preview_img.shape[0], _highlight_brighter_than_mag=None,
                                  scale_bar=False, scale_bar_size=20, _display_plot=False, _save_plot=False,
                                  path='./', name='field')

    ''' detect shift '''
    fov_size_arcsec = _config['telescope']['KPNO_2.1m']['fov_x']
    fov_size_det = fov_size_arcsec * np.pi / 180.0 / 3600
    mag_det /= np.median(mag_det)
    mag_det = -2.5 * np.log10(mag_det)
    # add (pretty arbitrary) baseline from ref
    mag_det += np.median(mag_ref)

    naxis_det = int(preview_img.shape[0] * (fov_size_det * 180.0 / np.pi * 3600) / fov_size_arcsec)
    naxis_ref = int(preview_img.shape[0] * (fov_size_ref * 180.0 / np.pi * 3600) / fov_size_arcsec)
    if _config['pipeline']['astrometry']['verbose']:
        print(naxis_det, naxis_ref)
    # effectively shift detected positions to center of ref frame to reduce distortion effect
    pix_det_ref = pix_det + np.array([naxis_ref // 2, naxis_ref // 2]) - np.array([naxis_det // 2, naxis_det // 2])
    # pix_det_ref = pix_det
    detected = make_image(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                          pix_stars=pix_det_ref, mag_stars=mag_det, num_pix=preview_img.shape[0])
    reference = make_image(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                           pix_stars=pix_ref, mag_stars=mag_ref, num_pix=preview_img.shape[0])

    detected[detected == 0] = 0.0001
    detected[np.isnan(detected)] = 0.0001
    detected = np.log(detected)

    reference[reference == 0] = 0.0001
    reference[np.isnan(reference)] = 0.0001
    reference = np.log(reference)

    # register shift: pixel precision first
    shift, error, diffphase = register_translation(reference, detected, upsample_factor=1)
    if _config['pipeline']['astrometry']['verbose']:
        print('pixel precision offset:', shift, error)
    # shift, error, diffphase = register_translation(reference, detected, upsample_factor=2)
    # print('subpixel precision offset:', shift, error)

    # associate!
    matched = []
    mask_matched = []
    for si, s in enumerate(pix_det_ref):
        s_shifted = s + np.array(shift[::-1])

        pix_distance = np.min(np.linalg.norm(pix_ref - s_shifted, axis=1))
        if _config['pipeline']['astrometry']['verbose']:
            print(pix_distance)

        if pix_distance < _config['pipeline']['astrometry']['max_pix_distance_for_match']:
            min_ind = np.argmin(np.linalg.norm(pix_ref - s_shifted, axis=1))

            # note: errors in Gaia position are given in mas, so convert to deg by  / 1e3 / 3600
            matched.append(np.hstack([pix_det[si], pix_det_err[si],
                                      np.array([fov_stars['RA'][min_ind],
                                                fov_stars['Dec'][min_ind],
                                                fov_stars['e_RA'][min_ind] / 1e3 / 3600,
                                                fov_stars['e_Dec'][min_ind] / 1e3 / 3600,
                                                fov_stars['mag'][min_ind],
                                                fov_stars['source_id'][min_ind]])]))
            # fov_stars['RADEcor'][min_ind]])]))
            mask_matched.append(min_ind)

    matched = np.array(matched)
    if _config['pipeline']['astrometry']['verbose']:
        # print(matched)
        print('total matched:', len(matched))

    ''' plot and save fake images used to detect shift '''
    plt.close('all')
    fig = plt.figure('fake detected')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scale_image(detected, correction='local'), cmap=plt.cm.magma, origin='lower', interpolation='nearest')
    # save figure
    fname = '{:s}_fake_detected.png'.format(_obs)
    plt.savefig(os.path.join(_path_out, fname), dpi=300)

    # apply shift:
    _nthreads = multiprocessing.cpu_count()
    shifted = image_registration.fft_tools.shiftnd(detected, (shift[0], shift[1]),
                                                   nthreads=_nthreads, use_numpy_fft=False)
    plt.close('all')
    fig = plt.figure('fake detected with estimated offset')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scale_image(shifted, correction='local'), cmap=plt.cm.magma, origin='lower', interpolation='nearest')
    # save figure
    fname = '{:s}_fake_detected_offset.png'.format(_obs)
    plt.savefig(os.path.join(_path_out, fname), dpi=300)

    plt.close('all')
    fig = plt.figure('fake reference')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scale_image(reference, correction='local'), cmap=plt.cm.magma, origin='lower', interpolation='nearest')
    # save figure
    fname = '{:s}_fake_reference.png'.format(_obs)
    plt.savefig(os.path.join(_path_out, fname), dpi=300)

    ''' solve field '''
    # a priori RA/Dec positions:
    X = matched[:, 5:7]
    # Gaia source ids and mags for bookkeeping/final matches:
    source_ids = matched[:, -1]
    source_mags = matched[:, -2]
    # measured CCD positions centered around zero:
    Y = matched[:, 0:2] - (np.array(preview_img.shape) / 2.0)

    # initial parameters of the linear transform + distortion:
    # p0 = np.array([star_sc.ra.deg, star_sc.dec.deg,
    #                -1. / ((264. / 1024.) / 3600. * 0.999),
    #                1. / ((264. / 1024.) / 3600. * 0.002),
    #                1. / ((264. / 1024.) / 3600. * 0.002),
    #                1. / ((264. / 1024.) / 3600. * 0.999),
    #                1e-7, 1e-7, 1e-7, 1e-7, 1e-7,
    #                1e-2, 1e-5, 1e-7, 1e-7, 1e-5])
    # initial parameters of the linear transform:
    p0 = np.array([star_sc.ra.deg, star_sc.dec.deg,
                   -1. / ((fov_size_arcsec / 1024.) / 3600.) * 0.999,
                   1. / ((fov_size_arcsec / 1024.) / 3600.) * 0.002,
                   1. / ((fov_size_arcsec / 1024.) / 3600.) * 0.002,
                   1. / ((fov_size_arcsec / 1024.) / 3600.) * 0.999])

    ''' estimate linear transform parameters'''
    # TODO: add weights depending on sextractor error?
    # scaling params to help leastsq land on a good solution
    scaling = [1e-2, 1e-2, 1e-5, 1e-3, 1e-3, 1e-5]
    if _config['pipeline']['astrometry']['verbose']:
        print('solving with LSQ to get initial parameter estimates')
    plsq = leastsq(residual, p0, args=(Y, X), ftol=1.49012e-13, xtol=1.49012e-13, full_output=True,
                   diag=scaling)
    # print(plsq)
    if _config['pipeline']['astrometry']['verbose']:
        print(plsq[0])
    # residuals = residual(plsq[0], Y, X)
    residuals = plsq[2]['fvec']
    if _config['pipeline']['astrometry']['verbose']:
        print('residuals:')
        print(residuals)

    for jj in range(int(_config['pipeline']['astrometry']['outlier_flagging_passes'])):
        # identify outliers. they are likely to be false identifications, so discard them and redo the fit
        print('flagging outliers and refitting, take {:d}'.format(jj + 1))
        mask_outliers = residuals <= _config['pipeline']['astrometry']['outlier_pix']  # pix

        # flag:
        X = X[mask_outliers, :]
        Y = Y[mask_outliers, :]
        source_ids = source_ids[mask_outliers]
        source_mags = source_mags[mask_outliers]

        # plsq = leastsq(residual, p0, args=(Y, X), ftol=1.49012e-13, xtol=1.49012e-13, full_output=True)
        plsq = leastsq(residual, plsq[0], args=(Y, X), ftol=1.49012e-13, xtol=1.49012e-13, full_output=True)
        if _config['pipeline']['astrometry']['verbose']:
            print(plsq[0])
        # residuals = residual(plsq[0], Y, X)
        residuals = plsq[2]['fvec']
        if _config['pipeline']['astrometry']['verbose']:
            print('residuals:')
            print(residuals)

        # get an estimate of the covariance matrix:
        pcov = plsq[1]
        if (len(X) > len(p0)) and pcov is not None:
            s_sq = (residuals ** 2).sum() / (len(X) - len(p0))
            pcov = pcov * s_sq
        else:
            pcov = np.inf
        if _config['pipeline']['astrometry']['verbose']:
            print('covariance matrix diagonal estimate:')
            # print(pcov)
            print(pcov.diagonal())

    # apply bootstrap to get a reasonable estimate of what the errors of the estimated parameters are
    if _config['pipeline']['astrometry']['verbose']:
        print('solving with LSQ bootstrap')
    # plsq_bootstrap, err_bootstrap = fit_bootstrap(residual, p0, Y, X, yerr_systematic=0.0, n_samp=100)
    n_samp = _config['pipeline']['astrometry']['bootstrap']['n_samp']
    Nsigma = _config['pipeline']['astrometry']['bootstrap']['Nsigma']
    plsq_bootstrap, err_bootstrap = fit_bootstrap(residual, plsq[0], Y, X,
                                                  yerr_systematic=0.0, n_samp=n_samp, Nsigma=Nsigma)
    if _config['pipeline']['astrometry']['verbose']:
        print(plsq_bootstrap)
        print(err_bootstrap)
    residuals = residual(plsq_bootstrap, Y, X)
    if _config['pipeline']['astrometry']['verbose']:
        print('residuals:')
        print(residuals)

    # use bootstrapped solution as the final solution:
    plsq = (plsq_bootstrap, err_bootstrap, plsq[2:])

    ''' plot the result '''
    M_m1 = np.matrix([[plsq[0][2], plsq[0][3]], [plsq[0][4], plsq[0][5]]])
    M = np.linalg.pinv(M_m1)
    if _config['pipeline']['astrometry']['verbose']:
        print('M:', M)
        print('M^-1:', M_m1)

    Q, R = np.linalg.qr(M)
    # print('Q:', Q)
    # print('R:', R)

    Y_C = compute_detector_position(plsq[0], X).T + preview_img.shape[0] / 2
    Y_tan = compute_detector_position(plsq[0], np.array([list(plsq[0][0:2])])).T + preview_img.shape[0] / 2
    # print(Y_C)
    if _config['pipeline']['astrometry']['verbose']:
        print('Tangent point pixel position: ', Y_tan)
    # print('max UV: ', compute_detector_position(plsq[0], np.array([[205.573314, 28.370672],
    #                                                                [205.564369, 28.361843]])))

    theta = np.arccos(Q[1, 1]) * 180 / np.pi
    s = np.mean((abs(R[0, 0]), abs(R[1, 1]))) * 3600
    size = s * preview_img.shape[0]

    if _config['pipeline']['astrometry']['verbose']:
        print('Estimate linear transformation:')
        print('rotation angle: {:.5f} degrees'.format(theta))
        print('pixel scale: {:.7f}\" -- mean, {:.7f}\" -- x, {:.7f}\" -- y'.format(s,
                                                                                   abs(R[0, 0]) * 3600,
                                                                                   abs(R[1, 1]) * 3600))
        print('image size for mean pixel scale: {:.4f}\" x {:.4f}\"'.format(size, size))
        print('image size: {:.4f}\" x {:.4f}\"'.format(abs(R[0, 0]) * 3600 * preview_img.shape[0],
                                                       abs(R[1, 1]) * 3600 * preview_img.shape[1]))

    ''' matches: '''
    # Gaia_DR2_source_id Gaia_DR2_source_G_mag Gaia_DR2_ra_dec ccd_pixel_positions postfit_residual_pix
    matched_sources = np.hstack((np.expand_dims(source_ids, axis=1),
                                 np.expand_dims(source_mags, axis=1),
                                 X, Y + (np.array(preview_img.shape) / 2.0),
                                 np.expand_dims(residuals, axis=1)))
    if _config['pipeline']['astrometry']['verbose']:
        print('Matched sources with Gaia DR2:')
        print(matched_sources)

    ''' dump to text files '''
    with open(os.path.join(_path_out, f'{_obs}.astrometric_solution.txt'), 'w') as f:
        f.write('# LSQ-bootstrapped solution: RA_tan[deg] Dec_tan[deg] M^-1[deg/pix]\n')
        f.write(str(plsq_bootstrap) + '\n')
        f.write('# LSQ-bootstrapped solution errors: RA_tan[deg] Dec_tan[deg] M^-1[deg/pix]\n')
        f.write(str(err_bootstrap) + '\n')
        f.write('# Linear transformation matrix M[pix/deg]\n')
        f.write(str(M) + '\n')
        f.write('# Linear transformation matrix M^-1[deg/pix]\n')
        f.write(str(M_m1) + '\n')
        f.write('# Tangent point position on the EMCCD[pix]\n')
        f.write(str(Y_tan) + '\n')
        f.write('# Field rotation angle [deg]\n')
        f.write(str(theta) + '\n')
        f.write('# Pixel scale: mean[arcsec] x[arcsec] y[arcsec]\n')
        f.write('{:.7f} {:.7f} {:.7f}\n'.format(s, abs(R[0, 0]) * 3600, abs(R[1, 1]) * 3600))
        f.write('# Image size: mean[arcsec] x[arcsec] y[arcsec]\n')
        f.write('{:.7f} {:.7f} {:.7f}\n'.format(size,
                                                abs(R[0, 0]) * 3600 * preview_img.shape[0],
                                                abs(R[1, 1]) * 3600 * preview_img.shape[1]))

    with open(os.path.join(_path_out, f'{_obs}.matches.txt'), 'w') as f:
        f.write('# Matches with Gaia DR2 catalog\n')
        f.write('# source_id G_mag RA[deg] Dec[deg] emccd_x[pix] emccd_y[pix] postfit_residual[pix]\n')
        for match in matched_sources:
            f.write(f'{int(match[0])} {match[1]:.6f} {match[2]:.13f} {match[3]:.13f}  ' +
                    f'{match[4]:.3f} {match[5]:.3f} {match[6]:.3f}\n')

    ''' test the solution '''
    fov_center = SkyCoord(ra=plsq[0][0], dec=plsq[0][1], unit=(u.deg, u.deg), frame='icrs')
    detected_solution = make_image(target=fov_center, window_size=[fov_size_det, fov_size_det], _model_psf=None,
                                   pix_stars=pix_det, mag_stars=mag_det, num_pix=preview_img.shape[0])
    detected_solution[detected_solution == 0] = 0.0001
    detected_solution[np.isnan(detected_solution)] = 0.0001
    detected_solution = np.log(detected_solution)

    plt.close('all')
    fig = plt.figure('detected stars')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scale_image(detected_solution, correction='local'), cmap=plt.cm.magma,
              origin='lower', interpolation='nearest')
    ax.plot(Y_C[:, 0], Y_C[:, 1], 'o', markersize=6,
            markeredgewidth=1, markerfacecolor='None', markeredgecolor=plt.cm.Blues(0.8),
            label='Linear transformation')

    # save figure
    fname = '{:s}_detections_solved.png'.format(_obs)
    plt.savefig(os.path.join(_path_out, fname), dpi=300)

    ''' add WCS to registered image '''
    # good solution?
    # tangent point position error < 1"?
    tan_point_error_ok = np.max(err_bootstrap[:2]) < 3e-4
    # linear mapping error < 30 pixel/degree (corresponds to error of ~2 pixel/FoV)?
    linear_mapping_error_ok = np.max(err_bootstrap[2:]) < 30

    if tan_point_error_ok and linear_mapping_error_ok:
        if _config['pipeline']['astrometry']['verbose']:
            print('Solution OK, adding WCS to registered image')

        # mapping parameters:
        RA_tan, Dec_tan, M_11, M_12, M_21, M_22 = plsq_bootstrap[0], plsq_bootstrap[1], \
                                                  M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        x_tan, y_tan = Y_tan

        nx, ny = preview_img.shape

        w = wcs.WCS(naxis=2)
        w._naxis1 = nx
        w._naxis2 = ny
        w.naxis1 = w._naxis1
        w.naxis2 = w._naxis2

        w.wcs.radesys = 'ICRS'
        w.wcs.equinox = 2000.0
        # position of the tangential point on the detector [pix]
        w.wcs.crpix = np.array([x_tan, y_tan])

        # sky coordinates of the tangential point;
        w.wcs.crval = [RA_tan, Dec_tan]
        # w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        # linear mapping detector :-> focal plane [deg/pix]
        w.wcs.cd = np.array([[-M_11, -M_12],
                             [M_21, M_22]])
        # w.wcs.cd = [[-M_11, -M_12],
        #             [M_21, M_22]]

        # print(w.wcs)

        # turn WCS object into header
        new_header = w.to_header(relax=False)
        new_header.rename_keyword('PC1_1', 'CD1_1')
        new_header.rename_keyword('PC1_2', 'CD1_2')
        new_header.rename_keyword('PC2_1', 'CD2_1')
        new_header.rename_keyword('PC2_2', 'CD2_2')
        new_header.set('CD1_1', new_header['CD1_1'], 'Linear projection matrix')
        new_header.set('CD1_2', new_header['CD1_2'], 'Linear projection matrix')
        new_header.set('CD2_1', new_header['CD2_1'], 'Linear projection matrix')
        new_header.set('CD2_2', new_header['CD2_2'], 'Linear projection matrix')
        # print(new_header)
        # merge with old header:
        # for key in header.keys():
        #     new_header[key] = header[key]

        with fits.open(os.path.join(_path_registered, _fits_in)) as fi:
            data_wcs = fi[0].data
        export_fits(os.path.join(_path_registered, f'{_obs}_registered_sum.wcs.fits'), data_wcs, _header=new_header)


if __name__ == '__main__':

    obs = '1819a_10_g_20180607_062809.034490_o'
    config = get_config('/Users/dmitryduev/_caltech/python/archiver-kped/archiver/code/config.local.json')
    astrometry(obs, config)

    raise Exception('HALT, HAENDE HOCH!!')

    #####################################################################

    path_in = '/Users/dmitryduev/_caltech/python/archiver-kped/_archive/' + \
              '20180607/1819a_10_g_20180607_062809.034490_o/registration'
    fits_in = '1819a_10_g_20180607_062809.034490_o_registered_sum.fits'

    # see /usr/local/Cellar/sextractor/2.19.5/share/sextractor/default.sex

    sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "X2_IMAGE", "Y2_IMAGE", "XY_IMAGE",
                            "XWIN_IMAGE", "YWIN_IMAGE",
                            "FLUX_AUTO", "FLUXERR_AUTO",
                            "A_IMAGE", "B_IMAGE", "FWHM_IMAGE",
                            "FLAGS", "FLAGS_WEIGHT", "FLUX_RADIUS"],
                    config={"DETECT_MINAREA": 5, "PHOT_APERTURES": 16, "SATUR_LEVEL": 1e10,
                            'DETECT_THRESH': '5.0', "THRESH_TYPE": "RELATIVE",
                            "WEIGHT_TYPE": "MAP_WEIGHT", "WEIGHT_GAIN": "Y",
                            "WEIGHT_IMAGE":
                                "/Users/dmitryduev/_caltech/python/archiver-kped/_archive/20180607/calib/flat_g.fits"},
                    sexpath="sex")

    out = sew(os.path.join(path_in, fits_in))
    # # sort according to FWHM
    # out['table'].sort('FWHM_IMAGE')
    # sort according to raw flux
    out['table'].sort('FLUX_AUTO')
    # descending order: first is brightest
    out['table'].reverse()

    # remove vignetted stuff:
    weights = load_fits('/Users/dmitryduev/_caltech/python/archiver-kped/_archive/20180607/calib/flat_g.fits')
    rows_to_remove = []
    for ri, row in enumerate(out['table']):
        x, y = int(row['X_IMAGE']), int(row['Y_IMAGE'])
        # print(x, y, weights[x, y])
        if weights[x, y] < 0.98:
            rows_to_remove.append(ri)

    out['table'].remove_rows(rows_to_remove)

    # generate triangles:
    pix_det = np.vstack((out['table']['X_IMAGE'], out['table']['Y_IMAGE'])).T
    pix_det_err = np.vstack((out['table']['X2_IMAGE'], out['table']['Y2_IMAGE'], out['table']['XY_IMAGE'])).T
    mag_det = np.array(out['table']['FLUX_AUTO'])
    # brightest 20:
    # tic = _time()
    # quads_detected = triangulate(xy, cut=20)
    # print(_time() - tic)

    # for l in out['table']:
    #     print(l)
    print(out['table'])  # This is an astropy table.
    # print('detected {:d} sources'.format(len(out['table'])))
    # print(np.mean(out['table']['A_IMAGE']), np.mean(out['table']['B_IMAGE']))
    # print(np.median(out['table']['A_IMAGE']), np.median(out['table']['B_IMAGE']))

    # load first image frame from the fits file
    preview_img = load_fits(os.path.join(path_in, fits_in))
    # print(preview_img.shape)
    # scale with local contrast optimization for preview:
    # preview_img = scale_image(preview_img, correction='local')
    # preview_img = scale_image(preview_img, correction='log')
    # preview_img = scale_image(preview_img, correction='global')

    plt.close('all')
    fig = plt.figure()
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot detected objects:
    for i, _ in enumerate(out['table']['XWIN_IMAGE']):
        ax.plot(out['table']['X_IMAGE'][i] - 1, out['table']['Y_IMAGE'][i] - 1,
                'o', markersize=out['table']['FWHM_IMAGE'][i] / 2,
                markeredgewidth=2.5, markerfacecolor='None', markeredgecolor=plt.cm.Greens(0.7),
                label='SExtracted')
        # ax.annotate(i, (out['table']['X_IMAGE'][i]+40, out['table']['Y_IMAGE'][i]+40),
        #             color=plt.cm.Blues(0.3), backgroundcolor='black')

    # ax.imshow(preview_img, cmap='gray', origin='lower', interpolation='nearest')
    # ax.imshow(preview_img, cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    preview_img_zerod = preview_img
    # preview_img_zerod += np.min(np.min(preview_img_zerod)) + 0.1
    # # preview_img_zerod = np.sqrt(preview_img_zerod)
    # preview_img_zerod[preview_img_zerod == 0] = 0.1
    # preview_img_zerod[np.isnan(preview_img_zerod)] = 0.1
    # # scale with local contrast optimization for preview:
    # preview_img_zerod = scale_image(preview_img_zerod, correction='global')
    # # preview_img = scale_image(preview_img, correction='log')
    # # preview_img = scale_image(preview_img, correction='global')
    # ax.imshow(np.log(preview_img_zerod), cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    preview_img_zerod[preview_img_zerod < np.median(np.median(preview_img_zerod))*0.5] = 0.01
    ax.imshow(np.sqrt(np.sqrt(preview_img_zerod)), cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    # preview_img_zerod = preview_img + np.min(np.min(preview_img)) + 0.1
    # ax.imshow(np.sqrt(preview_img_zerod), cmap='gray', origin='lower', interpolation='nearest')

    plt.grid(False)

    # save full figure
    # fname_full = '{:s}_full.png'.format(_obs)
    # if not (os.path.exists(_path_out)):
    #     os.makedirs(_path_out)
    # plt.savefig(os.path.join(_path_out, fname_full), dpi=300)

    plt.show()

    ''' get stars from Gaia DR2 catalogue, create fake images, then cross correlate them '''
    # stars in the field (without mag cut-off):
    # star_sc = SkyCoord(ra=header['TELRA'][0], dec=header['TELDEC'][0],
    #                    unit=(u.hourangle, u.deg), frame='icrs')
    star_sc = SkyCoord(ra='19:49:01.39', dec='+67:30:05.6', unit=(u.hourangle, u.deg), frame='icrs')
    print('nominal FoV center', star_sc)
    # print(star_sc.ra.deg, star_sc.dec.deg)

    # solved for:
    star_sc = SkyCoord(ra=297.25579166666665, dec=67.50155555555556, unit=(u.deg, u.deg), frame='icrs')

    # search radius*2: " -> rad
    fov_size_ref_arcsec = 400
    fov_size_ref = fov_size_ref_arcsec * np.pi / 180.0 / 3600

    # query Kowalski for Gaia stars:
    with Kowalski(username=secrets['kowalski']['user'], password=secrets['kowalski']['password']) as kowalski:
        # if False:
        q = {"query_type": "cone_search",
             "object_coordinates": {"radec": f"[({star_sc.ra.deg}, {star_sc.dec.deg})]",
                                    "cone_search_radius": str(fov_size_ref_arcsec/2),
                                    "cone_search_unit": "arcsec"},
             "catalogs": {"Gaia_DR2": {"filter": {},
                                       "projection": {"_id": 0, "source_id": 1,
                                                      "ra": 1, "dec": 1, "ra_error": 1, "dec_error": 1,
                                                      "phot_g_mean_mag": 1}}}
             }
        r = kowalski.query(query=q, timeout=10)
        key = list(r['result']['Gaia_DR2'].keys())[0]
        fov_stars = r['result']['Gaia_DR2'][key]

        # # for a box query, must specify bottom left and upper right corner:
        # bottom_left_ra = (star_sc.ra - Angle(fov_size_ref / 2 * u.rad)).deg
        # bottom_left_dec = (star_sc.dec - Angle(fov_size_ref / 2 * u.rad)).deg
        # upper_right_ra = (star_sc.ra + Angle(fov_size_ref / 2 * u.rad)).deg
        # upper_right_dec = (star_sc.dec + Angle(fov_size_ref / 2 * u.rad)).deg
        #
        # q = {"query_type": "general_search",
        #      "query": "db['Gaia_DR2'].find({{'coordinates.radec_geojson': {{'$geoWithin': {{ '$box': [[{:f} - 180.0, {:f}], [{:f} - 180.0, {:f}]] }}}}}}, {{'_id': 0, 'source_id': 1, 'ra': 1, 'dec': 1, 'ra_error': 1, 'dec_error': 1, 'phot_g_mean_mag': 1}})".format(bottom_left_ra, bottom_left_dec, upper_right_ra, upper_right_dec)
        #      }
        # print(q)
        # r = kowalski.query(query=q, timeout=10)
        # print(r)

        # print(fov_stars)

    # convert to astropy table:
    fov_stars = np.array([[fov_star['source_id'], fov_star['ra'], fov_star['dec'],
                           fov_star['ra_error'], fov_star['dec_error'], fov_star['phot_g_mean_mag']]
                          for fov_star in fov_stars])
    fov_stars = Table(fov_stars, names=('source_id', 'RA', 'Dec', 'e_RA', 'e_Dec', 'mag'))
    print(fov_stars)

    pix_ref, mag_ref = plot_field(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                                  grid_stars=fov_stars, num_pix=preview_img.shape[0], _highlight_brighter_than_mag=None,
                                  scale_bar=False, scale_bar_size=20, _display_plot=False, _save_plot=False,
                                  path='./', name='field')
    # print(pix_ref)

    # brightest 20:
    # tic = _time()
    # quads_reference = triangulate(xy_grid, cut=30, max_pix=1500)
    # print(_time() - tic)

    # print(len(quads_detected), len(quads_reference))

    ''' detect shift '''
    fov_size_det = 264 * np.pi / 180.0 / 3600
    mag_det /= np.median(mag_det)
    mag_det = -2.5 * np.log10(mag_det)
    # add (pretty arbitrary) baseline from ref
    mag_det += np.median(mag_ref)
    # mag_det += np.max(mag_ref)
    # print(mag_det)

    # plot_field detected:
    _, _ = plot_field(target=star_sc, window_size=[fov_size_det, fov_size_det], _model_psf=None,
                      grid_stars=None, pix_stars=pix_det, mag_stars=mag_det,
                      num_pix=preview_img.shape[0], _highlight_brighter_than_mag=None,
                      scale_bar=False, scale_bar_size=20, _display_plot=False, _save_plot=False,
                      path='./', name='field')

    ''' try astrometry.net-like approach '''
    # TODO
    if False:
        # detected:
        pix = [(_p[0], _p[1]) for _p in pix_det]
        # dump to file:
        with open('pix_det.txt', 'w') as f:
            for _p in pix:
                f.write(f'{_p}\n')
        tic = time.time()
        hashes_det, hashed_quads_det = build_quad_hashes(pix)
        print(f'Building hashes for {len(pix)} detected sources took {time.time()-tic} seconds.')
        # print('hashes:\n', hashes_det)
        # print('hashed quads:\n', hashed_quads_det)
        print('number of valid hashes for detected sources:', len(hashes_det))

        # reference:
        pix = [(_p[0], _p[1]) for _p in pix_ref[:100]]
        # dump to file:
        with open('pix_ref.txt', 'w') as f:
            for _p in pix:
                f.write(f'{_p}\n')
        tic = time.time()
        hashes_ref, hashed_quads_ref = build_quad_hashes(pix)
        print(f'Building hashes for {len(pix)} reference sources took {time.time()-tic} seconds.')
        # print('hashes:\n', hashes_ref)
        # print('hashed quads:\n', hashed_quads_ref)
        print('number of valid hashes for reference sources:', len(hashes_ref))

        raise Exception('stop!')

    # detected = make_image(target=star_sc, window_size=[fov_size_det, fov_size_det], _model_psf=None,
    #                       pix_stars=pix_det, mag_stars=mag_det)
    naxis_det = int(preview_img.shape[0] * (fov_size_det * 180.0 / np.pi * 3600) / 264)
    naxis_ref = int(preview_img.shape[0] * (fov_size_ref * 180.0 / np.pi * 3600) / 264)
    print(naxis_det, naxis_ref)
    # effectively shift detected positions to center of ref frame to reduce distortion effect
    pix_det_ref = pix_det + np.array([naxis_ref // 2, naxis_ref // 2]) - np.array([naxis_det // 2, naxis_det // 2])
    # pix_det_ref = pix_det
    detected = make_image(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                          pix_stars=pix_det_ref, mag_stars=mag_det, num_pix=preview_img.shape[0])
    reference = make_image(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                           pix_stars=pix_ref, mag_stars=mag_ref, num_pix=preview_img.shape[0])

    detected[detected == 0] = 0.0001
    detected[np.isnan(detected)] = 0.0001
    detected = np.log(detected)

    reference[reference == 0] = 0.0001
    reference[np.isnan(reference)] = 0.0001
    reference = np.log(reference)

    # register shift: pixel precision first
    from skimage.feature import register_translation

    shift, error, diffphase = register_translation(reference, detected, upsample_factor=1)
    print('pixel precision offset:', shift, error)
    # shift, error, diffphase = register_translation(reference, detected, upsample_factor=2)
    # print('subpixel precision offset:', shift, error)

    # associate!
    matched = []
    mask_matched = []
    for si, s in enumerate(pix_det_ref):
        s_shifted = s + np.array(shift[::-1])

        pix_distance = np.min(np.linalg.norm(pix_ref - s_shifted, axis=1))
        if True:
            print(pix_distance)

        # note: because of larger distortion in the y-direction, pix diff there is larger than in x
        # if pix_distance < 25 * preview_img.shape[0] / 1024:  # 25:
        if pix_distance < 10:
            min_ind = np.argmin(np.linalg.norm(pix_ref - s_shifted, axis=1))

            # note: errors in Gaia position are given in mas, so convert to deg by  / 1e3 / 3600
            matched.append(np.hstack([pix_det[si], pix_det_err[si],
                                      np.array([fov_stars['RA'][min_ind],
                                                fov_stars['Dec'][min_ind],
                                                fov_stars['e_RA'][min_ind] / 1e3 / 3600,
                                                fov_stars['e_Dec'][min_ind] / 1e3 / 3600,
                                                0.0])]))
            # fov_stars['RADEcor'][min_ind]])]))
            mask_matched.append(min_ind)

    matched = np.array(matched)
    if False:
        print('matched objects:')
        print(matched)
    print('total matched:', len(matched))

    ''' plot fake images used to detect shift '''
    fig = plt.figure('fake detected')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scale_image(detected, correction='local'), cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    # apply shift:
    if True:
        import multiprocessing

        _nthreads = multiprocessing.cpu_count()
        shifted = image_registration.fft_tools.shiftnd(detected, (shift[0], shift[1]),
                                                       nthreads=_nthreads, use_numpy_fft=False)
        fig = plt.figure('fake detected with estimated offset')
        fig.set_size_inches(4, 4, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(scale_image(shifted, correction='local'), cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    fig = plt.figure('fake reference')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scale_image(reference, correction='local'), cmap=plt.cm.magma, origin='lower', interpolation='nearest')

    # also plot matched objects:
    pix_matched, mag_matched = plot_field(target=star_sc, window_size=[fov_size_ref, fov_size_ref], _model_psf=None,
                                          grid_stars=fov_stars[mask_matched], num_pix=1024,
                                          _highlight_brighter_than_mag=None, scale_bar=False,
                                          scale_bar_size=20, _display_plot=False, _save_plot=False,
                                          path='./', name='field')

    plt.show()

    ''' solve field '''
    # plt.show()
    # a priori RA/Dec positions:
    X = matched[:, 5:7]
    # measured CCD positions centered around zero:
    Y = matched[:, 0:2] - (np.array(preview_img.shape) / 2.0)

    # initial parameters of the linear transform + distortion:
    # p0 = np.array([star_sc.ra.deg, star_sc.dec.deg,
    #                -1. / ((264. / 1024.) / 3600. * 0.999),
    #                1. / ((264. / 1024.) / 3600. * 0.002),
    #                1. / ((264. / 1024.) / 3600. * 0.002),
    #                1. / ((264. / 1024.) / 3600. * 0.999),
    #                1e-7, 1e-7, 1e-7, 1e-7, 1e-7,
    #                1e-2, 1e-5, 1e-7, 1e-7, 1e-5])
    # initial parameters of the linear transform:
    p0 = np.array([star_sc.ra.deg, star_sc.dec.deg,
                   -1. / ((264. / 1024.) / 3600.) * 0.999,
                   1. / ((264. / 1024.) / 3600.) * 0.002,
                   1. / ((264. / 1024.) / 3600.) * 0.002,
                   1. / ((264. / 1024.) / 3600.) * 0.999])

    ''' estimate linear transform parameters + 2nd order distortion '''
    # TODO: add weights depending on sextractor error?
    # print('testing')
    # scaling params to help leastsq land on a good solution
    scaling = [1e-2, 1e-2, 1e-5, 1e-3, 1e-3, 1e-5]
    plsq = leastsq(residual, p0, args=(Y, X), ftol=1.49012e-13, xtol=1.49012e-13, full_output=True,
                   diag=scaling)
    # print(plsq)
    # print('residuals:')
    # residuals = plsq[2]['fvec']
    # print(residuals)

    print('solving with LSQ')
    plsq = leastsq(residual, p0, args=(Y, X), ftol=1.49012e-13, xtol=1.49012e-13, full_output=True,
                   diag=scaling)
    # print(plsq)
    print(plsq[0])
    print('residuals:')
    # residuals = residual(plsq[0], Y, X)
    residuals = plsq[2]['fvec']
    print(residuals)

    for jj in range(2):
        # identify outliers. they are likely to be false identifications, so discard them and redo the fit
        print('flagging outliers and refitting, take {:d}'.format(jj + 1))
        mask_outliers = residuals <= 5  # pix

        # flag:
        X = X[mask_outliers, :]
        Y = Y[mask_outliers, :]

        # plsq = leastsq(residual, p0, args=(Y, X), ftol=1.49012e-13, xtol=1.49012e-13, full_output=True)
        plsq = leastsq(residual, plsq[0], args=(Y, X), ftol=1.49012e-13, xtol=1.49012e-13, full_output=True)
        print(plsq[0])
        print('residuals:')
        # residuals = residual(plsq[0], Y, X)
        residuals = plsq[2]['fvec']
        print(residuals)

        # get an estimate of the covariance matrix:
        pcov = plsq[1]
        if (len(X) > len(p0)) and pcov is not None:
            s_sq = (residuals ** 2).sum() / (len(X) - len(p0))
            pcov = pcov * s_sq
        else:
            pcov = np.inf
        print('covariance matrix diagonal estimate:')
        # print(pcov)
        print(pcov.diagonal())

    # apply bootstrap to get a reasonable estimate of what the errors of the estimated parameters are
    print('solving with LSQ bootstrap')
    # plsq_bootstrap, err_bootstrap = fit_bootstrap(residual, p0, Y, X, yerr_systematic=0.0, n_samp=100)
    plsq_bootstrap, err_bootstrap = fit_bootstrap(residual, plsq[0], Y, X,
                                                  yerr_systematic=0.0, n_samp=100, Nsigma=2.0)
    print(plsq_bootstrap)
    print(err_bootstrap)
    print('residuals:')
    residuals = residual(plsq_bootstrap, Y, X)
    print(residuals)

    # FIXME: use bootstrapped solution:
    plsq = (plsq_bootstrap, err_bootstrap, plsq[2:])

    ''' plot the result '''
    M_m1 = np.matrix([[plsq[0][2], plsq[0][3]], [plsq[0][4], plsq[0][5]]])
    M = np.linalg.pinv(M_m1)
    print('M:', M)
    print('M^-1:', M_m1)

    Q, R = np.linalg.qr(M)
    # print('Q:', Q)
    # print('R:', R)

    Y_C = compute_detector_position(plsq[0], X).T + preview_img.shape[0] / 2
    Y_tan = compute_detector_position(plsq[0], np.array([list(plsq[0][0:2])])).T + preview_img.shape[0] / 2
    # print(Y_C)
    print('Tangent point pixel position: ', Y_tan)
    # print('max UV: ', compute_detector_position(plsq[0], np.array([[205.573314, 28.370672],
    #                                                                [205.564369, 28.361843]])))

    print('Estimate linear transformation:')
    theta = np.arccos(Q[1, 1]) * 180 / np.pi
    print('rotation angle: {:.5f} degrees'.format(theta))
    s = np.mean((abs(R[0, 0]), abs(R[1, 1]))) * 3600
    print('pixel scale: {:.7f}\" -- mean, {:.7f}\" -- x, {:.7f}\" -- y'.format(s,
                                                                               abs(R[0, 0]) * 3600,
                                                                               abs(R[1, 1]) * 3600))
    size = s * preview_img.shape[0]
    print('image size for mean pixel scale: {:.4f}\" x {:.4f}\"'.format(size, size))
    print('image size: {:.4f}\" x {:.4f}\"'.format(abs(R[0, 0]) * 3600 * preview_img.shape[0],
                                                   abs(R[1, 1]) * 3600 * preview_img.shape[1]))

    plt.show()

    ''' test the solution '''
    fov_center = SkyCoord(ra=plsq[0][0], dec=plsq[0][1], unit=(u.deg, u.deg), frame='icrs')
    detected_solution = make_image(target=fov_center, window_size=[fov_size_det, fov_size_det], _model_psf=None,
                                   pix_stars=pix_det, mag_stars=mag_det, num_pix=preview_img.shape[0])
    detected_solution[detected_solution == 0] = 0.0001
    detected_solution[np.isnan(detected_solution)] = 0.0001
    detected_solution = np.log(detected_solution)

    fig = plt.figure('detected stars')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scale_image(detected_solution, correction='local'), cmap=plt.cm.magma,
              origin='lower', interpolation='nearest')
    ax.plot(Y_C[:, 0], Y_C[:, 1], 'o', markersize=6,
            markeredgewidth=1, markerfacecolor='None', markeredgecolor=plt.cm.Blues(0.8),
            label='Linear transformation')

    plt.show()
