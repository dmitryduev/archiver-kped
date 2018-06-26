"""
    Archiver

    Dr. Dmitry A. Duev @ Caltech, 2016-2018
"""

import matplotlib
matplotlib.use('Agg')
import argparse
import signal
import timeout_decorator
import psutil
from collections import OrderedDict
import pytz
from distributed import Client, LocalCluster
import threading
from queue import Queue
import time
import datetime
import inspect
import os
import shutil
import subprocess
import numpy as np
from scipy.optimize import fmin
from astropy.modeling import models, fitting
import scipy.ndimage as ndimage
from scipy.stats import sigmaclip
from scipy.ndimage import gaussian_filter
import image_registration
from scipy import stats
import glob
import traceback
import sys
import json
import logging
import collections
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from itertools import chain
import pymongo
from bson.json_util import loads, dumps
import re
from astropy.io import fits
import pyprind
import functools
import hashlib
from skimage import exposure, img_as_float
from copy import deepcopy
import sewpy
import lacosmicx as lax
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, TextArea

from penquins import Kowalski

# load secrets:
with open('secrets.json') as sjson:
    secrets = json.load(sjson)

# connect to Kowalski
kowalski = Kowalski(username=secrets['kowalski']['user'], password=secrets['kowalski']['password'])


# Scale bars
class AnchoredSizeBar(AnchoredOffsetbox):
    def __init__(self, transform, size, label, loc,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, frameon=True):
        """
        Draw a horizontal bar with the size in data coordinate of the give axes.
        A label will be drawn underneath (center-aligned).

        pad, borderpad in fraction of the legend font size (or prop)
        sep in points.
        loc:
            'upper right'  : 1,
            'upper left'   : 2,
            'lower left'   : 3,
            'lower right'  : 4,
            'right'        : 5,
            'center left'  : 6,
            'center right' : 7,
            'lower center' : 8,
            'upper center' : 9,
            'center'       : 10
        """
        self.size_bar = AuxTransformBox(transform)
        self.size_bar.add_artist(Rectangle((0, 0), size, 0, fc='none', color='white', lw=3))

        self.txt_label = TextArea(label, dict(color='white', size='x-large', weight='normal'),
                                  minimumdescent=False)

        self._box = VPacker(children=[self.size_bar, self.txt_label],
                            align="center",
                            pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box,
                                   prop=prop,
                                   frameon=frameon)


class Star(object):
    """ Define a star by its coordinates and modelled FWHM
        Given the coordinates of a star within a 2D array, fit a model to the star and determine its
        Full Width at Half Maximum (FWHM).The star will be modelled using astropy.modelling. Currently
        accepted models are: 'Gaussian2D', 'Moffat2D'
    """

    _GAUSSIAN2D = 'Gaussian2D'
    _MOFFAT2D = 'Moffat2D'
    # _MODELS = set([_GAUSSIAN2D, _MOFFAT2D])

    def __init__(self, x0, y0, data, model_type=_GAUSSIAN2D,
                 box=100, plate_scale=0.0351594, exp=0.0, out_path='./'):
        """ Instantiation method for the class Star.
        The 2D array in which the star is located (data), together with the pixel coordinates (x0,y0) must be
        passed to the instantiation method. .
        """
        self.x = x0
        self.y = y0
        self._box = box
        # field of view in x in arcsec:
        self._plate_scale = plate_scale
        self._exp = exp
        self._XGrid, self._YGrid = self._grid_around_star(x0, y0, data)
        self.data = data[self._XGrid, self._YGrid]
        self.model_type = model_type
        self.out_path = out_path

    def model(self):
        """ Fit a model to the star. """
        return self._fit_model()

    @property
    def model_psf(self):
        """ Return a modelled PSF for the given model  """
        return self.model()(self._XGrid, self._YGrid)

    @property
    def fwhm(self):
        """ Extract the FWHM from the model of the star.
            The FWHM needs to be calculated for each model. For the Moffat, the FWHM is a function of the gamma and
            alpha parameters (in other words, the scaling factor and the exponent of the expression), while for a
            Gaussian FWHM = 2.3548 * sigma. Unfortunately, our case is a 2D Gaussian, so a compromise between the
            two sigmas (sigma_x, sigma_y) must be reached. We will use the average of the two.
        """
        model_dict = dict(zip(self.model().param_names, self.model().parameters))
        if self.model_type == self._MOFFAT2D:
            gamma, alpha = [model_dict[ii] for ii in ("gamma_0", "alpha_0")]
            FWHM = 2. * gamma * np.sqrt(2 ** (1/alpha) - 1)
            FWHM_x, FWHM_y = None, None
        elif self.model_type == self._GAUSSIAN2D:
            sigma_x, sigma_y = [model_dict[ii] for ii in ("x_stddev_0", "y_stddev_0")]
            FWHM = 2.3548 * np.mean([sigma_x, sigma_y])
            FWHM_x, FWHM_y = 2.3548 * sigma_x, 2.3548 * sigma_y
        return FWHM, FWHM_x, FWHM_y

    # @memoize
    def _fit_model(self):
        fit_p = fitting.LevMarLSQFitter()
        model = self._initialize_model()
        _p = fit_p(model, self._XGrid, self._YGrid, self.data)
        return _p

    def _initialize_model(self):
        """ Initialize a model with first guesses for the parameters.
        The user can select between several astropy models, e.g., 'Gaussian2D', 'Moffat2D'. We will use the data to get
        the first estimates of the parameters of each model. Finally, a Constant2D model is added to account for the
        background or sky level around the star.
        """
        max_value = self.data.max()

        if self.model_type == self._GAUSSIAN2D:
            model = models.Gaussian2D(x_mean=self.x, y_mean=self.y, x_stddev=1, y_stddev=1)
            model.amplitude = max_value

            # Establish reasonable bounds for the fitted parameters
            model.x_stddev.bounds = (0, self._box/4)
            model.y_stddev.bounds = (0, self._box/4)
            model.x_mean.bounds = (self.x - 5, self.x + 5)
            model.y_mean.bounds = (self.y - 5, self.y + 5)

        elif self.model_type == self._MOFFAT2D:
            model = models.Moffat2D()
            model.x_0 = self.x
            model.y_0 = self.y
            model.gamma = 2
            model.alpha = 2
            model.amplitude = max_value

            #  Establish reasonable bounds for the fitted parameters
            model.alpha.bounds = (1,6)
            model.gamma.bounds = (0, self._box/4)
            model.x_0.bounds = (self.x - 5, self.x + 5)
            model.y_0.bounds = (self.y - 5, self.y + 5)

        model += models.Const2D(self.fit_sky())
        model.amplitude_1.fixed = True
        return model

    def fit_sky(self):
        """ Fit the sky using a Ring2D model in which all parameters but the amplitude are fixed.
        """
        min_value = self.data.min()
        ring_model = models.Ring2D(min_value, self.x, self.y, self._box * 0.4, width=self._box * 0.4)
        ring_model.r_in.fixed = True
        ring_model.width.fixed = True
        ring_model.x_0.fixed = True
        ring_model.y_0.fixed = True
        fit_p = fitting.LevMarLSQFitter()
        return fit_p(ring_model, self._XGrid, self._YGrid, self.data).amplitude

    def _grid_around_star(self, x0, y0, data):
        """ Build a grid of side 'box' centered in coordinates (x0,y0). """
        lenx, leny = data.shape
        xmin, xmax = max(x0-self._box/2, 0), min(x0+self._box/2+1, lenx-1)
        ymin, ymax = max(y0-self._box/2, 0), min(y0+self._box/2+1, leny-1)
        return np.mgrid[int(xmin):int(xmax), int(ymin):int(ymax)]

    def plot_resulting_model(self, frame_name):
        # import matplotlib
        # matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        """ Make a plot showing data, model and residuals. """
        data = self.data
        model = self.model()(self._XGrid, self._YGrid)
        _residuals = data - model

        bar_len = data.shape[0] * 0.1
        bar_len_str = '{:.1f}'.format(bar_len * self._plate_scale)

        plt.close('all')
        fig = plt.figure(figsize=(9, 3))
        # data
        ax1 = fig.add_subplot(1, 3, 1)
        # print(sns.diverging_palette(10, 220, sep=80, n=7))
        ax1.imshow(data, origin='lower', interpolation='nearest',
                   vmin=data.min(), vmax=data.max(), cmap=plt.cm.magma)
        # ax1.title('Data', fontsize=14)
        ax1.grid('off')
        ax1.set_axis_off()
        # ax1.text(0.1, 0.8,
        #          'exposure: {:.0f} sec'.format(self._exp), color='0.75',
        #          horizontalalignment='center', verticalalignment='center')

        asb = AnchoredSizeBar(ax1.transData,
                              bar_len,
                              bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                              loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
        ax1.add_artist(asb)

        # model
        ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
        ax2.imshow(model, origin='lower', interpolation='nearest',
                   vmin=data.min(), vmax=data.max(), cmap=plt.cm.magma)
        # RdBu_r, magma, inferno, viridis
        ax2.set_axis_off()
        # ax2.title('Model', fontsize=14)
        ax2.grid('off')

        asb = AnchoredSizeBar(ax2.transData,
                              bar_len,
                              bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                              loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
        ax2.add_artist(asb)

        # residuals
        ax3 = fig.add_subplot(1, 3, 3, sharey=ax1)
        ax3.imshow(_residuals, origin='lower', interpolation='nearest', cmap=plt.cm.magma)
        # ax3.title('Residuals', fontsize=14)
        ax3.grid('off')
        ax3.set_axis_off()

        asb = AnchoredSizeBar(ax3.transData,
                              bar_len,
                              bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                              loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
        ax3.add_artist(asb)

        # plt.tight_layout()

        # dancing with a tambourine to remove the white spaces on the plot:
        fig.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, right=1, left=0)
        plt.margins(0, 0)
        from matplotlib.ticker import NullLocator
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        fig.savefig(os.path.join(self.out_path, '{:s}.png'.format(frame_name)), dpi=200)


def export_fits(path, _data, _header=None):
    """
        Save fits file overwriting if exists
    :param path:
    :param _data:
    :param _header:
    :return:
    """
    if _header is not None:
        # _header is a dict? convert to astropy.io.fits.Header first:
        if isinstance(_header, dict):
            new_header = fits.Header()
            for _k in _header:
                new_header[_k] = tuple(_header[_k])
            _header = new_header
        hdu = fits.PrimaryHDU(_data, header=_header)
    else:
        hdu = fits.PrimaryHDU(_data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(path, overwrite=True)


def load_fits(fin):
    """
        Load fits-file
    :param fin:
    :return:
    """
    with fits.open(fin) as _f:
        scidata = np.nan_to_num(_f[0].data)
    return scidata


def get_fits_header(fits_file):
    """
        Get fits-file header
    :param fits_file:
    :return:
    """
    # read fits:
    with fits.open(os.path.join(fits_file)) as hdulist:
        # header:
        header = OrderedDict()
        for _entry in hdulist[0].header.cards:
            header[_entry[0]] = _entry[1:]

    return header


def radec_str2rad(_ra_str, _dec_str):
    """

    :param _ra_str: 'H:M:S'
    :param _dec_str: 'D:M:S'
    :return: ra, dec in rad
    """
    # convert to rad:
    _ra = list(map(float, _ra_str.split(':')))
    _ra = (_ra[0] + _ra[1] / 60.0 + _ra[2] / 3600.0) * np.pi / 12.
    _dec = list(map(float, _dec_str.split(':')))
    _sign = np.sign(_dec[0]) if _dec[0] != 0 else 1
    _dec = _sign * (abs(_dec[0]) + abs(_dec[1]) / 60.0 + abs(_dec[2]) / 3600.0) * np.pi / 180.

    return _ra, _dec


def radec_str2geojson(ra_str, dec_str):

    # hms -> ::, dms -> ::
    if isinstance(ra_str, str) and isinstance(dec_str, str):
        if ('h' in ra_str) and ('m' in ra_str) and ('s' in ra_str):
            ra_str = ra_str[:-1]  # strip 's' at the end
            for char in ('h', 'm'):
                ra_str = ra_str.replace(char, ':')
        if ('d' in dec_str) and ('m' in dec_str) and ('s' in dec_str):
            dec_str = dec_str[:-1]  # strip 's' at the end
            for char in ('d', 'm'):
                dec_str = dec_str.replace(char, ':')

        if (':' in ra_str) and (':' in dec_str):
            ra, dec = radec_str2rad(ra_str, dec_str)
            # convert to geojson-friendly degrees:
            ra = ra * 180.0 / np.pi - 180.0
            dec = dec * 180.0 / np.pi
        else:
            raise Exception('Unrecognized string ra/dec format.')
    else:
        # already in degrees?
        ra = float(ra_str)
        # geojson-friendly ra:
        ra -= 180.0
        dec = float(dec_str)

    return ra, dec


def great_circle_distance(phi1, lambda1, phi2, lambda2):
    # input: dec1, ra1, dec2, ra2
    # this is much faster than astropy.coordinates.Skycoord.separation
    delta_lambda = lambda2 - lambda1
    return np.arctan2(np.sqrt((np.cos(phi2)*np.sin(delta_lambda))**2
                              + (np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(delta_lambda))**2),
                      np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(delta_lambda))


def sigma_clip(x, sigma, niter):
    x = np.array(x)
    if len(x) > 3:
        for i in range(niter):
            xt = x - np.mean(x)
            x = x[np.where(abs(xt) < sigma*np.std(xt))]
    return list(x)


def log_gauss_score(_x, _mu=1.27, _sigma=0.17):
    """
        _x: pixel for pixel in [1,2048] - source FWHM.
            has a max of 1 around 35 pix, drops fast to the left, drops slower to the right
    """
    return np.exp(-(np.log(np.log(_x)) - _mu)**2 / (2*_sigma**2))  # / 2


def gauss_score(_r, _mu=0, _sigma=512):
    """
        _r - distance from centre to source in pix
    """
    return np.exp(-(_r - _mu)**2 / (2*_sigma**2))  # / 2


def rho(x, y, x_0=1024, y_0=1024):
    return np.sqrt((x-x_0)**2 + (y-y_0)**2)


def lbunzip2(_path_in, _files, _path_out, _cmd='lbunzip2', _keep=True, _rewrite=True, _v=False):

    """
        A wrapper around lbunzip2 - a parallel version of bunzip2
    :param _path_in: folder with the files to be unzipped
    :param _files: string or list of strings with file names to be uncompressed
    :param _path_out: folder to place the output
    :param _cmd: bunzip2 or lbunzip2?
    :param _rewrite: rewrite if output exists?
    :param _keep: keep the original?
    :param _v: verbose?
    :return:
    """

    # try:
    #     subprocess.run(['which', _cmd], check=True)
    #     print('found {:s} in the system'.format(_cmd))
    # except Exception as _e:
    #     print(_e)
    #     print('{:s} not installed in the system. go ahead and install it!'.format(_cmd))
    #     return False

    if isinstance(_files, str):
        _files_list = [_files]
    else:
        _files_list = _files

    files_size = sum([os.stat(os.path.join(_path_in, fs)).st_size for fs in _files_list])
    # print(files_size)

    if _v:
        bar = pyprind.ProgBar(files_size, stream=1, title='Unzipping files', monitor=True)
    for _file in _files_list:
        file_in = os.path.join(_path_in, _file)
        file_out = os.path.join(_path_out, os.path.splitext(_file)[0])

        if not _rewrite:
            if os.path.exists(file_out) and os.stat(file_out).st_size == 0:
                os.remove(file_out)
            if os.path.exists(file_out) and os.stat(file_out).st_size != 0:
                # print('uncompressed file {:s} already exists, skipping'.format(file_in))
                if _v:
                    bar.update(iterations=os.stat(file_in).st_size)
                continue
        # else go ahead
        # print('lbunzip2 <{:s} >{:s}'.format(file_in, file_out))
        # with open(file_in, 'rb') as _f_in, open(file_out, 'wb') as _f_out:
        #     subprocess.run([_cmd], input=_f_in.read(), stdout=_f_out)

        # NOTE: trying to get rid of i/o wait issues
        # copy zipped to tmp dir:
        shutil.copy2(file_in, _path_out)
        # bz2.decompress and remove copied original
        subprocess.run([_cmd, os.path.join(_path_out, _file)])

        # remove the original if requested:
        if not _keep:
            subprocess.run(['rm', '-f', '{:s}'.format(os.path.join(_path_in, _file))], check=True)
        if _v:
            bar.update(iterations=os.stat(file_in).st_size)

    return True


def memoize(f):
    """ Minimalistic memoization decorator.
    http://code.activestate.com/recipes/577219-minimalistic-memoization/ """

    cache = {}

    @functools.wraps(f)
    def memf(*x):
        if x not in cache:
            cache[x] = f(*x)
        return cache[x]
    return memf


def mdate_walk(_path):
    """
        Inspect directory tree contents to get max mdate of all files within it
    :param _path:
    :return:
    """
    if not os.path.exists(_path):
        return utc_now()
    # modified time for the parent folder:
    mtime = datetime.datetime.utcfromtimestamp(os.stat(_path).st_mtime)
    # print(mtime)
    for root, _, files in os.walk(_path, topdown=False):
        # only check the files:
        for _f in files:
            path_f = os.path.join(root, _f)
            mtime_f = datetime.datetime.utcfromtimestamp(os.stat(path_f).st_mtime)
            if mtime_f > mtime:
                mtime = mtime_f
            # print(path_f, mtime_f)
        # don't actually need to check dirs
        # for _d in dirs:
        #     print(os.path.join(root, _d))
    return mtime


def utc_now():
    return datetime.datetime.now(pytz.utc)


class TimeoutError(Exception):
    def __init__(self, value='Operation timed out'):
        self.value = value

    def __str__(self):
        return repr(self.value)


def timeout(seconds_before_timeout):
    """
        A decorator that raises a TimeoutError error if a function/method runs longer than seconds_before_timeout
    :param seconds_before_timeout:
    :return:
    """
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutError()

        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            old_time_left = signal.alarm(seconds_before_timeout)
            if 0 < old_time_left < seconds_before_timeout:  # never lengthen existing timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
            finally:
                if old_time_left > 0:  # deduct f's run time from the saved timer
                    old_time_left -= time.time() - start_time
                signal.signal(signal.SIGALRM, old)
                signal.alarm(old_time_left)
            return result
        new_f.__name__ = f.__name__
        return new_f
    return decorate


class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


class SetQueue(Queue):
    def _init(self, maxsize):
        self.queue = set()

    def _put(self, item):

        # print('_______________________')
        # print(item in self.queue)
        # # print(item)
        # for ii, i in enumerate(self.queue):
        #     print(ii, i['id'], i['task'])
        # print('_______________________')

        if item in self.queue:
            # do not count it since it's not going to be added to the set!
            # see line 144 in queue.py (python 3.6)
            self.unfinished_tasks -= 1
        self.queue.add(item)

    def _get(self):
        self.unfinished_tasks = max(self.unfinished_tasks - 1, 0)
        return self.queue.pop()


class OrderedSetQueue(SetQueue):
    def _init(self, maxsize):
        self.queue = OrderedSet()


class Archiver(object):
    """
        A class representation of major data house-keeping/archiving tasks
    """
    def __init__(self, config_file=None):
        assert config_file is not None, 'Must specify config file'

        try:
            ''' load config data '''
            self.config = self.get_config(_config_file=config_file)

            ''' set up logging at init '''
            self.logger, self.logger_utc_date = self.set_up_logging(_name='archive', _mode='a')

            # make dirs if necessary:
            for _pp in ('archive', 'tmp'):
                _path = self.config['path']['path_{:s}'.format(_pp)]
                if not os.path.exists(_path):
                    os.makedirs(_path)
                    self.logger.debug('Created {:s}'.format(_path))

            ''' initialize dask.distributed LocalCluster for distributed task processing '''
            # alternative, or if workers are to be run on different machines
            '''
            In different terminals start the scheduler and a few workers:
            $ dask-scheduler
            $ dask-worker 127.0.0.1:8786 --nprocs 2 --nthreads 1
            $ dask-worker 127.0.0.1:8786 --nprocs 2 --nthreads 1
            $ ...
            then here:
            self.c = Client('127.0.0.1:8786')
            '''
            # set up a LocalCluster
            self.cluster = LocalCluster(n_workers=self.config['parallel']['n_workers'],
                                        threads_per_worker=self.config['parallel']['threads_per_worker'])
            # connect to local cluster
            self.c = Client(self.cluster)

            ''' set up processing queue '''
            # we will be submitting processing tasks to it
            # self.q = OrderedSetQueue()
            # self.q = SetQueue()
            self.q = Queue()

            # keep hash values of enqueued tasks to prevent submitting particular task multiple times
            self.task_hashes = set()

            # now we need to map our queue over task_runner and gather results in another queue.
            # user must setup specific tasks/jobs in task_runner, which (unfortunately)
            # cannot be defined inside a subclass -- only as a standalone function
            self.futures = self.c.map(self.task_runner, self.q,
                                      maxsize=self.config['parallel']['n_workers']*
                                              self.config['parallel']['threads_per_worker'])
            self.results = self.c.gather(self.futures, maxsize=self.config['parallel']['n_workers']*
                                         self.config['parallel']['threads_per_worker'])  # Gather results
            # self.futures = self.c.map(self.task_runner, self.q)
            # self.results = self.c.gather(self.futures)  # Gather results

            self.logger.debug('Successfully set up dask.distributed cluster')

            # Pipelining tasks (dict of form {'task': 'task_name', 'param_a': param_a_value}, jsonified)
            # to distributed queue for execution as self.q.put(task)

            # note: result harvester is defined and started in subclass!

            ''' DB connection is handled in subclass '''
            self.db = None

            ''' raw data are handled in subclass '''
            self.raw_data = None

        except Exception as e:
            print(e)
            traceback.print_exc()
            sys.exit()

    def hash_task(self, _task):
        """
            Compute hash for a hashable task
        :return:
        """
        ht = hashlib.blake2b(digest_size=12)
        ht.update(_task.encode('utf-8'))
        hsh = ht.hexdigest()
        # # it's a set, so don't worry about adding a hash multiple times
        # self.task_hashes.add(hsh)

        return hsh

    # def unhash_task(self, _hsh):
    #     """
    #         Remove hexdigest-ed hash from self.task_hashes
    #     :return:
    #     """
    #     self.task_hashes.remove(_hsh)

    def harvester(self):
        """
            Harvest processing results from dask.distributed results queue, update DB entries if necessary.
            Specific implementation details are defined in subclass
        :return:
        """
        raise NotImplementedError

    def telemetry(self):
        """
            Output basic telemetry when running self.cycle()
            Specific implementation details are defined in subclass
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def task_runner(argdict):
        """
            Helper function that maps over 'data'

        :param argdict: json-dumped dictionary with (named) parameters for the task.
                        must contain 'task' key with the task name known to this helper function
                bson.dumps is used to convert the dict to a hashable type - string - so that
                it can be used with SetQueue or OrderedSetQueue. the latter two are in turn
                used instead of regular queues to be able to check if a task has been enqueued already
        :return:
        """
        raise NotImplementedError

    @staticmethod
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

    def set_up_logging(self, _name='archive', _mode='w'):
        """ Set up logging

            :param _name:
            :param _level: DEBUG, INFO, etc.
            :param _mode: overwrite log-file or append: w or a
            :return: logger instance
            """
        # 'debug', 'info', 'warning', 'error', or 'critical'
        if self.config['misc']['logging_level'] == 'debug':
            _level = logging.DEBUG
        elif self.config['misc']['logging_level'] == 'info':
            _level = logging.INFO
        elif self.config['misc']['logging_level'] == 'warning':
            _level = logging.WARNING
        elif self.config['misc']['logging_level'] == 'error':
            _level = logging.ERROR
        elif self.config['misc']['logging_level'] == 'critical':
            _level = logging.CRITICAL
        else:
            raise ValueError('Config file error: logging level must be ' +
                             '\'debug\', \'info\', \'warning\', \'error\', or \'critical\'')

        # get path to logs from config:
        _path = self.config['path']['path_logs']

        if not os.path.exists(_path):
            os.makedirs(_path)
        utc_now = datetime.datetime.utcnow()

        # http://www.blog.pythonlibrary.org/2012/08/02/python-101-an-intro-to-logging/
        _logger = logging.getLogger(_name)

        _logger.setLevel(_level)
        # create the logging file handler
        fh = logging.FileHandler(os.path.join(_path, '{:s}.{:s}.log'.format(_name, utc_now.strftime('%Y%m%d'))),
                                 mode=_mode)
        logging.Formatter.converter = time.gmtime

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # formatter = logging.Formatter('%(asctime)s %(message)s')
        fh.setFormatter(formatter)

        # add handler to logger object
        _logger.addHandler(fh)

        return _logger, utc_now.strftime('%Y%m%d')

    def shut_down_logger(self):
        """
            Prevent writing to multiple log-files after 'manual rollover'
        :return:
        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def check_logging(self):
        """
            Check if a new log file needs to be started and start it if necessary
        """
        if datetime.datetime.utcnow().strftime('%Y%m%d') != self.logger_utc_date:
            # reset
            self.shut_down_logger()
            self.logger, self.logger_utc_date = self.set_up_logging(_name='archive', _mode='a')

    def connect_to_db(self):
        """
            Connect to database. Specific details will differ for different users,
            so this should be implemented in subclass
        :return:
        """
        raise NotImplementedError

    def disconnect_from_db(self):
        """
            Disconnect from database. Specific details will differ for different users,
            so this should be implemented in subclass
        :return:
        """
        raise NotImplementedError

    def check_db_connection(self):
        """
            Check if DB connection is alive
        :return:
        """
        raise NotImplementedError

    def get_raw_data_descriptors(self):
        """
            Parse sources containing raw data and get high-level descriptors (like dates),
            by which raw data are sorted.
        :return:
        """
        raise NotImplementedError

    def cycle(self):
        """
            Main processing cycle to loop over dates and observations
        :return:
        """
        raise NotImplementedError

    def naptime(self):
        """
            Return time to sleep (in seconds) for archiving engine
            before waking up to rerun itself.
             During "working hours", it's set up in the config
             During nap time, it's nap_time_start_utc - utc_now()
        :return: time interval in seconds to sleep for
        """
        _config = self.config['misc']
        try:
            # local or UTC?
            tz = pytz.utc if _config['nap_time_frame'] == 'UTC' else None
            now = datetime.datetime.now(tz)

            if _config['nap_at_night']:

                last_midnight = datetime.datetime(now.year, now.month, now.day, tzinfo=tz)
                next_midnight = datetime.datetime(now.year, now.month, now.day, tzinfo=tz) \
                                + datetime.timedelta(days=1)

                hm_start = list(map(int, _config['nap_time_start'].split(':')))
                hm_stop = list(map(int, _config['nap_time_stop'].split(':')))

                if hm_stop[0] < hm_start[0]:
                    h_before_midnight = 24 - (hm_start[0] + hm_start[1] / 60.0)
                    h_after_midnight = hm_stop[0] + hm_stop[1] / 60.0

                    # print((next_midnight - now).total_seconds() / 3600.0, h_before_midnight)
                    # print((now - last_midnight).total_seconds() / 3600.0, h_after_midnight)

                    if (next_midnight - now).total_seconds() / 3600.0 < h_before_midnight:
                        sleep_until = next_midnight + datetime.timedelta(hours=h_after_midnight)
                        print('sleep until:', sleep_until)
                    elif (now - last_midnight).total_seconds() / 3600.0 < h_after_midnight:
                        sleep_until = last_midnight + datetime.timedelta(hours=h_after_midnight)
                        print('sleep until:', sleep_until)
                    else:
                        sleep_until = now + datetime.timedelta(minutes=_config['loop_interval'])
                        print('sleep until:', sleep_until)

                else:
                    h_after_midnight_start = hm_start[0] + hm_start[1] / 60.0
                    h_after_midnight_stop = hm_stop[0] + hm_stop[1] / 60.0

                    if (last_midnight + datetime.timedelta(hours=h_after_midnight_start) <
                            now < last_midnight + datetime.timedelta(hours=h_after_midnight_stop)):
                        sleep_until = last_midnight + datetime.timedelta(hours=h_after_midnight_stop)
                        print('sleep until:', sleep_until)
                    else:
                        sleep_until = now + datetime.timedelta(minutes=_config['loop_interval'])
                    print('sleep until:', sleep_until)

                return (sleep_until - now).total_seconds()

            else:
                # sleep for loop_interval minutes otherwise (return seconds)
                return _config['loop_interval'] * 60.0

        except Exception as _e:
            traceback.print_exc()
            self.logger.error('Failed to take a nap, taking a pill to fall asleep for an hour.')
            self.logger.error(_e)
            return 3600.0

    def sleep(self):
        """
            Take a nap in between cycles
        :return:
        """
        sleep_for = self.naptime()  # seconds
        self.logger.debug('Falling asleep for {:.1f} minutes.'.format(sleep_for / 60.0))
        time.sleep(sleep_for)


class KPEDArchiver(Archiver):
    """
        A class representation of major data house-keeping/archiving tasks for
        Robo-AO, a robotic laser guide star adaptive optics system
    """
    def __init__(self, config_file=None):
        """
            Init
        :param config_file:
        """
        ''' initialize super class '''
        super(KPEDArchiver, self).__init__(config_file=config_file)

        ''' init db if necessary '''
        self.init_db()

        ''' connect to db: '''
        # will exit if this fails
        self.connect_to_db()

        ''' start results harvester in separate thread '''
        self.running = True
        self.h = threading.Thread(target=self.harvester)
        self.h.start()

        ''' start outputting telemetry in separate thread '''
        self.start_time = utc_now()
        self.t = threading.Thread(target=self.telemetry)
        self.t.start()

    def init_db(self):
        """
            Initialize db if new Mongo instance
        :return:
        """
        _client = pymongo.MongoClient(username=self.config['database']['admin'],
                                      password=self.config['database']['admin_pwd'],
                                      host=self.config['database']['host'],
                                      port=self.config['database']['port'])
        # _id: db_name.user_name
        user_ids = [_u['_id'] for _u in _client.admin.system.users.find({}, {'_id': 1})]

        db_name = self.config['database']['db']
        username = self.config['database']['user']

        # print(f'{db_name}.{username}')
        # print(user_ids)

        if f'{db_name}.{username}' not in user_ids:
            _client[db_name].command('createUser', self.config['database']['user'],
                                     pwd=self.config['database']['pwd'], roles=['readWrite'])
            print('Successfully initialized db')

    def harvester(self):
        """
            Harvest processing results from dask.distributed results queue, update DB entries if necessary
        :return:
        """
        # make sure the archiver is running. this is to protect from frozen thread on (user) exit/crash
        while self.running:
            # get new results from queue one by one
            while not self.results.empty():
                try:
                    result = self.results.get()
                    # self.logger.debug('Task finished saying: {:s}'.format(str(result)))
                    print('Task finished saying:\n', str(result))
                    self.logger.info('Task {:s} for {:s} finished with status {:s}'.format(result['job'],
                                                                                           result['_id'],
                                                                                           result['status']))
                    # remove from self.task_hashes
                    if 'hash' in result:
                        self.task_hashes.remove(result['hash'])

                    # update DB entry
                    if 'db_record_update' in result:
                        self.update_db_entry(_collection='coll_obs', upd=result['db_record_update'])
                except Exception as _e:
                    print(_e)
                    traceback.print_exc()
                    self.logger.error(_e)
            # don't need to check that too often
            time.sleep(5)

    def telemetry(self):
        """
            Output basic telemetry when running self.cycle()
            Specific implementation details are defined in subclass
        :return:
        """
        # make sure the archiver is running. this is to protect from frozen thread on (user) exit/crash
        while self.running:
            # construct line with telemetry
            try:
                # UTC running? start_time #_enqueued_tasks system_CPU_usage_% system_memory_usage_%
                _r = 'YES' if self.running else 'NO'
                _start_time = self.start_time.strftime('%Y-%m-%d %H:%M:%S')
                _utc_now = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                _n_tasks = len(self.task_hashes)
                _cpu_usage = psutil.cpu_percent(interval=None)
                _mem_usage = psutil.virtual_memory().percent
                _root = psutil.disk_usage('/').percent
                _data_1 = psutil.disk_usage('/data').percent
                _data_2 = psutil.disk_usage('/archive').percent
                _t = '{:s} {:s} {:s} {:d} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(_utc_now, _r,
                                                                                       _start_time, _n_tasks,
                                                                                       _cpu_usage, _mem_usage,
                                                                                       _root, _data_1,
                                                                                       _data_2)
                with open(os.path.join(self.config['path']['path_logs'], 'archiver_status'), 'w') as _f:
                    _f.write(_t)
            except Exception as _e:
                print(_e)
                traceback.print_exc()
                self.logger.error(_e)

            # take a nap
            time.sleep(1.)

        ''' process killed? let the world know! '''
        # construct line with telemetry
        try:
            # UTC running? start_time #_enqueued_tasks system_CPU_usage_% system_memory_usage_%
            _r = 'YES' if self.running else 'NO'
            _start_time = self.start_time.strftime('%Y-%m-%d %H:%M:%S')
            _n_tasks = len(self.task_hashes)
            _cpu_usage = psutil.cpu_percent(interval=None)
            _mem_usage = psutil.virtual_memory().percent
            _root = psutil.disk_usage('/').percent
            _data_1 = psutil.disk_usage('/data').percent
            _data_2 = psutil.disk_usage('/archive').percent
            _t = '{:s} {:s} {:s} {:d} {:.1f} {:.1f}\n'.format(utc_now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                                              _r, _start_time, _n_tasks, _cpu_usage, _mem_usage,
                                                              _root, _data_1, _data_2)
            with open(os.path.join(self.config['path']['path_logs'], 'archiver_status'), 'w') as _f:
                _f.write(_t)
        except Exception as _e:
            print(_e)
            traceback.print_exc()
            self.logger.error(_e)

    @staticmethod
    def task_runner(argdict_and_hash):
        """
            Helper function that maps over 'data'

        :param argdict: json-dumped dictionary with (named) parameters for the task.
                        must contain 'task' key with the task name known to this helper function.
                        json.dumps is used to convert the dict to a hashable type - string - so that
                        it can be serialized.
        :return:
        """
        try:
            # unpack jsonified dict representing task:
            argdict = loads(argdict_and_hash[0])
            # get task hash:
            _task_hash = argdict_and_hash[1]

            assert 'task' in argdict, 'specify which task to run'
            print('running task {:s}'.format(argdict['task']))

            if argdict['task'] == 'registration_pipeline':
                result = job_registration_pipeline(_id=argdict['id'], _config=argdict['config'],
                                                   _db_entry=argdict['db_entry'], _task_hash=_task_hash)

            elif argdict['task'] == 'registration_pipeline:preview':
                result = job_registration_pipeline_preview(_id=argdict['id'], _config=argdict['config'],
                                                           _db_entry=argdict['db_entry'], _task_hash=_task_hash)

            elif argdict['task'] == 'photometry_pipeline':
                result = {'status': 'error', 'message': 'not implemented'}

            elif argdict['task'] == 'astrometry_pipeline':
                result = {'status': 'error', 'message': 'not implemented yet'}

            else:
                result = {'status': 'error', 'message': 'unknown task'}

        except Exception as _e:
            # exception here means bad argdict.
            print(_e)
            traceback.print_exc()
            result = {'status': 'error', 'message': str(_e)}

        return result

    @timeout(seconds_before_timeout=120)
    def connect_to_db(self):
        """
            Connect to Robo-AO's MongoDB-powered database
        :return:
        """
        _config = self.config
        try:
            if self.logger is not None:
                self.logger.debug('Connecting to the KPED database at {:s}:{:d}'.
                                  format(_config['database']['host'], _config['database']['port']))
            _client = pymongo.MongoClient(host=_config['database']['host'], port=_config['database']['port'])
            # grab main database:
            _db = _client[_config['database']['db']]

        except Exception as _e:
            if self.logger is not None:
                self.logger.error(_e)
                self.logger.error('Failed to connect to the KPED database at {:s}:{:d}'.
                                  format(_config['database']['host'], _config['database']['port']))
            # raise error
            raise ConnectionRefusedError
        try:
            # authenticate
            _db.authenticate(_config['database']['user'], _config['database']['pwd'])
            if self.logger is not None:
                self.logger.debug('Successfully authenticated with the KPED database at {:s}:{:d}'.
                                  format(_config['database']['host'], _config['database']['port']))
        except Exception as _e:
            if self.logger is not None:
                self.logger.error(_e)
                self.logger.error('Authentication failed for the KPED database at {:s}:{:d}'.
                                  format(_config['database']['host'], _config['database']['port']))
            raise ConnectionRefusedError
        try:
            # get collection with observations
            _coll_obs = _db[_config['database']['collection_obs']]
            if self.logger is not None:
                self.logger.debug('Using collection {:s} with obs data in the database'.
                                  format(_config['database']['collection_obs']))
        except Exception as _e:
            if self.logger is not None:
                self.logger.error(_e)
                self.logger.error('Failed to use a collection {:s} with obs data in the database'.
                                  format(_config['database']['collection_obs']))
            raise NameError
        try:
            # get collection with auxiliary stuff
            _coll_aux = _db[_config['database']['collection_aux']]
            if self.logger is not None:
                self.logger.debug('Using collection {:s} with aux data in the database'.
                                  format(_config['database']['collection_aux']))
        except Exception as _e:
            if self.logger is not None:
                self.logger.error(_e)
                self.logger.error('Failed to use a collection {:s} with aux data in the database'.
                                  format(_config['database']['collection_aux']))
            raise NameError
        try:
            # get collection with user access credentials
            _coll_usr = _db[_config['database']['collection_pwd']]
            if self.logger is not None:
                self.logger.debug('Using collection {:s} with user access credentials in the database'.
                                  format(_config['database']['collection_pwd']))
        except Exception as _e:
            if self.logger is not None:
                self.logger.error(_e)
                self.logger.error('Failed to use a collection {:s} with user access credentials in the database'.
                                  format(_config['database']['collection_pwd']))
            raise NameError
        try:
            # build dictionary program num -> pi name
            cursor = _coll_usr.find()
            _program_pi = {}
            for doc in cursor:
                # handle admin separately
                if str(doc['_id']) == 'admin':
                    continue
                _progs = doc['programs']
                for v in _progs:
                    # multiple users could have access to the same program, that's totally fine!
                    if str(v) not in _program_pi:
                        _program_pi[str(v)] = [str(doc['_id'])]
                    else:
                        _program_pi[str(v)].append(str(doc['_id']))
                        # print(program_pi)
        except Exception as _e:
            _program_pi = {}
            if self.logger is not None:
                self.logger.error(_e)

        if self.logger is not None:
            self.logger.debug('Successfully connected to KPED database at {:s}:{:d}'.
                              format(_config['database']['host'], _config['database']['port']))

        # (re)define self.db
        self.db = dict()
        self.db['client'] = _client
        self.db['db'] = _db
        self.db['coll_obs'] = _coll_obs
        self.db['coll_aux'] = _coll_aux
        self.db['program_pi'] = _program_pi

    @timeout(seconds_before_timeout=120)
    def disconnect_from_db(self):
        """
            Disconnect from Robo-AO's MongoDB database.
        :return:
        """
        self.logger.debug('Disconnecting from the database.')
        if self.db is not None:
            try:
                self.db['client'].close()
                self.logger.debug('Successfully disconnected from the database.')
            except Exception as e:
                self.logger.error('Failed to disconnect from the database.')
                self.logger.error(e)
            finally:
                # reset
                self.db = None
        else:
            self.logger.debug('No connection found.')

    @timeout(seconds_before_timeout=120)
    def check_db_connection(self):
        """
            Check if DB connection is alive/established.
        :return: True if connection is OK
        """
        self.logger.debug('Checking database connection.')
        if self.db is None:
            try:
                self.connect_to_db()
            except Exception as e:
                print('Lost database connection.')
                self.logger.error('Lost database connection.')
                self.logger.error(e)
                return False
        else:
            try:
                # force connection on a request as the connect=True parameter of MongoClient seems
                # to be useless here
                self.db['client'].server_info()
            except pymongo.errors.ServerSelectionTimeoutError as e:
                print('Lost database connection.')
                self.logger.error('Lost database connection.')
                self.logger.error(e)
                return False

        return True

    @timeout(seconds_before_timeout=120)
    def get_raw_data_descriptors(self):
        """
            Parse source(s) containing raw data and get dates with observational data.
        :return:
        """
        def is_date(_d, _fmt='%Y%m%d'):
            """
                Check if string (folder name) matches datetime format fmt
            :param _d:
            :param _fmt:
            :return:
            """
            try:
                datetime.datetime.strptime(_d, _fmt)
            except Exception as e:
                self.logger.error(e)
                return False
            return True

        # get all dates with some raw data from all input sources
        dates = dict()
        # KPED's NAS archive contains folders named as YYYYMMDD.
        # Only consider data taken starting from archiving_start_date
        archiving_start_date = datetime.datetime.strptime(self.config['misc']['archiving_start_date'], '%Y/%m/%d')
        for _p in self.config['path']['path_raw']:
            dates[_p] = sorted([d for d in os.listdir(_p)
                                if os.path.isdir(os.path.join(_p, d))
                                and is_date(d, _fmt='%Y%m%d')
                                and datetime.datetime.strptime(d, '%Y%m%d') >= archiving_start_date
                                ])
        return dates

    @staticmethod
    def get_raw_data(_location, _date):
        """
            Get bzipped raw data file names at _location/_date
        :param _location:
        :param _date:
        :return:
        """
        return sorted([os.path.basename(_p) for _p in glob.glob(os.path.join(_location, _date, '*.fits.fz'))])

    @timeout(seconds_before_timeout=20)
    def insert_db_entry(self, _collection=None, _db_entry=None):
        """
            Insert a document _doc to collection _collection in DB.
            It is monitored for timeout in case DB connection hangs for some reason
        :param _collection:
        :param _db_entry:
        :return:
        """
        assert _collection is not None, 'Must specify collection'
        assert _db_entry is not None, 'Must specify document'
        try:
            self.db[_collection].insert_one(_db_entry)
        except Exception as e:
            self.logger.error('Error inserting {:s} into {:s}'.format(_db_entry, _collection))
            self.logger.error(e)

    @timeout_decorator.timeout(20, use_signals=False)
    def update_db_entry(self, _collection=None, upd=None):
        """
            Update DB entry
            Note: it's mainly used by archiver's harvester, which is run in separate thread,
                  therefore signals don't work, so can't use @timeout.
                  instead, using @timeout_decorator with use_signals=False to utilize multiprocessing
        :return:
        """
        assert _collection is not None, 'Must specify collection'
        assert upd is not None, 'Must specify update'
        try:
            self.db[_collection].update_one(upd[0], upd[1])
            self.logger.info('Updated DB entry for {:s}'.format(upd[0]['_id']))
        except Exception as e:
            self.logger.error('Error executing {:s} for {:s}'.format(upd, _collection))
            self.logger.error(e)

    @staticmethod
    def empty_db_aux_entry(_date):
        """
                A dummy database record for a science observation
            :param: _date YYYYMMDD which serves as _id
            :return:
            """
        time_now_utc = utc_now()
        return {
            '_id': _date,
            'calib': {'done': False,
                      'raw': {'bias': [],
                              'dark': [],
                              'flat': []},
                      'retries': 0,
                      'last_modified': time_now_utc},
            'seeing': {'done': False,
                       'frames': [],
                       'retries': 0,
                       'last_modified': time_now_utc}
               }

    def cycle(self):
        """
            Main processing cycle
        :return:
        """
        try:
            # set up patterns, as these are not going to change
            # check the endings (\Z) and skip _N.fits.fz:
            # # science obs must start with program number (e.g. 24_ or 24.1_)
            # pattern_start = r'\d+.?\d??_'
            # must be a fpacked fits file
            pattern_end = r'.[0-9]{6}.fits.fz\Z'
            pattern_fits = r'.fits.fz\Z'

            while True:

                # check if a new log file needs to be started
                self.check_logging()

                # check if DB connection is alive/established
                connected = self.check_db_connection()

                if connected:

                    # get all dates with raw data for each raw data location
                    dates = self.get_raw_data_descriptors()
                    print(dates)

                    # iterate over data locations:
                    for location in dates:
                        for date in dates[location][::-1]:
                            # Each individual step where I know something could go wrong is placed inside a try-except
                            # clause. Everything else is captured outside causing the main while loop to terminate.
                            self.logger.debug('Processing {:s} at {:s}'.format(date, location))
                            print(date)

                            # get all raw data file names for the date, including calibration, seeing, and pointing:
                            try:
                                date_raw_data = self.get_raw_data(location, date)
                                if len(date_raw_data) == 0:
                                    # no data? proceed to next date
                                    self.logger.debug('No data found for {:s}'.format(date))
                                    continue
                                else:
                                    self.logger.debug('Found {:d} zipped fits-files for {:s}'.format(len(date_raw_data),
                                                                                                     date))
                            except Exception as _e:
                                print(_e)
                                traceback.print_exc()
                                self.logger.error(_e)
                                self.logger.error('Failed to get all raw data file names for {:s}.'.format(date))
                                continue

                            ''' auxiliary data '''
                            # look up aux entry for date in the database:
                            try:
                                select = self.db['coll_aux'].find_one({'_id': date}, max_time_ms=5000)
                                # if entry not in database, create empty one and populate it
                                if select is None:
                                    # insert empty entry for date into aux database:
                                    self.insert_db_entry(_collection='coll_aux',
                                                         _db_entry=self.empty_db_aux_entry(date))
                            except Exception as _e:
                                print(_e)
                                traceback.print_exc()
                                self.logger.error(_e)
                                self.logger.error('Error in handling aux database entry for {:s}.'.format(date))
                                continue

                            # handle calibration data
                            try:
                                # we do this in a serial way, i.e. before proceeding with everything else
                                # because calibration data are needed by everything else
                                s = self.calibration(location, date, date_raw_data)
                                if s['status'] == 'ok' and s['message'] is not None:
                                    self.update_db_entry(_collection='coll_aux', upd=s['db_record_update'])
                                    self.logger.info('Updated auxiliary entry for {:s}'.format(date))
                                    self.logger.debug(dumps(s['db_record_update']))
                                elif s['status'] == 'error':
                                    if 'db_record_update' in s:
                                        self.update_db_entry(_collection='coll_aux', upd=s['db_record_update'])
                                    raise RuntimeError(s['message'])
                            except Exception as _e:
                                print('Error in calibration():', _e)
                                # TODO: use "default" calib instead
                                # traceback.print_exc()
                                self.logger.error(_e)
                                self.logger.error('Failed to process calibration data for {:s}.'.format(date))
                                continue

                            # TODO: handle auxiliary data (seeing, summary contrast curves and Strehl ratios)

                            # once done with aux data processing, get entry from aux collection in DB:
                            # look up aux entry for date in the database:
                            try:
                                s = self.auxiliary(location, date, date_raw_data)
                                if s['status'] == 'ok' and s['message'] is not None:
                                    self.update_db_entry(_collection='coll_aux', upd=s['db_record_update'])
                                    self.logger.info('Updated auxiliary entry for {:s}'.format(date))
                                    self.logger.debug(dumps(s['db_record_update']))
                                elif s['status'] == 'error':
                                    if 'db_record_update' in s:
                                        self.update_db_entry(_collection='coll_aux', upd=s['db_record_update'])
                                    raise RuntimeError(s['message'])

                                # reload aux data:
                                aux_date = self.db['coll_aux'].find_one({'_id': date}, max_time_ms=5000)
                            except Exception as _e:
                                print(_e)
                                traceback.print_exc()
                                self.logger.error(_e)
                                self.logger.error('Error in handling aux database entry for {:s}.'.format(date))
                                continue

                            ''' science data '''
                            # get all science observations
                            try:
                                # skip calibration files and pointings
                                date_obs = [re.split(pattern_fits, s)[0] for s in date_raw_data
                                            if re.search(pattern_end, s) is not None and
                                            # re.match(pattern_start, s) is not None and
                                            re.match('bias_', s) is None and
                                            re.match('dark_', s) is None and
                                            re.match('flat_', s) is None and
                                            re.match('pointing_', s) is None and
                                            re.match('seeing_', s) is None]
                                print(date_obs)
                                if len(date_obs) == 0:
                                    # no data? proceed to next date
                                    self.logger.info('No science data found for {:s}'.format(date))
                                    continue
                                else:
                                    self.logger.debug(
                                        'Found {:d} zipped science fits-files for {:s}'.format(len(date_raw_data),
                                                                                               date))
                            except Exception as _e:
                                print(_e)
                                traceback.print_exc()
                                self.logger.error(_e)
                                self.logger.error('Failed to get all raw science data file names for {:s}.'
                                                  .format(date))
                                continue

                            ''' iterate over individual observations '''
                            for obs in date_obs:
                                try:
                                    # look up entry for obs in DB:
                                    select = self.db['coll_obs'].find_one({'_id': obs}, max_time_ms=10000)

                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error('Failed to look up entry for {:s} in DB.'.format(obs))
                                    self.logger.error(_e)
                                    continue

                                try:
                                    # init KPEDObservation object
                                    kped_obs = KPEDObservation(_id=obs, _aux=aux_date,
                                                               _program_pi=self.db['program_pi'],
                                                               _db_entry=select,
                                                               _config=self.config)
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error('Failed to set up obs object for {:s}.'.format(obs))
                                    self.logger.error(_e)
                                    continue

                                try:
                                    # init DB entry if not in DB
                                    if select is None:
                                        self.insert_db_entry(_collection='coll_obs', _db_entry=kped_obs.db_entry)
                                        self.logger.info('Inserted {:s} into DB'.format(obs))
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error(_e)
                                    self.logger.error('Initial DB insertion failed for {:s}.'.format(obs))
                                    continue

                                try:
                                    # check raws
                                    s = kped_obs.check_raws(_location=location, _date=date,
                                                            _date_raw_data=date_raw_data)
                                    # changes detected?
                                    if s['status'] == 'ok' and s['message'] is not None:
                                        # print(s['db_record_update'])
                                        self.update_db_entry(_collection='coll_obs', upd=s['db_record_update'])
                                        self.logger.info('Corrected raw_data entry for {:s}'.format(obs))
                                        self.logger.debug(dumps(s['db_record_update']))
                                    # something failed?
                                    elif s['status'] == 'error':
                                        self.logger.error('{:s}, checking raw files failed: {:s}'.format(obs,
                                                                                                         s['message']))
                                        # proceed to next obs:
                                        continue
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error(_e)
                                    self.logger.error('Raw files check failed for {:s}.'.format(obs))
                                    continue

                                try:
                                    # check that DB entry reflects reality
                                    s = kped_obs.check_db_entry()
                                    # discrepancy detected?
                                    if s['status'] == 'ok' and s['message'] is not None:
                                        self.update_db_entry(_collection='coll_obs', upd=s['db_record_update'])
                                        self.logger.info('Corrected DB entry for {:s}'.format(obs))
                                        self.logger.debug(dumps(s['db_record_update']))
                                    # something failed?
                                    elif s['status'] == 'error':
                                        self.logger.error('{:s}, checking DB entry failed: {:s}'.format(obs,
                                                                                                        s['message']))
                                        if 'db_record_update' in s:
                                            self.update_db_entry(_collection='coll_obs', upd=s['db_record_update'])
                                        self.logger.info('Corrected DB entry for {:s}'.format(obs))
                                        # proceed to next obs:
                                        continue
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error(_e)
                                    self.logger.error('Raw files check failed for {:s}.'.format(obs))
                                    continue

                                try:
                                    # check that DB entry reflects reality
                                    s = kped_obs.check_aux()
                                    # discrepancy detected?
                                    if s['status'] == 'ok' and s['message'] is not None:
                                        self.update_db_entry(_collection='coll_obs', upd=s['db_record_update'])
                                        self.logger.info('Corrected DB entry for {:s}'.format(obs))
                                        self.logger.debug(dumps(s['db_record_update']))
                                    # something failed?
                                    elif s['status'] == 'error':
                                        self.logger.error('{:s}, checking DB entry failed: {:s}'.format(obs,
                                                                                                        s['message']))
                                        if 'db_record_update' in s:
                                            self.update_db_entry(_collection='coll_obs', upd=s['db_record_update'])
                                        self.logger.info('Corrected DB entry for {:s}'.format(obs))
                                        # proceed to next obs:
                                        continue
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error(_e)
                                    self.logger.error('Raw files check failed for {:s}.'.format(obs))
                                    continue

                                try:
                                    # we'll be adding one task per observation at a time to avoid
                                    # complicating things.
                                    # self.task_runner will take care of executing the task
                                    pipe_task = kped_obs.get_task()

                                    if pipe_task is not None:
                                        # print(pipe_task)
                                        # try enqueueing. self.task_hashes takes care of possible duplicates
                                        # use bson dumps to serialize input dictionary _task. this way,
                                        # it may be pickled and enqueued (and also hashed):
                                        pipe_task_hashable = dumps(pipe_task)

                                        # compute hash for task:
                                        pipe_task_hash = self.hash_task(pipe_task_hashable)
                                        # not enqueued?
                                        if pipe_task_hash not in self.task_hashes:
                                            print({'id': pipe_task['id'], 'task': pipe_task['task']})
                                            if 'db_record_update' in pipe_task:
                                                # mark as enqueued in DB:
                                                self.update_db_entry(_collection='coll_obs',
                                                                     upd=pipe_task['db_record_update'])
                                            # enqueue the task together with its hash:
                                            self.q.put((pipe_task_hashable, pipe_task_hash))
                                            # bookkeeping:
                                            self.task_hashes.add(pipe_task_hash)
                                            self.logger.info('Enqueueing task {:s} for {:s}'.format(pipe_task['task'],
                                                                                                    pipe_task['id']))
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error(_e)
                                    self.logger.error('Failed to get/enqueue a task for {:s}.'.format(obs))
                                    continue

                                try:
                                    # nothing to do about obs?
                                    if pipe_task is None:
                                        # check distribution status
                                        s = kped_obs.check_distributed()
                                        if s['status'] == 'ok' and s['message'] is not None:
                                            self.update_db_entry(_collection='coll_obs', upd=s['db_record_update'])
                                            self.logger.info('updated distribution status for {:s}'.format(obs))
                                            self.logger.debug(dumps(s['db_record_update']))
                                        # something failed?
                                        elif s['status'] == 'error':
                                            self.logger.error('{:s}, checking distribution status failed: {:s}'.
                                                              format(obs, s['message']))
                                            # proceed to next obs:
                                            continue
                                except Exception as _e:
                                    print(_e)
                                    traceback.print_exc()
                                    self.logger.error(_e)
                                    self.logger.error('Distribution status check failed for {:s}.'.format(obs))
                                    continue

                # unfinished tasks live here:
                # print(self.q.unfinished_tasks)
                # released tasks live here:
                # print(self.c.scheduler.released())
                self.sleep()

        except KeyboardInterrupt:
            # user ctrl-c'ed
            self.running = False
            self.logger.error('User exited the archiver.')
            # try disconnecting from the database (if connected) and closing the cluster
            try:
                self.logger.info('Shutting down.')
                self.logger.debug('Cleaning tmp directory.')
                shutil.rmtree(self.config['path']['path_tmp'])
                os.makedirs(self.config['path']['path_tmp'])
                self.logger.debug('Disconnecting from DB.')
                self.disconnect_from_db()
                self.logger.debug('Shutting down dask.distributed cluster.')
                self.cluster.close()
            finally:
                self.logger.info('Finished archiving cycle.')
                return False

        except RuntimeError as e:
            # any other error not captured otherwise
            print(e)
            traceback.print_exc()
            self.running = False
            self.logger.error(e)
            self.logger.error('Unknown error, exiting. Please check the logs.')
            try:
                self.logger.info('Shutting down.')
                self.logger.debug('Cleaning tmp directory.')
                shutil.rmtree(self.config['path']['path_tmp'])
                os.makedirs(self.config['path']['path_tmp'])
                self.logger.debug('Disconnecting from DB.')
                self.disconnect_from_db()
                self.logger.debug('Shutting down dask.distributed cluster.')
                self.cluster.close()
            finally:
                self.logger.info('Finished archiving cycle.')
                return False

    @timeout(seconds_before_timeout=600)
    def load_darks_and_flats(self, _date, _mode, _filt, image_size_x=1024):
        """
            Load darks and flats
        :param _date:
        :param _mode:
        :param _filt:
        :param image_size_x:
        :return:
        """
        try:
            _path_calib = os.path.join(self.config['path_archive', _date, 'calib'])
            if image_size_x == 256:
                dark_image = os.path.join(_path_calib, 'dark_{:s}4.fits'.format(str(_mode)))
            else:
                dark_image = os.path.join(_path_calib, 'dark_{:s}.fits'.format(str(_mode)))
            flat_image = os.path.join(_path_calib, 'flat_{:s}.fits'.format(_filt))

            if not os.path.exists(dark_image) or not os.path.exists(flat_image):
                return None, None
            else:
                with fits.open(dark_image) as dark, fits.open(flat_image) as flat:
                    # replace NaNs if necessary
                    if image_size_x == 256:
                        return np.nan_to_num(dark[0].data), np.nan_to_num(flat[0].data[384:640, 384:640])
                    else:
                        return np.nan_to_num(dark[0].data), np.nan_to_num(flat[0].data)
        except RuntimeError:
            # Failed? Make sure to mark calib.done in DB as False!
            self.db['coll_aux'].update_one(
                {'_id': _date},
                {
                    '$set': {
                        'calib.done': False,
                        'calib.raw.flat': [],
                        'calib.raw.dark': [],
                        'calib.last_modified': utc_now()
                    }
                }
            )

    @timeout(seconds_before_timeout=600)
    def calibration(self, _location, _date, _date_raw_data):
        """
            Handle calibration data

            It is monitored for timout

            Originally written by N. Law
        :param _location:
        :param _date:
        :param _date_raw_data:
        :return:
        """

        def sigma_clip_combine(img, out_fn, normalise=False, dark_bias_sub=False):
            """
                Combine all frames in FITS file img
            :param img:
            :param out_fn:
            :param normalise:
            :param dark_bias_sub:
            :return:
            """

            self.logger.debug('Making {:s}'.format(out_fn))
            # two passes:
            # 1/ generate average and RMS for each pixel
            # 2/ sigma-clipped average and RMS for each pixel
            # (i.e. only need to keep track of avg-sq and sq-avg arrays at any one time)
            sx = img[0].shape[1]
            sy = img[0].shape[0]

            avg = np.zeros((sy, sx), dtype=np.float32)
            avg_sq = np.zeros((sy, sx), dtype=np.float32)
            n_dps = np.zeros((sy, sx), dtype=np.float32)

            print("Image size:", sx, sy, len(img))
            self.logger.debug('Image size: {:d} {:d} {:d}'.format(sx, sy, len(img)))

            print("First pass")
            self.logger.debug("First pass")

            for i in img:
                avg += i.data
                avg_sq += i.data * i.data

            avg = avg / float(len(img))
            rms = np.sqrt((avg_sq / float(len(img))) - (avg * avg))

            # fits.PrimaryHDU(avg).writeto("avg.fits", overwrite=True)
            # fits.PrimaryHDU(rms).writeto("rms.fits", overwrite=True)

            sigma_clip_avg = np.zeros((sy, sx), dtype=np.float32)
            sigma_clip_n = np.zeros((sy, sx), dtype=np.float32)

            print("Second pass")
            self.logger.debug("Second pass")
            for i in img:
                sigma_mask = np.fabs((np.array(i.data, dtype=np.float32) - avg) / rms)

                sigma_mask[sigma_mask > 3.0] = 100
                sigma_mask[sigma_mask <= 1.0] = 1
                sigma_mask[sigma_mask == 100] = 0

                sigma_clip_avg += i.data * sigma_mask
                sigma_clip_n += sigma_mask

            sigma_clip_avg /= sigma_clip_n

            # set the average flat level to 1.0
            if normalise:
                sigma_clip_avg /= np.average(sigma_clip_avg[np.isfinite(sigma_clip_avg)])

            if dark_bias_sub:
                sigma_clip_avg -= np.average(sigma_clip_avg[sy - 50:sy, sx - 50:sx])

            fits.PrimaryHDU(sigma_clip_avg).writeto(out_fn, overwrite=True)
            self.logger.debug("Successfully made {:s}".format(out_fn))

        # path to zipped raw files
        path_date = os.path.join(_location, _date)

        # output dir
        _path_out = os.path.join(self.config['path']['path_archive'], _date, 'calib')

        # get calibration file names:
        pattern_fits = r'.fits.fz\Z'
        bias = [re.split(pattern_fits, s)[0] for s in _date_raw_data if re.match('bias_', s) is not None]
        flat = [re.split(pattern_fits, s)[0] for s in _date_raw_data if re.match('flat_', s) is not None]
        dark = [re.split(pattern_fits, s)[0] for s in _date_raw_data if re.match('dark_', s) is not None]

        # check in database if needs to be (re)done
        _select = self.db['coll_aux'].find_one({'_id': _date})

        # check folder modified date:
        time_tag = mdate_walk(_path_out)

        # calib data copied from other date?
        if _select['calib']['done'] and ('comment' in _select['calib']) and \
                ('Copied from' in _select['calib']['comment']):
            return {'status': 'ok', 'message': None}

        # not done or files changed?
        if (not _select['calib']['done']) or \
                (('flat' in _select['calib']['raw']) and (set(_select['calib']['raw']['flat']) != set(flat))) or \
                (('dark' in _select['calib']['raw']) and (set(_select['calib']['raw']['dark']) != set(dark))) or \
                (time_tag - _select['calib']['last_modified']).total_seconds() > 1.0:
            make_calib = True
        else:
            make_calib = False

        if make_calib:
            n_darks = len(dark)
            n_flats = len(flat)

            # enough data to make master calibration files?
            if n_darks > 9 and n_flats > 4:
                # update DB entry:
                self.db['coll_aux'].update_one(
                    {'_id': _date},
                    {
                        '$set': {
                            'calib.done': False,
                            'calib.raw.flat': flat,
                            'calib.raw.dark': dark,
                            'calib.last_modified': time_tag
                        }
                    }
                )

                # output dir exists?
                if not os.path.exists(_path_out):
                    os.makedirs(_path_out)

                # find the combine the darks:
                for d in dark:
                    # file name with extension
                    f = '{:s}.fits.fz'.format(d)
                    camera_mode = f.split('_')[1]
                    if camera_mode != '0':
                        # mode 0 is handled below

                        unzipped = os.path.join(self.config['path']['path_tmp'], f)

                        out_fn = os.path.join(_path_out, 'dark_{:s}.fits'.format(camera_mode))
                        with fits.open(unzipped) as img:
                            sigma_clip_combine(img, out_fn, dark_bias_sub=True)

                        # clean up after yoself!
                        try:
                            os.remove(unzipped)
                        except Exception as _e:
                            self.logger.error(_e)

                # generate the flats' dark
                # those files with mode '0' are the relevant ones:
                f = ['{:s}.fits.fz'.format(d) for d in dark if d.split('_')[1] == '0']
                unzipped = [os.path.join(self.config['path']['path_tmp'], _f) for _f in f]

                img = []
                for uz in unzipped:
                    img.append(fits.open(uz)[0])
                    # clean up after yoself!
                    try:
                        os.remove(uz)
                    except Exception as _e:
                        self.logger.error(_e)

                flat_dark_fn = os.path.join(_path_out, 'dark_0.fits')
                sigma_clip_combine(img, flat_dark_fn, dark_bias_sub=False)

                with fits.open(flat_dark_fn) as fdn:
                    flat_dark = np.array(fdn[0].data, dtype=np.float32)

                # make the flats:
                for filt in ["B", "g", "I", "r", "R", "U", "V"]:
                    print("Making {:s} flat".format(filt))
                    flats = []

                    f = ['{:s}.fits.fz'.format(_f) for _f in flat if _f.split('_')[2] == filt]
                    unzipped = [os.path.join(self.config['path']['path_tmp'], _f) for _f in f]

                    for uz in unzipped:

                        flt = fits.open(uz)[0]
                        flt.data = np.array(flt.data, dtype=np.float32)
                        flt.data -= flat_dark
                        # clean up after yoself!
                        try:
                            os.remove(uz)
                        except Exception as _e:
                            self.logger.error(_e)

                        flats.append(flt)

                    out_fn = os.path.join(_path_out, 'flat_{:s}.fits'.format(filt))
                    sigma_clip_combine(flats, out_fn, normalise=True)

                # success!
                # check new folder modified date:
                time_tag = mdate_walk(_path_out)
                # # update DB entry:
                # self.db['coll_aux'].update_one(
                #     {'_id': _date},
                #     {
                #         '$set': {
                #             'calib.done': True,
                #             'calib.last_modified': time_tag
                #         }
                #     }
                # )

                return {'status': 'ok',
                        'message': 'successfully generated master calibration files',
                        'db_record_update': ({'_id': _date},
                                             {
                                                 '$set': {
                                                     'calib.done': True,
                                                     'calib.last_modified': time_tag
                                                 }
                                             }
                                             )
                        }

            else:
                return {'status': 'error',
                        'message': 'No enough calibration files for {:s}'.format(_date)}
        else:
            return {'status': 'ok',
                    'message': None}

    # @timeout(seconds_before_timeout=600)
    def process_seeing(self, _path_in, _seeing_frames, _path_calib, _path_out,
                       _plate_scale=0.0351594, _fit_model='Gaussian2D', _box_size=100):
        # TODO!
        try:
            # parse observation name
            seeing_obs = KPEDObservation(_id='9999_' + _seeing_frames[0], _config=self.config)
            _filt, _date_utc = seeing_obs.db_entry['filter'], seeing_obs.db_entry['date_utc']
            # get fits header
            fits_header = get_fits_header(os.path.join(_path_in, '{:s}.fits.fz'.format(_seeing_frames[0])))
            _mode, _exp = str(int(fits_header['MODE_NUM'][0])), fits_header['EXPOSURE'][0]
            # print(_filt, _date_utc, _mode, _exp)

            # get total number of frames to allocate
            # number of frames in each fits file
            n_frames_files = []
            for jj, _file in enumerate(_seeing_frames):
                with fits.open(os.path.join(_path_in, '{:s}.fits.fz'.format(_file)), memmap=True) as _hdulist:
                    if jj == 0:
                        # get image size (this would be (1024, 1024) for the Andor camera)
                        image_size = _hdulist[0].shape
                    n_frames_files.append(len(_hdulist))
                    # bar.update(iterations=files_sizes[jj])
            # total number of frames
            nf = sum(n_frames_files)

            # Stack to seeing-limited image
            summed_seeing_limited_frame = np.zeros(image_size, dtype=np.float)
            for jj, _file in enumerate(_seeing_frames):
                # print(jj)
                with fits.open(os.path.join(_path_in, '{:s}.fits.fz'.format(_file)), memmap=True) as _hdulist:
                    for ii, _ in enumerate(_hdulist):
                        try:
                            summed_seeing_limited_frame += np.nan_to_num(_hdulist[ii].data)
                        except Exception as _e:
                            print(_e)
                            continue

            # check if there are data to be processed:
            if np.abs(np.max(summed_seeing_limited_frame)) < 1e-9:  # only zeros in summed_seeing_limited_frame
                raise Exception('No data in the cube to be processed.')

            # load darks and flats
            dark, flat = KPEDPipeline.load_darks_and_flats(_path_calib, _mode, _filt, image_size[0])
            if dark is None or flat is None:
                raise Exception('Could not open darks and flats')

            summed_seeing_limited_frame = KPEDRegistrationPipeline.calibrate_frame(summed_seeing_limited_frame / nf,
                                                                                   dark, flat, _iter=2)
            summed_seeing_limited_frame = gaussian_filter(summed_seeing_limited_frame, sigma=1)

            # remove cosmic rays:
            # print('removing cosmic rays from the seeing limited image')
            summed_seeing_limited_frame = \
            lax.lacosmicx(np.ascontiguousarray(summed_seeing_limited_frame, dtype=np.float32),
                          sigclip=20, sigfrac=0.3, objlim=5.0,
                          gain=1.0, readnoise=6.5, satlevel=65536.0, pssl=0.0, niter=4,
                          sepmed=True, cleantype='meanmask', fsmode='median',
                          psfmodel='gauss', psffwhm=2.5, psfsize=7, psfk=None,
                          psfbeta=4.765, verbose=False)[1]

            # dump fits for sextraction:
            _fits_stacked = '{:s}.summed.fits'.format(_seeing_frames[0])

            export_fits(os.path.join(_path_in, _fits_stacked), summed_seeing_limited_frame)

            _, x, y = KPEDPipeline.trim_frame(_path_in, _fits_name=_fits_stacked,
                                              _win=_box_size, _method='sextractor', _x=None, _y=None, _drizzled=False)
            print('centroid position: ', x, y)

            # remove fits:
            os.remove(os.path.join(_path_in, _fits_stacked))

            centroid = Star(x, y, summed_seeing_limited_frame, model_type=_fit_model, box=_box_size,
                            plate_scale=_plate_scale, exp=_exp, out_path=os.path.join(_path_out, 'seeing'))
            seeing, seeing_x, seeing_y = centroid.fwhm
            print('Estimated seeing = {:.3f} pixels'.format(seeing))
            print('Estimated seeing = {:.3f}\"'.format(seeing * _plate_scale))

            if seeing < 10:
                print('Estimated seeing suspiciously small, discarding.')
                return _date_utc, None, None, _filt, None, None, _exp
            else:
                # plot image, model, and residuals:
                # print('plotting seeing preview, see', os.path.join(_path_out, 'seeing'))
                centroid.plot_resulting_model(frame_name=_seeing_frames[0])
                # print('done')

                return _date_utc, seeing * _plate_scale, seeing, _filt, \
                       seeing_x * _plate_scale, seeing_y * _plate_scale, _exp

        except Exception as _e:
            print(_e)
            traceback.print_exc()
            try:
                if os.path.exists(os.path.join(_path_in, _fits_stacked)):
                    # remove fits:
                    os.remove(os.path.join(_path_in, _fits_stacked))
                return _date_utc, None, None, _filt, None, None, _exp
            except Exception as _e:
                print(_e)
                traceback.print_exc()
                return None, None, None, None, None, None, None

    @timeout(seconds_before_timeout=600)
    def auxiliary(self, _location, _date, _date_raw_data):
        """
        TODO:
            Handle auxiliary data

            It is monitored for time out
        :param _location:
        :param _date:
        :param _date_raw_data:
        :return:
        """

        return {'status': 'ok', 'message': None}

        # DB entry:
        _select = self.db['coll_aux'].find_one({'_id': _date}, max_time_ms=5000)

        _path_out = os.path.join(self.config['path']['path_archive'], _date, 'summary')

        # get calibration file names:
        pattern_fits = r'.fits.fz\Z'
        pattern_end = r'.[0-9]{6}.fits.fz\Z'
        date_seeing = [re.split(pattern_fits, s)[0] for s in _date_raw_data
                       if re.search(pattern_end, s) is not None and
                       re.match('seeing_', s) is not None]

        # for each observation, count number of fits files with data:
        date_seeing_num_fits = [np.count_nonzero([s in df for df in _date_raw_data]) for s in date_seeing]
        # print(date_seeing_num_fits)
        # stack with obs names:
        _seeing_frames = list(zip(date_seeing, date_seeing_num_fits))

        ''' check/do seeing '''
        last_modified = _select['seeing']['last_modified'].replace(tzinfo=pytz.utc)
        # path to store unzipped raw files
        _path_tmp = self.config['path']['path_tmp']
        # path to raw seeing data:
        _path_seeing = os.path.join(_location, _date)
        # path to calibration data:
        _path_calib = os.path.join(self.config['path']['path_archive'], _date, 'calib')

        # _seeing_frames = _select['seeing']['frames']
        # deal with long seeing observations properly
        _seeing_raws = []
        # unzipped file names:
        _obsz = []
        for _s in _seeing_frames:
            _s_raws = ['{:s}.fits.fz'.format(_s[0])]
            _s_obsz = [_s[0]]
            for _si in range(_s[1] - 1):
                _s_raws.append('{:s}_{:d}.fits.fz'.format(_s[0], _si))
                _s_obsz.append('{:s}_{:d}'.format(_s[0], _si))
            _seeing_raws.append(_s_raws)
            _obsz.append(_s_obsz)

        _seeing_data = [[_s[0], None, None, None, None, None, None] for _s in _seeing_frames]

        # get plate scale:
        telescope = 'KPNO_2.1m'

        # this is not drizzled!
        plate_scale = self.config['telescope'][telescope]['scale']

        if len(_seeing_raws) > 0:
            try:
                time_tags = [max([datetime.datetime.utcfromtimestamp(os.stat(os.path.join(_path_seeing, _ss)).st_mtime)
                                  for _ss in _s]) for _s in _seeing_raws]
                time_tag = max(time_tags)
                time_tag = time_tag.replace(tzinfo=pytz.utc)

                # not done or new files appeared in the raw directory
                if ('seeing' not in _select) or (
                        (not _select['seeing']['done'] or (time_tag - last_modified).total_seconds() > 1.0) and \
                        (_select['seeing']['retries'] <= self.config['misc']['max_retries'])):

                    seeing_plot = []
                    for ii, _obs in enumerate(_obsz):
                        print('processing {:s}'.format(str(_obs)))
                        # this returns datetime, seeing in " and in pix, and used filter:
                        _date_utc, seeing, _, _filt, seeing_x, seeing_y, _exp = \
                            self.process_seeing(_path_in=_path_tmp, _seeing_frames=_obs,
                                                _path_calib=_path_calib, _path_out=_path_out,
                                                _plate_scale=plate_scale,
                                                _fit_model=self.config['pipeline']['seeing']['fit_model'],
                                                _box_size=self.config['pipeline']['seeing']['win'])
                        if seeing is not None:
                            seeing_plot.append([_date_utc, seeing, _filt, _exp])
                        _seeing_data[ii][1] = _date_utc
                        _seeing_data[ii][2] = _filt
                        _seeing_data[ii][3] = seeing
                        _seeing_data[ii][4] = seeing_x
                        _seeing_data[ii][5] = seeing_y
                        _seeing_data[ii][6] = _exp

                    # generate summary plot for the whole night:
                    if len(seeing_plot) > 0:
                        # import matplotlib
                        # matplotlib.use('Agg')
                        import matplotlib.pyplot as plt

                        seeing_plot = np.array(seeing_plot)
                        # sort by time stamp:
                        seeing_plot = seeing_plot[seeing_plot[:, 0].argsort()]

                        # filter colors on the plot:
                        filter_colors = {'lp600': plt.cm.Blues(0.82),
                                         'Sg': plt.cm.Greens(0.7),
                                         'Sr': plt.cm.Reds(0.7),
                                         'Si': plt.cm.Oranges(0.7),
                                         'Sz': plt.cm.Oranges(0.5)}

                        plt.close('all')
                        fig = plt.figure('Seeing data for {:s}'.format(_date), figsize=(8, 3), dpi=200)
                        ax = fig.add_subplot(111)

                        # all filters used that night:
                        filters_used = set(seeing_plot[:, 2])

                        # plot different filters in different colors
                        for filter_used in filters_used:
                            fc = filter_colors[filter_used] if filter_used in filter_colors else plt.cm.Greys(0.7)

                            # mask = seeing_plot[:, 2] == filter_used

                            # short exposures:
                            mask = np.all(np.vstack((seeing_plot[:, 2] == filter_used,
                                                     seeing_plot[:, 3] <= 15.0)), axis=0)
                            # print(filter_used, mask)
                            if np.count_nonzero(mask) > 0:
                                ax.plot(seeing_plot[mask, 0], seeing_plot[mask, 1], '.',
                                        c=fc, markersize=8, label='{:s}, short exp'.format(filter_used))

                            # long exposures:
                            mask = np.all(np.vstack((seeing_plot[:, 2] == filter_used,
                                                     seeing_plot[:, 3] > 15.0)), axis=0)
                            # print(filter_used, mask)
                            if np.count_nonzero(mask) > 0:
                                ax.plot(seeing_plot[mask, 0], seeing_plot[mask, 1], '.',
                                        c=fc, markersize=12, label='{:s}, long exp'.format(filter_used))

                        ax.set_ylabel('Seeing, arcsec')  # , fontsize=18)
                        ax.grid(linewidth=0.5)

                        try:
                            # make a robust fit to seeing data for visual reference
                            t_seeing_plot = np.array([(_t - seeing_plot[0, 0]).total_seconds()
                                                      for _t in seeing_plot[:, 0]])
                            t_seeing_plot = np.expand_dims(t_seeing_plot, axis=1)
                            estimators = [('RANSAC', linear_model.RANSACRegressor()), ]
                            for name, estimator in estimators:
                                model = make_pipeline(PolynomialFeatures(degree=5), estimator)
                                model.fit(t_seeing_plot, seeing_plot[:, 1])
                                y_plot = model.predict(t_seeing_plot)
                                # noinspection PyUnboundLocalVariable
                                ax.plot(seeing_plot[:, 0], y_plot, '--', c=plt.cm.Blues(0.4),
                                        linewidth=1, label='Robust {:s} fit'.format(name), clip_on=True)
                        except Exception as _e:
                            print(_e)
                            traceback.print_exc()

                        myFmt = mdates.DateFormatter('%H:%M')
                        ax.xaxis.set_major_formatter(myFmt)
                        fig.autofmt_xdate()

                        # make sure our 'robust' fit didn't spoil the scale:
                        ax.set_ylim([np.min(seeing_plot[:, 1]) * 0.9, np.max(seeing_plot[:, 1]) * 1.1])
                        dt = datetime.timedelta(seconds=(seeing_plot[-1, 0] -
                                                         seeing_plot[0, 0]).total_seconds() * 0.05)
                        ax.set_xlim([seeing_plot[0, 0] - dt, seeing_plot[-1, 0] + dt])

                        # add legend:
                        ax.legend(loc='best', numpoints=1, fancybox=True, prop={'size': 6})

                        plt.tight_layout()

                        # plt.show()
                        f_seeing_plot = os.path.join(_path_out, 'seeing.{:s}.png'.format(_date))
                        fig.savefig(f_seeing_plot, dpi=300)

                    # update database record:
                    s = {'status': 'ok',
                         'message': 'successfully processed seeing data for {:s}'.format(_date),
                         'db_record_update': ({'_id': _date},
                                              {
                                                  '$set': {
                                                      'seeing.done': True,
                                                      'seeing.frames': _seeing_data,
                                                      'seeing.last_modified': time_tag
                                                  },
                                                  '$inc': {
                                                      'seeing.retries': 1
                                                  }
                                              }
                                              )
                         }
                else:
                    s = {'status': 'ok', 'message': None}

            except Exception as _e:
                print(_e)
                traceback.print_exc()
                try:
                    # clean up stuff
                    seeing_summary_plot = os.path.join(_path_out, 'seeing.{:s}.png'.format(_date))
                    if os.path.exists(seeing_summary_plot):
                        os.remove(seeing_summary_plot)
                    individual_frames_path = os.path.join(_path_out, 'seeing')
                    for individual_frame in os.listdir(individual_frames_path):
                        if os.path.exists(os.path.join(individual_frames_path, individual_frame)):
                            os.remove(os.path.join(individual_frames_path, individual_frame))
                finally:
                    s = {'status': 'error',
                         'message': 'failed to process seeing data for {:s}'.format(_date),
                         'db_record_update': ({'_id': _date},
                                              {
                                                  '$set': {
                                                      'seeing.done': False,
                                                      'seeing.frames': [],
                                                      'seeing.last_modified': utc_now()
                                                  },
                                                  '$inc': {
                                                      'seeing.retries': 1
                                                  }
                                              }
                                              )
                         }

            finally:
                # remove unzipped files
                _seeing_raws_unzipped = [os.path.splitext(_f)[0] for _f in chain.from_iterable(_seeing_raws)]
                for _seeing_raw_unzipped in _seeing_raws_unzipped:
                    if os.path.exists(os.path.join(_path_tmp, _seeing_raw_unzipped)):
                        os.remove(os.path.join(_path_tmp, _seeing_raw_unzipped))
        else:
            s = {'status': 'ok', 'message': None}

        return s


class Observation(object):
    """
        This is mainly to show the tentative structure for future use cases
    """
    def __init__(self, _id=None, _aux=None):
        """
            Initialize Observation object
        :param _id:
        :param _aux:
        :return:
        """
        assert _id is not None, 'Must specify unique obs id'
        # obs unique id:
        self.id = _id

        self.aux = _aux

    def parse(self, **kwargs):
        """
            Parse obs info (e.g. contained in id, or fits header) to be injected into DB
            Define decision chain for observation pipelining
        :return:
        """
        raise NotImplementedError

    def init_db_entry(self):
        """
            Initialize DB entry
        :return:
        """
        raise NotImplementedError

    def check_db_entry(self):
        """
            Check if DB entry reflects reality
        :return:
        """
        raise NotImplementedError

    def get_task(self, **kwargs):
        """
            Construct decision chain
        :return:
        """
        raise NotImplementedError


class KPEDObservation(Observation):
    def __init__(self, _id=None, _aux=None, _program_pi=None, _db_entry=None, _config=None):
        """
            Initialize Observation object
        :param _id:
        :param _aux:
        :return:
        """
        ''' initialize super class '''
        super(KPEDObservation, self).__init__(_id=_id, _aux=_aux)

        # current DB entry
        if _db_entry is None:
            # db entry does not exist?
            # parse obs name
            _obs_info = self.parse(_program_pi)
            # create "empty" record:
            self.db_entry = self.init_db_entry()
            # populate with basic info:
            for k in _obs_info:
                self.db_entry[k] = _obs_info[k]
        else:
            self.db_entry = _db_entry

        # print(self.db_entry)
        # pass on the config
        assert _config is not None, 'must pass config to KPEDObservation ' + _id
        self.config = _config

    def check_db_entry(self):
        """
            Check if DB entry reflects reality

        :return:
        """
        _date = self.db_entry['date_utc'].strftime('%Y%m%d')

        # TODO?
        # TODO: instead, at startup, mark all enqueued things failed

        return {'status': 'ok', 'message': None}

    def check_aux(self):
        # TODO?
        _date = self.db_entry['date_utc'].strftime('%Y%m%d')

        return {'status': 'ok', 'message': None}

    def get_task(self):
        """
            Figure out what needs to be done with the observation.
            Here is where the processing decision chain is defined
        :return:
        """
        _task = None

        # Registration?
        ''' Registration pipeline '''
        pipe = KPEDRegistrationPipeline(_config=self.config, _db_entry=self.db_entry)
        # check conditions necessary to run (defined in config.json):
        go = pipe.check_necessary_conditions()
        # print('{:s} RP go: '.format(self.id), go)

        # good to go?
        if go:
            # should and can run BSP pipeline itself?
            _part = 'registration_pipeline'
            go = pipe.check_conditions(part=_part)
            if go:
                # mark enqueued
                pipe.db_entry['pipelined']['{:s}'.format(pipe.name)]['status']['enqueued'] = True
                # pipe.db_entry['pipelined']['{:s}'.format(pipe.name)]['last_modified'] = utc_now()
                _task = {'task': _part, 'id': self.id, 'config': self.config, 'db_entry': pipe.db_entry,
                         'db_record_update': ({'_id': self.id},
                                              {'$set': {
                                                  'pipelined.{:s}'.format(pipe.name):
                                                      pipe.db_entry['pipelined']['{:s}'.format(pipe.name)]
                                              }}
                                              )
                         }
                return _task

            # should and can run preview generation for BSP pipeline itself?
            _part = 'registration_pipeline:preview'
            go = pipe.check_conditions(part=_part)
            # print(self.id, _part, go)
            if go:
                _task = {'task': _part, 'id': self.id, 'config': self.config, 'db_entry': pipe.db_entry}
                return _task

        # TODO:
        # Photometry?
        ''' Photometry pipeline '''

        # Photometry?
        ''' Astrometry pipeline '''

        return _task

    def check_raws(self, _location, _date, _date_raw_data):
        """
            Check if raw data files info in DB is correct and up-to-date
        :param _location:
        :param _date:
        :param _date_raw_data:
        :return:
        """
        try:
            # raw file names
            _raws = [_s for _s in _date_raw_data if re.match(re.escape(self.id), _s) is not None]
            # deleted?! unset pipelined as well
            if len(_raws) == 0:
                self.db_entry['raw_data']['location'] = []
                self.db_entry['raw_data']['data'] = []
                self.db_entry['raw_data']['last_modified'] = utc_now()
                return {'status': 'error', 'message': 'raw files for {:s} not available any more'.format(self.id),
                        'db_record_update': ({'_id': self.id},
                                             {
                                                 '$set': {
                                                     # 'exposure': None,
                                                     # 'magnitude': None,
                                                     # 'fits_header': {},
                                                     # 'coordinates': {},
                                                     'raw_data.location': [],
                                                     'raw_data.data': [],
                                                     'raw_data.last_modified':
                                                         self.db_entry['raw_data']['last_modified']
                                                 },
                                                 '$unset': {
                                                     'pipelined': 1
                                                 }
                                             })
                        }
            # time tags. use the 'freshest' time tag for 'last_modified'
            time_tags = [datetime.datetime.utcfromtimestamp(os.stat(os.path.join(_location, _date, _s)).st_mtime)
                         for _s in _raws]
            time_tag = max(time_tags)

            # init/changed? the archiver will have to update database entry then:
            if (len(self.db_entry['raw_data']['data']) == 0) or \
                    (abs((time_tag - self.db_entry['raw_data']['last_modified']).total_seconds()) > 1.0):
                self.db_entry['raw_data']['location'] = ['{:s}:{:s}'.format(
                                                            self.config['server']['analysis_machine_external_host'],
                                                            self.config['server']['analysis_machine_external_port']),
                                                            _location]
                self.db_entry['raw_data']['data'] = sorted(_raws)
                self.db_entry['raw_data']['last_modified'] = time_tag

                # additionally, fetch FITS header and parse some info from there
                # grab header from the last file, as it's usually the smallest.
                fits_header = get_fits_header(os.path.join(self.db_entry['raw_data']['location'][1],
                                                           self.db_entry['date_utc'].strftime('%Y%m%d'),
                                                           self.db_entry['raw_data']['data'][-1]))
                # save fits header
                self.db_entry['fits_header'] = fits_header

                # separately, parse exposure and magnitude
                self.db_entry['exposure'] = float(fits_header['EXPOSURE'][0]) if ('EXPOSURE' in fits_header) else None
                self.db_entry['magnitude'] = float(fits_header['MAGNITUD'][0]) if ('MAGNITUD' in fits_header) else None

                # Get and parse coordinates
                _ra_str = fits_header['TELRA'][0] if 'TELRA' in fits_header else None
                _objra = fits_header['OBJRA'][0] if 'OBJRA' in fits_header else None
                if _objra is not None:
                    for letter in ('h, m'):
                        _objra = _objra.replace(letter, ':')
                    _objra = _objra[:-1]
                # print(_objra)
                # TELRA not available? try replacing with OBJRA:
                if _ra_str is None:
                    _ra_str = _objra

                _dec_str = fits_header['TELDEC'][0] if 'TELDEC' in fits_header else None
                _objdec = fits_header['OBJDEC'][0] if 'OBJDEC' in fits_header else None
                if _objdec is not None:
                    for letter in ('d, m'):
                        _objdec = _objdec.replace(letter, ':')
                    _objdec = _objdec[:-1]
                # print(_objdec)
                # TELDEC not available? try replacing with OBJDEC:
                if _dec_str is None:
                    _dec_str = _objdec

                _az_str = str(fits_header['AZIMUTH'][0]) if 'AZIMUTH' in fits_header else None
                _el_str = str(fits_header['ELVATION'][0]) if 'ELVATION' in fits_header else None
                _epoch = float(fits_header['EQUINOX'][0]) if 'EQUINOX' in fits_header else 2000.0

                if None in (_ra_str, _dec_str):
                    _azel = None
                    _radec = None
                    _radec_str = None
                    _radec_deg = None
                else:
                    if not ('9999' in _ra_str or '9999' in _dec_str):
                        # string format: H:M:S, D:M:S
                        _radec_str = [_ra_str, _dec_str]
                        # the rest are floats [rad]
                        _ra, _dec = radec_str2rad(_ra_str, _dec_str)
                        _radec = [_ra, _dec]
                        # for GeoJSON, must be lon:[-180, 180], lat:[-90, 90] (i.e. in deg)
                        _radec_deg = [_ra * 180.0 / np.pi - 180.0, _dec * 180.0 / np.pi]
                        if (None not in (_az_str, _el_str)) and ('9999' not in (_az_str, _el_str)):
                            _azel = [float(_az_str) * np.pi / 180., float(_el_str) * np.pi / 180.]
                        else:
                            _azel = None
                    else:
                        _azel = None
                        _radec = None
                        _radec_str = None
                        _radec_deg = None

                self.db_entry['coordinates']['epoch'] = _epoch
                self.db_entry['coordinates']['radec_str'] = _radec_str
                self.db_entry['coordinates']['radec_geojson'] = {'type': 'Point', 'coordinates': _radec_deg}
                self.db_entry['coordinates']['radec'] = _radec
                self.db_entry['coordinates']['azel'] = _azel

                # DB updates are handled by the main archiver process
                # we'll provide it with proper query to feed into pymongo's update_one()
                return {'status': 'ok', 'message': 'raw files changed',
                        'db_record_update': ({'_id': self.id},
                                             {
                                                 '$set': {
                                                    'exposure': self.db_entry['exposure'],
                                                    # 'magnitude': self.db_entry['magnitude'],
                                                    'fits_header': self.db_entry['fits_header'],
                                                    'coordinates': self.db_entry['coordinates'],
                                                    'raw_data.location': self.db_entry['raw_data']['location'],
                                                    'raw_data.data': self.db_entry['raw_data']['data'],
                                                    'raw_data.last_modified': time_tag
                                                 },
                                                 '$unset': {
                                                     'pipelined.*': 1
                                                 }
                                             }
                                             )
                        }
            else:
                return {'status': 'ok', 'message': None}

        except Exception as _e:
            traceback.print_exc()
            return {'status': 'error', 'message': str(_e)}

    def check_distributed(self):
        try:
            # TODO:
            return {'status': 'ok', 'message': None}

        except Exception as _e:
            traceback.print_exc()
            return {'status': 'error', 'message': 'Checking distributed status failed for {:s}'.format(self.id),
                    'db_record_update': ({'_id': self.id},
                                         {
                                             '$set': {
                                                 'distributed.status': False,
                                                 'distributed.location': [],
                                                 'distributed.last_modified': utc_now()
                                             }
                                         }
                                         )
                    }

    def parse(self, _program_pi):
        """
            Parse obs info (e.g. contained in id, or fits header) to be injected into DB

        :param _program_pi: dict program_num -> PI

        :return:
        """
        _obs = self.id
        # parse name:
        _tmp = _obs.split('_')
        # TODO: if we add program nums in the future
        # _prog_num = str(_tmp[0])
        _prog_num = str(0)

        # who's pi?
        if (_program_pi is not None) and (_prog_num in _program_pi.keys()):
            _prog_pi = str(_program_pi[_prog_num])
        else:
            # play safe if pi's unknown:
            _prog_pi = ['admin']
        # stack name together if necessary (if contains underscores):
        _sou_name = '_'.join(_tmp[0:-5])
        # code of the filter used:
        _filt = _tmp[-5:-4][0]
        # date and time of obs:
        _date_utc = datetime.datetime.strptime(_tmp[-3] + _tmp[-2], '%Y%m%d%H%M%S.%f')
        # marker:
        _marker = _tmp[-3:-2][0]

        # telescope:
        _telescope = 'KPNO_2.1m'

        return {
            'science_program': {
                'program_id': _prog_num,
                'program_PI': _prog_pi
            },
            'name': _sou_name,
            'filter': _filt,
            'date_utc': _date_utc,
            'marker': _marker,
            'telescope': _telescope
        }

    def init_db_entry(self):
        """
                An empty database record for a science observation
            :return:
            """
        time_now_utc = utc_now()
        return {
            '_id': self.id,
            'date_added': time_now_utc,
            'name': None,
            'alternative_names': [],
            'science_program': {
                'program_id': None,
                'program_PI': None
            },
            'date_utc': None,
            'telescope': None,
            'filter': None,
            'exposure': None,
            'coordinates': {
                'epoch': None,
                'radec': None,
                'radec_str': None,
                'azel': None
            },
            'fits_header': {},

            'pipelined': {},

            'distributed': {
                'status': False,
                'location': [],
                'last_modified': time_now_utc
            },
            'raw_data': {
                'location': [],
                'data': [],
                'last_modified': time_now_utc
            },
            'comment': None
        }


class Pipeline(object):
    def __init__(self, _config, _db_entry):
        """
            Pipeline
        :param _config: dict with configuration
        :param _db_entry: observation DB entry
        """
        self.config = _config
        self.db_entry = _db_entry

    def check_necessary_conditions(self):
        """
            Check if should be run on an obs
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def init_status():
        """
            Initialize status dict
        :return:
        """
        raise NotImplementedError

    def run(self, part):
        """
            Run the pipeline
            # :param aux: auxiliary data including calibration
        :return:
        """
        raise NotImplementedError

    def generate_preview(self, **kwargs):
        """
            Generate preview images
        :return:
        """
        raise NotImplementedError


class KPEDPipeline(Pipeline):
    def __init__(self, _config, _db_entry):
        """
            Pipeline
        :param _config: dict with configuration
        :param _db_entry: observation DB entry
        """
        ''' initialize super class '''
        super(KPEDPipeline, self).__init__(_config=_config, _db_entry=_db_entry)

        ''' figure out where we are '''
        self.telescope = 'KPNO_2.1m'

    def check_necessary_conditions(self):
        """
            Check if should be run on an obs
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def init_status():
        """
            Initialize status dict
        :return:
        """
        raise NotImplementedError

    def run(self, part):
        """
            Run the pipeline
            # :param aux: auxiliary data including calibration
        :return:
        """
        raise NotImplementedError

    def generate_preview(self, **kwargs):
        """
            Generate preview images
        :return:
        """
        raise NotImplementedError

    def generate_pca_preview(self, **kwargs):
        """
            Generate preview images
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def shift2d(fftn, ifftn, data, deltax, deltay, xfreq_0, yfreq_0,
                return_abs=False, return_real=True):
        """
        2D version: obsolete - use ND version instead
        (though it's probably easier to parse the source of this one)

        FFT-based sub-pixel image shift.
        Will turn NaNs into zeros

        Shift Theorem:

        .. math::
            FT[f(t-t_0)](x) = e^{-2 \pi i x t_0} F(x)


        Parameters
        ----------
        data : np.ndarray
            2D image
        """

        xfreq = deltax * xfreq_0
        yfreq = deltay * yfreq_0
        freq_grid = xfreq + yfreq

        kernel = np.exp(-1j * 2 * np.pi * freq_grid)

        result = ifftn(fftn(data) * kernel)

        if return_real:
            return np.real(result)
        elif return_abs:
            return np.abs(result)
        else:
            return result

    @staticmethod
    def image_center(_path, _fits_name, _x0=None, _y0=None, _win=None):

        # extract sources
        sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "XPEAK_IMAGE", "YPEAK_IMAGE",
                                "A_IMAGE", "B_IMAGE", "FWHM_IMAGE", "FLAGS", "FLAGS_WEIGHT",
                                "FLUX_AUTO", "FLUXERR_AUTO", "FLUX_RADIUS"],
                        config={"DETECT_MINAREA": 5, "PHOT_APERTURES": "10", 'DETECT_THRESH': '5.0'},
                        sexpath="sex")

        out = sew(os.path.join(_path, _fits_name))
        # sort by FWHM
        out['table'].sort('FLUX_AUTO')
        # descending order: first is brightest
        out['table'].reverse()

        # print(out['table'])  # This is an astropy table.

        # get first 10 and score them:
        scores = []
        # search everywhere in the image?
        if _x0 is None and _y0 is None and _win is None:
            # maximum error of a Gaussian fit. Real sources usually have larger 'errors'
            gauss_error_max = [np.max([sou['A_IMAGE'] for sou in out['table'][0:10]]),
                               np.max([sou['B_IMAGE'] for sou in out['table'][0:10]])]
            for sou in out['table'][0:10]:
                if sou['FWHM_IMAGE'] > 1:
                    score = (log_gauss_score(sou['FWHM_IMAGE']) +
                             gauss_score(rho(sou['X_IMAGE'], sou['Y_IMAGE'])) +
                             np.mean([sou['A_IMAGE'] / gauss_error_max[0],
                                      sou['B_IMAGE'] / gauss_error_max[1]])) / 3.0
                else:
                    score = 0  # it could so happen that reported FWHM is 0
                scores.append(score)
        # search around (_x0, _y0) in a window of width _win
        else:
            for sou in out['table'][0:10]:
                _r = rho(sou['X_IMAGE'], sou['Y_IMAGE'], x_0=_x0, y_0=_y0)
                if sou['FWHM_IMAGE'] > 1 and _r < _win:
                    score = gauss_score(_r)
                else:
                    score = 0  # it could so happen that reported FWHM is 0
                scores.append(score)

        # there was something to score? get the best score then
        if len(scores) > 0:
            best_score = np.argmax(scores)
            x_center = out['table']['YPEAK_IMAGE'][best_score]
            y_center = out['table']['XPEAK_IMAGE'][best_score]
        # somehow no sources detected? but _x0 and _y0 set? return the latter then
        elif _x0 is not None and _y0 is not None:
            x_center, y_center = _x0, _y0
        # no sources detected and _x0 and _y0 not set? return the simple maximum:
        else:
            scidata = fits.open(os.path.join(_path, _fits_name))[0].data
            x_center, y_center = np.unravel_index(scidata.argmax(), scidata.shape)

        return x_center, y_center

    # @timeout(seconds_before_timeout=10)
    @staticmethod
    def load_darks_and_flats(_path_calib, _mode, _filt, image_size_x=1024):
        """
            Load darks and flats
        :param _date:
        :param _mode:
        :param _filt:
        :param image_size_x:
        :return:
        """

        if image_size_x == 256:
            # quarter frame observing mode
            dark_image = os.path.join(_path_calib, 'dark_{:s}4.fits'.format(str(_mode)))
        else:
            dark_image = os.path.join(_path_calib, 'dark_{:s}.fits'.format(str(_mode)))
        flat_image = os.path.join(_path_calib, 'flat_{:s}.fits'.format(_filt))

        if not os.path.exists(dark_image):
            raise Exception('Could not find calibration {:s} in {:s}'.format(dark_image, _path_calib))
        elif not os.path.exists(flat_image):
            raise Exception('Could not find calibration {:s} in {:s}'.format(flat_image, _path_calib))
        else:
            with fits.open(dark_image) as dark, fits.open(flat_image) as flat:
                # replace NaNs if necessary
                if image_size_x == 256:
                    return np.nan_to_num(dark[0].data), np.nan_to_num(flat[0].data[384:640, 384:640])
                else:
                    return np.nan_to_num(dark[0].data), np.nan_to_num(flat[0].data)

    @staticmethod
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
            return exposure.rescale_intensity(scidata, in_range=(p_1, p_2))
        elif correction == 'local':
            # perform local histogram equalization instead:
            return exposure.equalize_adapthist(scidata, clip_limit=0.03)
        else:
            raise Exception('Contrast correction option not recognized')

    @staticmethod
    def get_xy_from_shifts_txt(_path):
        with open(os.path.join(_path, 'shifts.txt')) as _f:
            f_lines = _f.readlines()
        # skip empty lines (if accidentally present in the file)
        f_lines = [_l for _l in f_lines if len(_l) > 1]

        _tmp = f_lines[0].split()
        x_lock, y_lock = int(_tmp[-2]), int(_tmp[-1])

        return x_lock, y_lock

    @classmethod
    def trim_frame(cls, _path, _fits_name, _win=100, _method='sextractor', _x=None, _y=None, _drizzled=True):
        """
            Crop image around a star, which is detected by one of the _methods
            (e.g. SExtracted and rated)

        :param _path: path
        :param _fits_name: fits-file name
        :param _win: window width
        :param _method: from 'frames.txt' (if this is the output of the standard lucky pipeline),
                        from 'pipeline_settings.txt' (if this is the output of the standard lucky pipeline),
                        from 'shifts.txt' (if this is the output of the faint pipeline),
                        using 'sextractor', a simple 'max', or 'manual'
        :param _x: source x position -- if known in advance
        :param _y: source y position -- if known in advance
        :param _drizzled: was it drizzled?

        :return: image, cropped around a lock position and the lock position itself
        """
        with fits.open(os.path.join(_path, _fits_name)) as _hdu:
            scidata = np.nan_to_num(_hdu[0].data)

        if _method == 'sextractor':
            # extract sources
            sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "XPEAK_IMAGE", "YPEAK_IMAGE",
                                    "A_IMAGE", "B_IMAGE", "FWHM_IMAGE", "FLAGS", "FLAGS_WEIGHT",
                                    "FLUX_AUTO", "FLUXERR_AUTO", "FLUX_RADIUS"],
                            config={"DETECT_MINAREA": 8, "PHOT_APERTURES": "10", 'DETECT_THRESH': '5.0'},
                            sexpath="sex")

            out = sew(os.path.join(_path, _fits_name))
            # sort by FWHM
            out['table'].sort('FLUX_AUTO')
            # descending order: brightest first
            out['table'].reverse()

            # print(out['table'])  # This is an astropy table.

            # get first 10 and score them:
            scores = []
            # maximum error of a Gaussian fit. Real sources usually have larger 'errors'
            gauss_error_max = [np.max([sou['A_IMAGE'] for sou in out['table'][0:10]]),
                               np.max([sou['B_IMAGE'] for sou in out['table'][0:10]])]
            for sou in out['table'][0:10]:
                if sou['FWHM_IMAGE'] > 1:
                    score = (log_gauss_score(sou['FWHM_IMAGE']) +
                             gauss_score(rho(sou['X_IMAGE'], sou['Y_IMAGE'])) +
                             np.mean([sou['A_IMAGE'] / gauss_error_max[0],
                                      sou['B_IMAGE'] / gauss_error_max[1]])) / 3.0
                else:
                    score = 0  # it could so happen that reported FWHM is 0
                scores.append(score)

            # print('scores: ', scores)

            N_sou = len(out['table'])
            # do not crop large planets and crowded fields
            if N_sou != 0 and N_sou < 30:
                # sou_xy = [out['table']['X_IMAGE'][0], out['table']['Y_IMAGE'][0]]
                best_score = np.argmax(scores) if len(scores) > 0 else 0
                # window size not set? set it automatically based on source fwhm
                if _win is None:
                    sou_size = np.max((int(out['table']['FWHM_IMAGE'][best_score] * 3), 100))
                    _win = sou_size
                # print(out['table']['XPEAK_IMAGE'][best_score], out['table']['YPEAK_IMAGE'][best_score])
                # print(get_xy_from_frames_txt(_path))
                x = out['table']['YPEAK_IMAGE'][best_score]
                y = out['table']['XPEAK_IMAGE'][best_score]
                x, y = map(int, [x, y])
            else:
                if _win is None:
                    _win = 100
                # use a simple max instead:
                x, y = np.unravel_index(scidata.argmax(), scidata.shape)
                x, y = map(int, [x, y])
        elif _method == 'max':
            if _win is None:
                _win = 100
            x, y = np.unravel_index(scidata.argmax(), scidata.shape)
            x, y = map(int, [x, y])
        elif _method == 'shifts.txt':
            if _win is None:
                _win = 100
            y, x = cls.get_xy_from_shifts_txt(_path)
            if _drizzled:
                x *= 2.0
                y *= 2.0
            else:
                _win = 50
            x, y = map(int, [x, y])

        elif _method == 'manual' and _x is not None and _y is not None:
            if _win is None:
                _win = 100
            x, y = _x, _y
            x, y = map(int, [x, y])
        else:
            raise Exception('unrecognized trimming method.')

        # out of the frame? fix that!
        if x - _win < 0:
            _win -= abs(x - _win)
        if x + _win + 1 >= scidata.shape[0]:
            _win -= abs(scidata.shape[0] - x - _win - 1)
        if y - _win < 0:
            _win -= abs(y - _win)
        if y + _win + 1 >= scidata.shape[1]:
            _win -= abs(scidata.shape[1] - y - _win - 1)

        scidata_cropped = scidata[x - _win: x + _win + 1,
                          y - _win: y + _win + 1]

        return scidata_cropped, int(x), int(y)

    @staticmethod
    def makebox(array, halfwidth, peak1, peak2):
        boxside1a = peak1 - halfwidth
        boxside1b = peak1 + halfwidth
        boxside2a = peak2 - halfwidth
        boxside2b = peak2 + halfwidth

        box = array[int(boxside1a):int(boxside1b), int(boxside2a):int(boxside2b)]
        box_fraction = np.sum(box) / np.sum(array)
        # print('box has: {:.2f}% of light'.format(box_fraction * 100))

        return box, box_fraction

    @staticmethod
    def preview(_path_out, _obs, preview_img, preview_img_cropped,
                SR=None, _fow_x=264, _pix_x=1024, _drizzled=False,
                _x=None, _y=None, objects=None):
        """
        :param _path_out:
        :param preview_img:
        :param preview_img_cropped:
        :param SR:
        :param _x: cropped image will be centered around these _x
        :param _y: and _y + a box will be drawn on full image around this position
        :param objects: np.array([[x_0,y_0], ..., [x_N,y_N]])
        :return:
        """
        # import matplotlib
        # matplotlib.use('Agg')
        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt

        ''' full image '''
        plt.close('all')
        fig = plt.figure()
        fig.set_size_inches(4, 4, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        # plot detected objects:
        if objects is not None:
            ax.plot(objects[:, 0] - 1, objects[:, 1] - 1, 'o',
                    markeredgewidth=1, markerfacecolor='None', markeredgecolor=plt.cm.Oranges(0.8))
        # ax.imshow(preview_img, cmap='gray', origin='lower', interpolation='nearest')
        ax.imshow(preview_img, cmap=plt.cm.magma, origin='lower', interpolation='nearest')
        # plot a box around the cropped object
        if _x is not None and _y is not None:
            _h = int(preview_img_cropped.shape[0])
            _w = int(preview_img_cropped.shape[1])
            ax.add_patch(Rectangle((_y - _w / 2, _x - _h / 2), _w, _h,
                                   fill=False, edgecolor='#f3f3f3', linestyle='dotted'))
        # ax.imshow(preview_img, cmap='gist_heat', origin='lower', interpolation='nearest')
        # plt.axis('off')
        plt.grid('off')

        # save full figure
        fname_full = '{:s}_full.png'.format(_obs)
        if not (os.path.exists(_path_out)):
            os.makedirs(_path_out)
        plt.savefig(os.path.join(_path_out, fname_full), dpi=300)

    @staticmethod
    def gaussian(p, x):
        return p[0] + p[1] * (np.exp(-x * x / (2.0 * p[2] * p[2])))

    @staticmethod
    def moffat(p, x):
        base = 0.0
        scale = p[1]
        fwhm = p[2]
        beta = p[3]

        if np.power(2.0, (1.0 / beta)) > 1.0:
            alpha = fwhm / (2.0 * np.sqrt(np.power(2.0, (1.0 / beta)) - 1.0))
            return base + scale * np.power(1.0 + ((x / alpha) ** 2), -beta)
        else:
            return 1.0

    @classmethod
    def residuals(cls, p, x, y):
        res = 0.0
        for a, b in zip(x, y):
            res += np.fabs(b - cls.moffat(p, a))

        return res


class KPEDRegistrationPipeline(KPEDPipeline):
    """
        KPED's Registration Pipeline
    """
    def __init__(self, _config, _db_entry):
        """
            Init Robo-AO's Bright Star Pipeline
            :param _config: dict with configuration
            :param _db_entry: observation DB entry
        """
        ''' initialize super class '''
        super(KPEDRegistrationPipeline, self).__init__(_config=_config, _db_entry=_db_entry)

        # pipeline name. This goes to 'pipelined' field of obs DB entry
        self.name = 'registration'

        # initialize status
        if self.name not in self.db_entry['pipelined']:
            self.db_entry['pipelined'][self.name] = self.init_status()

    def check_necessary_conditions(self):
        """
            Check if should be run on an obs (if necessary conditions are met)
            :param _db_entry: observation DB entry
        :return:
        """
        go = True

        if 'go' in self.config['pipeline'][self.name]:
            for field_name, field_condition in self.config['pipeline'][self.name]['go'].items():
                if field_name != 'help':
                    # build proper dict reference expression
                    keys = field_name.split('.')
                    expr = 'self.db_entry'
                    for key in keys:
                        expr += "['{:s}']".format(key)
                    # get condition
                    condition = eval(expr + ' ' + field_condition)
                    # eval condition
                    go = go and condition

        return go

    def check_conditions(self, part=None):
        """
            Perform condition checks for running specific parts of pipeline
            :param part: which part of pipeline to run
        :return:
        """
        assert part is not None, 'must specify what to check'

        # check the RP itself?
        if part == 'registration_pipeline':

            # force redo requested?
            _force_redo = self.db_entry['pipelined'][self.name]['status']['force_redo']
            # pipeline done?
            _done = self.db_entry['pipelined'][self.name]['status']['done']
            # how many times tried?
            _num_tries = self.db_entry['pipelined'][self.name]['status']['retries']

            go = (_force_redo or ((not _done) and (_num_tries <= self.config['misc']['max_retries'])))

            return go

        # Preview generation for the results of RP processing?
        elif part == 'registration_pipeline:preview':

            # pipeline done?
            _pipe_done = self.db_entry['pipelined'][self.name]['status']['done']

            # preview generated?
            _preview_done = self.db_entry['pipelined'][self.name]['preview']['done']

            # last_modified == pipe_last_modified?
            _outdated = abs((self.db_entry['pipelined'][self.name]['preview']['last_modified'] -
                             self.db_entry['pipelined'][self.name]['last_modified']).total_seconds()) > 5.0

            # how many times tried?
            _num_tries = self.db_entry['pipelined'][self.name]['preview']['retries']

            go = _pipe_done and ((not _preview_done) or _outdated) \
                 and (_num_tries <= self.config['misc']['max_retries'])

            return go

    @staticmethod
    def init_status():
        time_now_utc = utc_now()
        return {
            'status': {
                'done': False,
                'enqueued': False,
                'force_redo': False,
                'retries': 0,
            },
            'last_modified': time_now_utc,
            'preview': {
                'done': False,
                'force_redo': False,
                'retries': 0,
                'last_modified': time_now_utc
            },
            'location': [],
            'lock_position': None,
            'shifts': None
        }

    def generate_preview(self, f_fits, path_obs, path_out):
        # load first image frame from the fits file
        preview_img = np.nan_to_num(load_fits(f_fits))
        # scale with local contrast optimization for preview:
        preview_img = self.scale_image(preview_img, correction='local')
        # cropped image [_win=None to try to detect]
        _drizzled = False if self.config['pipeline'][self.name]['upsampling_factor'] == 1 else True
        preview_img_cropped, _x, _y = self.trim_frame(_path=path_obs,
                                                      _fits_name=os.path.split(f_fits)[1],
                                                      _win=None, _method='shifts.txt',
                                                      _x=None, _y=None, _drizzled=_drizzled)

        # Strehl ratio (if available, otherwise will be None)
        SR = None

        # fits_header = get_fits_header(f_fits)
        fits_header = self.db_entry['fits_header']
        try:
            # _pix_x = int(re.search(r'(:)(\d+)',
            #                        _select['pipelined'][_pipe]['fits_header']['DETSIZE'][0]).group(2))
            _pix_x = int(re.search(r'(:)(\d+)', fits_header['DETSIZE'][0]).group(2))
        except KeyError:
            # this should be there, even if it's sum.fits
            _pix_x = int(fits_header['NAXIS1'][0])

        self.preview(path_out, self.db_entry['_id'], preview_img, preview_img_cropped,
                     SR, _fow_x=self.config['telescope'][self.telescope]['fov_x'],
                     _pix_x=_pix_x, _drizzled=_drizzled,
                     _x=_x, _y=_y)

    def run(self, part=None):
        """
            Execute specific part of pipeline.
            Possible errors are caught and handled by tasks running specific parts of pipeline
        :return:
        """
        # TODO:
        assert part is not None, 'must specify part to execute'

        # verbose?
        _v = self.config['pipeline'][self.name]['verbose']

        # UTC date of obs:
        _date = self.db_entry['date_utc'].strftime('%Y%m%d')

        # path to store unzipped raw files
        _path_tmp = self.config['path']['path_tmp']
        # path to raw files:
        _path_raw = os.path.join(self.db_entry['raw_data']['location'][1], _date)
        # path to archive:
        _path_archive = os.path.join(self.config['path']['path_archive'], _date)
        # path to output:
        _path_out = os.path.join(_path_archive, self.db_entry['_id'], self.name)
        # path to calibration data produced by lucky pipeline:
        _path_calib = os.path.join(self.config['path']['path_archive'], _date, 'calib')

        if part == 'faint_star_pipeline':

            # raw files:
            _raws_zipped = sorted(self.db_entry['raw_data']['data'])

            # full file names:
            raws = [os.path.join(_path_tmp, _f) for _f in _raws_zipped]

            ''' go off with processing '''
            # get frame size
            x_size = self.db_entry['fits_header']['NAXIS1'][0]
            y_size = self.db_entry['fits_header']['NAXIS2'][0]

            base_val = 0

            with fits.open(sorted(raws)[0]) as p:
                img_size = p[0].data.shape

                # make 5 frames and median combine them to avoid selecting cosmic rays as the guide star
                if _v:
                    print("Getting initial frame average")
                avg_imgs = np.zeros((5, x_size, y_size))

                for avg_n in range(0, 5):
                    n_avg_frames = 0.0
                    for frame_n in list(range(avg_n, len(p), 5))[::4]:
                        avg_imgs[avg_n] += p[frame_n].data + base_val
                        n_avg_frames += 1.0
                    avg_imgs[avg_n] /= n_avg_frames

                avg_img = np.median(avg_imgs, axis=0)

                mid_portion = avg_img[30:avg_img.shape[0] - 30, 30:avg_img.shape[1] - 30]

                # if there's a NaN something's gone horribly wrong
                if np.sum(mid_portion) != np.sum(mid_portion):

                    raise RuntimeError('Something went horribly wrong')

                if _v:
                    print(mid_portion.shape)

                mid_portion = ndimage.gaussian_filter(mid_portion, sigma=10)

                # subtract off a much more smoothed version to remove large-scale gradients across the image
                mid_portion -= ndimage.gaussian_filter(mid_portion, sigma=60)

                mid_portion = mid_portion[30:mid_portion.shape[0] - 30, 30:mid_portion.shape[1] - 30]

                final_gs_y, final_gs_x = np.unravel_index(mid_portion.argmax(), mid_portion.shape)
                final_gs_y += 60
                final_gs_x += 60
                if _v:
                    print("\tGuide star selected at:", final_gs_x, final_gs_y)

            # will cut a window around x_lock, y_lock of size win to do image registration
            x_lock, y_lock = final_gs_x, final_gs_y

            # convert to nearest integers
            cy0, cx0 = int(y_lock), int(x_lock)
            if _v:
                print('initial lock position:', cx0, cy0)

            # make sure win is not too close to image edge
            win = int(np.min([self.config['pipeline'][self.name]['win'], np.min([x_lock, x_size - x_lock]),
                              np.min([y_lock, y_size - y_lock])]))
            # use avg_img to align individual frames to:
            pivot = avg_img

            files_sizes = [os.stat(fs).st_size for fs in raws]

            # get total number of frames to allocate
            # bar = pyprind.ProgBar(sum(files_sizes), stream=1, title='Getting total number of frames')
            # number of frames in each fits file
            n_frames_files = []
            for jj, _file in enumerate(raws):
                with fits.open(_file) as _hdulist:
                    if jj == 0:
                        # get image size (this would be (1024, 1024) for the Andor camera)
                        image_size = _hdulist[0].shape
                    n_frames_files.append(len(_hdulist))
                    # bar.update(iterations=files_sizes[jj])
            # total number of frames
            numFrames = sum(n_frames_files)

            # Stack to seeing-limited image
            if _v:
                bar = pyprind.ProgBar(sum(files_sizes), stream=1, title='Stacking to seeing-limited image')
            summed_seeing_limited_frame = np.zeros((image_size[0], image_size[1]), dtype=np.float)
            for jj, _file in enumerate(raws):
                # print(jj)
                with fits.open(_file, memmap=True) as _hdulist:
                    # frames_before = sum(n_frames_files[:jj])
                    for ii, _ in enumerate(_hdulist):
                        try:
                            summed_seeing_limited_frame += np.nan_to_num(_hdulist[ii].data)
                        except Exception as _e:
                            print(_e)
                            continue
                        # print(ii + frames_before, '\n', _data[ii, :, :])
                if _v:
                    bar.update(iterations=files_sizes[jj])

            # check if there are data to be processed:
            if np.abs(np.max(summed_seeing_limited_frame)) < 1e-9:  # only zeros in summed_seeing_limited_frame
                raise Exception('No data in the cube to be processed.')

            # remove cosmic rays:
            if _v:
                print('removing cosmic rays from the seeing limited image')
            summed_seeing_limited_frame = \
            lax.lacosmicx(np.ascontiguousarray(summed_seeing_limited_frame, dtype=np.float32),
                          sigclip=20, sigfrac=0.3, objlim=5.0,
                          gain=1.0, readnoise=6.5, satlevel=65536.0, pssl=0.0, niter=4,
                          sepmed=True, cleantype='meanmask', fsmode='median',
                          psfmodel='gauss', psffwhm=2.5, psfsize=7, psfk=None,
                          psfbeta=4.765, verbose=False)[1]

            # load darks and flats
            if _v:
                print('Loading darks and flats')
            dark, flat = self.load_darks_and_flats(_path_calib, str(int(self.db_entry['fits_header']['MODE_NUM'][0])),
                                                   self.db_entry['filter'], image_size[0])
            if dark is None or flat is None:
                raise Exception('Could not open darks and flats')

            if _v:
                print('Total number of frames to be registered: {:d}'.format(numFrames))

            # Sum of all (properly shifted) frames (with not too large a shift and chi**2)
            _upsampling_factor = int(self.config['pipeline'][self.name]['upsampling_factor'])
            # assert (type(_upsampling_factor) == int), '_upsampling_factor must be int'
            if _upsampling_factor == 1:
                summed_frame = np.zeros_like(summed_seeing_limited_frame, dtype=np.float)
            else:
                summed_frame = np.zeros((summed_seeing_limited_frame.shape[0] * _upsampling_factor,
                                         summed_seeing_limited_frame.shape[1] * _upsampling_factor), dtype=np.float)

            # Pick a frame to align to
            # seeing-limited sum of all frames:
            if _v:
                print(pivot)
            if pivot == (-1, -1):
                im1 = deepcopy(summed_seeing_limited_frame)
                print('using seeing-limited image as pivot frame')
            else:
                try:
                    with fits.open(raws[pivot[0]], memmap=True) as _hdulist:
                        im1 = np.array(np.nan_to_num(_hdulist[pivot[1]].data), dtype=np.float)
                    print('using frame {:d} from raw fits-file #{:d} as pivot frame'.format(*pivot[::-1]))
                except Exception as _e:
                    print(_e)
                    im1 = deepcopy(summed_seeing_limited_frame)
                    print('using seeing-limited image as pivot frame')

            # print(im1.shape, dark.shape, flat.shape)
            im1 = self.calibrate_frame(im1, dark, flat, _iter=3)
            im1 = gaussian_filter(im1, sigma=5)  # 5, 10
            im1 = im1[cy0 - win: cy0 + win, cx0 - win: cx0 + win]

            # add resolution!
            if _upsampling_factor != 1:
                im1 = image_registration.fft_tools.upsample_image(im1, upsample_factor=_upsampling_factor)

            # export_fits(os.path.join(_path_out, self.db_entry['_id'] + '_pivot_win.fits'), im1)

            # frame_num x y ex ey:
            shifts = np.zeros((numFrames, 5))

            # set up frequency grid for shift2d
            ny, nx = image_size
            if _upsampling_factor != 1:
                ny *= _upsampling_factor
                nx *= _upsampling_factor

            xfreq_0 = np.fft.fftfreq(nx)[np.newaxis, :]
            yfreq_0 = np.fft.fftfreq(ny)[:, np.newaxis]

            nthreads = self.config['pipeline'][self.name]['n_threads']
            fftn, ifftn = image_registration.fft_tools.fast_ffts.get_ffts(
                                nthreads=nthreads, use_numpy_fft=False)

            if _v:
                bar = pyprind.ProgBar(numFrames, stream=1, title='Registering frames')

            fn = 0
            # from time import time as _time
            for jj, _file in enumerate(raws):
                with fits.open(_file) as _hdulist:
                    # frames_before = sum(n_frames_files[:jj])
                    for ii, _ in enumerate(_hdulist):
                        try:
                            img = np.array(np.nan_to_num(_hdulist[ii].data), dtype=np.float)  # do proper casting
                        except Exception as _e:
                            print(_e)
                            continue

                        # tic = _time()
                        img = self.calibrate_frame(img, dark, flat, _iter=3)
                        # print(_time()-tic)

                        # tic = _time()
                        img_comp = gaussian_filter(img, sigma=5)
                        img_comp = img_comp[cy0 - win: cy0 + win, cx0 - win: cx0 + win]
                        # print(_time() - tic)

                        # add resolution!
                        if _upsampling_factor != 1:
                            img_comp = image_registration.fft_tools.upsample_image(img_comp,
                                                                                   upsample_factor=_upsampling_factor)

                        # tic = _time()
                        # chi2_shift -> chi2_shift_iterzoom
                        dy2, dx2, edy2, edx2 = image_registration.chi2_shift(im1, img_comp, nthreads=nthreads,
                                                                             upsample_factor='auto', zeromean=True)
                        # print(dx2, dy2, edx2, edy2)
                        # print(_time() - tic)
                        # tic = _time()
                        # note the order of dx and dy in shift2d vs shiftnd!!!
                        # img = image_registration.fft_tools.shiftnd(img, (-dx2, -dy2),
                        #                                            nthreads=_nthreads, use_numpy_fft=False)
                        # img = self.shift2d(fftn, ifftn, img, -dy2, -dx2, xfreq_0, yfreq_0)
                        if _upsampling_factor == 1:
                            img = self.shift2d(fftn, ifftn, img, -dy2, -dx2, xfreq_0, yfreq_0)
                        else:
                            img = self.shift2d(fftn, ifftn, image_registration.fft_tools.upsample_image(img,
                                                                                   upsample_factor=_upsampling_factor),
                                          -dy2, -dx2, xfreq_0, yfreq_0)
                        # print(_time() - tic, '\n')

                        # if np.sqrt(dx2 ** 2 + dy2 ** 2) > 0.8 * _win \
                        #     or np.sqrt(edx2 ** 2 + edy2 ** 2) > 0.5:
                        if np.sqrt(dx2 ** 2 + dy2 ** 2) > 0.8 * win:
                            # skip frames with too large a shift
                            pass
                            # print(' # {:d} shift was too big: '.format(i),
                            #       np.sqrt(shifts[i, 1] ** 2 + shifts[i, 2] ** 2), shifts[i, 1], shifts[i, 2])
                        else:
                            # otherwise store the shift values and add to the 'integrated' image
                            shifts[fn, :] = [fn, -dx2, -dy2, edx2, edy2]
                            summed_frame += img

                        if _v:
                            bar.update()

                        # increment frame number
                        fn += 1

            if _v:
                print('Largest move was {:.2f} pixels for frame {:d}'.
                      format(np.max(np.sqrt(shifts[:, 1] ** 2 + shifts[:, 2] ** 2)),
                             np.argmax(np.sqrt(shifts[:, 1] ** 2 + shifts[:, 2] ** 2))))

            # remove cosmic rays:
            if _v:
                print('removing cosmic rays from the stacked image')
            summed_frame = lax.lacosmicx(np.ascontiguousarray(summed_frame, dtype=np.float32),
                                         sigclip=20, sigfrac=0.3, objlim=5.0,
                                         gain=1.0, readnoise=6.5, satlevel=65536.0, pssl=0.0, niter=4,
                                         sepmed=True, cleantype='meanmask', fsmode='median',
                                         psfmodel='gauss', psffwhm=2.5, psfsize=7, psfk=None,
                                         psfbeta=4.765, verbose=False)[1]

            # output
            if not os.path.exists(_path_out):
                os.makedirs(os.path.join(_path_out))

            # get original fits header for output
            header = self.db_entry['fits_header']

            # save seeing-limited
            export_fits(os.path.join(_path_out, self.db_entry['_id'] + '_simple_sum.fits'),
                        summed_seeing_limited_frame, header)

            # save stacked
            export_fits(os.path.join(_path_out, self.db_entry['_id'] + '_summed.fits'),
                        summed_frame, header)

            cyf, cxf = self.image_center(_path=_path_out, _fits_name=self.db_entry['_id'] + '_summed.fits',
                                         _x0=cx0, _y0=cy0, _win=win)
            if _v:
                print('Output lock position:', cxf, cyf)
            with open(os.path.join(_path_out, 'shifts.txt'), 'w') as _f:
                _f.write('# lock position: {:d} {:d}\n'.format(cxf, cyf))
                _f.write('# frame_number x_shift[pix] y_shift[pix] ex_shift[pix] ey_shift[pix]\n')
                for _i, _x, _y, _ex, _ey in shifts:
                    _f.write('{:.0f} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(_i, _x, _y, _ex, _ey))

            # reduction successful? prepare db entry for update
            self.db_entry['pipelined'][self.name]['status']['done'] = True
            self.db_entry['pipelined'][self.name]['status']['enqueued'] = False
            self.db_entry['pipelined'][self.name]['status']['force_redo'] = False
            self.db_entry['pipelined'][self.name]['status']['retries'] += 1

            # set last_modified as summed.fits modified date:
            time_tag = datetime.datetime.utcfromtimestamp(os.stat(os.path.join(_path_out,
                               '{:s}_summed.fits'.format(self.db_entry['_id']))).st_mtime)
            self.db_entry['pipelined'][self.name]['last_modified'] = time_tag

            self.db_entry['pipelined'][self.name]['location'] = _path_out

            self.db_entry['pipelined'][self.name]['lock_position'] = [int(cxf), int(cyf)]

            shifts_db = []
            for l in shifts:
                shifts_db.append([int(l[0])] + list(map(float, l[1:])))
            self.db_entry['pipelined'][self.name]['shifts'] = shifts_db

            for _file in raws:
                if _v:
                    print('removing', _file)
                os.remove(os.path.join(_file))

        elif part == 'faint_star_pipeline:preview':
            # generate previews
            path_obs = os.path.join(_path_archive, self.db_entry['_id'], self.name)
            f_fits = os.path.join(path_obs, '{:s}_summed.fits'.format(self.db_entry['_id']))
            path_out = os.path.join(path_obs, 'preview')
            self.generate_preview(f_fits=f_fits, path_obs=path_obs, path_out=path_out)

            # prepare to update db entry:
            self.db_entry['pipelined'][self.name]['preview']['done'] = True
            self.db_entry['pipelined'][self.name]['preview']['force_redo'] = False
            self.db_entry['pipelined'][self.name]['preview']['retries'] += 1
            self.db_entry['pipelined'][self.name]['preview']['last_modified'] = \
                self.db_entry['pipelined'][self.name]['last_modified']

    @staticmethod
    def calibrate_frame(im, _dark, _flat, _iter=3):
        im_BKGD = deepcopy(im)
        for j in range(int(_iter)):  # do 3 iterations of sigma-clipping
            try:
                temp = sigmaclip(im_BKGD, 3.0, 3.0)
                im_BKGD = temp[0]  # return arr is 1st element
            except Exception as _e:
                print(_e)
                pass
        sum_BKGD = np.mean(im_BKGD)  # average CCD BKGD
        im -= sum_BKGD
        im -= _dark
        im /= _flat

        return im


def job_registration_pipeline(_id=None, _config=None, _db_entry=None, _task_hash=None):
    try:
        # init pipe here again. [as it's not JSON serializable]
        pip = KPEDRegistrationPipeline(_config=_config, _db_entry=_db_entry)
        # run the pipeline
        pip.run(part='registration_pipeline')

        return {'_id': _id, 'job': 'registration_pipeline', 'hash': _task_hash,
                'status': 'ok', 'message': str(datetime.datetime.now()),
                'db_record_update': ({'_id': _id},
                                     {'$set': {
                                         'pipelined.registration': pip.db_entry['pipelined']['registration']
                                     }}
                                     )
                }
    except Exception as _e:
        traceback.print_exc()
        try:
            _status = _db_entry['pipelined']['registration']
        except Exception as _ee:
            print(str(_ee))
            traceback.print_exc()
            # failed? flush status:
            _status = KPEDRegistrationPipeline.init_status()
        # retries++
        _status['status']['retries'] += 1
        _status['status']['enqueued'] = False
        _status['status']['force_redo'] = False
        _status['status']['done'] = False
        _status['last_modified'] = utc_now()
        return {'_id': _id, 'job': 'registration_pipeline', 'hash': _task_hash,
                'status': 'error', 'message': str(_e),
                'db_record_update': ({'_id': _id},
                                     {'$set': {
                                         'pipelined.registration': _status
                                     }}
                                     )
                }


def job_registration_pipeline_preview(_id=None, _config=None, _db_entry=None, _task_hash=None):
    try:
        # init pipe here again. [as it's not JSON serializable]
        pip = KPEDRegistrationPipeline(_config=_config, _db_entry=_db_entry)
        pip.run(part='registration_pipeline:preview')

        return {'_id': _id, 'job': 'registration_pipeline:preview', 'hash': _task_hash,
                'status': 'ok', 'message': str(datetime.datetime.now()),
                'db_record_update': ({'_id': _id},
                                     {'$set': {
                                         'pipelined.{:s}.preview'.format(pip.name):
                                             pip.db_entry['pipelined']['registration']['preview']
                                     }}
                                     )
                }
    except Exception as _e:
        traceback.print_exc()
        try:
            _status = _db_entry['pipelined']['registration_pipeline']
        except Exception as _ee:
            print(str(_ee))
            traceback.print_exc()
            # failed? flush status:
            _status = KPEDRegistrationPipeline.init_status()
        # retries++
        _status['preview']['retries'] += 1
        _status['preview']['done'] = False
        _status['preview']['force_redo'] = False
        _status['preview']['last_modified'] = utc_now()
        return {'_id': _id, 'job': 'registration_pipeline:preview', 'hash': _task_hash,
                'status': 'error', 'message': str(_e),
                'db_record_update': ({'_id': _id},
                                     {'$set': {
                                         'pipelined.registration_pipeline.preview': _status['preview']
                                     }}
                                     )
                }


if __name__ == '__main__':

    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Data archive for KPED')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    args = parser.parse_args()

    # init archiver:
    arch = KPEDArchiver(args.config_file)

    # start the archiver main house-keeping cycle:
    # arch.cycle()
