import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from itertools import groupby
from functools import reduce
import os.path
import re
import sys

# sys.path.append('/scratch/gaa/alecat/dev/catedra/notebooks')

# from Platform.platform_models.models import AreaGrid


def shape_data(data, lon_l, lon_r, lat_l, lat_r, variables, nsteps):
    idxlon = np.logical_and(data[:, 0] >= lon_l,
                            data[:, 0] <= lon_r)
    idxlat = np.logical_and(data[:, 1] >= lat_l,
                            data[:, 1] <= lat_r)
    idx = np.logical_and(idxlon, idxlat)
    data = data[idx]

    data = np.delete(data, np.s_[0:2], axis=1)
    data = data.reshape((data.shape[0], nsteps, variables))
    data = data.transpose((1, 0, 2))
    data = data.reshape((nsteps, -1))
    return data


def group(x):
    ts = datetime.strptime(str(x), "%Y%m%d%H%M")
    return ts.year, ts.month, ts.day, ts.hour


def to_hourly(df):
    """External function to convert X minute data to hourly data."""
    def group(x):
        ts = datetime.strptime(str(x), "%Y%m%d%H%M")
        if ts.minute > 0:
            ts += timedelta(hours=1)
        return ts.year, ts.month, ts.day, ts.hour

    grp = df.groupby(group)
    new_df = pd.DataFrame(columns=df.columns)

    for (year, month, day, hour), indices in grp.indices.items():
        new_ts = datetime(year, month, day, hour).strftime("%Y%m%d%H")
        new_df.loc[int(new_ts)] = df.iloc[indices].mean()

    return new_df.sort_index(ascending=True)


def select_pen_grid():
    cols = pd.read_csv(
        "/scratch/gaa/alecat/dev/cols_ext.txt", sep=';').columns
    land_grid = []
    for col in cols:
        m = re.search('([-]*\d+[.]\d*), ([-]*\d+[.]\d*)', col)
        lon, lat = float(m.group(1)), float(m.group(2))
        if [lon, lat] not in land_grid:
            land_grid.append([lat, lon])
    return sorted({tuple(x) for x in land_grid})


def create_dataframe(year):
    """Creates a dataframe out of a DataMatrix containing 1 year of hourly
aggregated data (source files are 15 minute)."""
    dfs = []
    pathin = '/scratch/gaa/alecat/data/eumetsat/processed/myp/15min/'
    pathout = '/scratch/gaa/alecat/data/eumetsat/processed/'
    for t in range(365):
        m = DataMatrix(datetime(year + 1, 1, 1) - timedelta(days=t),
                       pathin, pathout, delta=1)
        dfs.append(m.dataMatrix if not (type(m.dataMatrix) is bool) else
                   pd.DataFrame())
    return pd.concat(map(to_hourly, dfs))


def select_grid(df, latlons):
    """Extract a set of grid points out of a dataframe (DataMatix format) using
latlons."""
    pathin = '/scratch/gaa/alecat/data/eumetsat/processed/myp/15min/'
    pathout = '/scratch/gaa/alecat/data/eumetsat/processed/'
    m = DataMatrix(datetime.strptime(str(df.index.max()), "%Y%m%d%H"),
                   pathin, pathout, delta=0)
    m = m.data_matrix_from_data_frame(df)
    return m.query_subgrid(latlons=latlons)


def remove_dups(df):
    idx = np.unique(df.index, return_index=True)[1]
    data = df.iloc[idx]
    return np.concatenate((np.matrix(data.index).T, data.values), axis=1)


def get_seviri_tags(calibration_modes, channels):
    tags = []
    for chn_name in sorted(channels.keys()):
        for cal_mode in calibration_modes:
            if cal_mode == 'reflectances/bt':
                if chn_name in ['VIS006', 'VIS008', 'IR_016']:
                    tags.append(chn_name + ' %')
                else:
                    tags.append(chn_name + ' K')
            else:
                tags.append(chn_name + ' R')
    return tags


def get_clmask_tags():
    return ['CLMASK']


class AreaGrid(object):
    '''
    AreaGrid allows the creation and use of unconstrained grids for general
    purpose, similar to Grid. This kind of Grid is not attached to a farm, so
    its use is further flexible and useful in a lot more situations.
    '''

    def __init__(self, ullat, ullon, lrlat, lrlon, res):
        self.ullat = ullat
        self.ullon = ullon
        self.lrlat = lrlat
        self.lrlon = lrlon
        self.res = res

    def get_lats(self):
        return self.lrlat, self.ullat

    def get_lons(self):
        return self.ullon, self.lrlon

    def get_lats_lons(self):
        latlons = []
        for i in np.arange(self.lrlat, self.ullat + self.res, self.res):
            for j in np.arange(self.ullon, self.lrlon + self.res, self.res):
                latlons.append([i, j])
        return latlons

    def get_res(self):
        return self.res


class DataMatrix(object):
    '''
    Template for the dataMatrix model, the most important attribute is
    dataMatrix, which stores a pandas DataFrame containing the matrix
    itself. another attributes are:

      * config; it\'s only used to identify the farms and its predefined
          coords, until only square grids are used. will be deprecated.
      * date_format; format to translate the dates to.
      * path; the path to the directory containing all the myp files.
      * date; the date when the dataMatrix is created for.

    '''
    date_format = '%Y%m%d%H%M'

    channels = {"VIS006": 'ch1',
                "VIS008": 'ch2',
                "IR_016": 'ch3',
                "IR_039": 'ch4',
                "WV_062": 'ch5',
                "WV_073": 'ch6',
                "IR_087": 'ch7',
                "IR_097": 'ch8',
                "IR_108": 'ch9',
                "IR_120": 'ch10',
                "IR_134": 'ch11'}

    calibration_modes = ['radiances', 'reflectances/bt']

    def __init__(self, date, pathin, pathout, ncoord=2, delta=122,
                 grid=False, latlons=False, ifexists=False,
                 create=True, dm=False, suffix="", freq='15min'):
        '''
        Builds a dataMatrix object by reading up to DELTA MYP files stored
        in PATHIN. The PATHOUT argument indicates the path where the final
        dataMatrix may be stored.

        We can indicate a specific GRID for the data or a DataFrame for
        which to create the wrapper.
        '''
        self.path = pathin
        self.out = pathout
        self.file = self.out + \
            date.strftime(self.date_format) + '.mdata' + suffix + '.npy'
        self.date = date

        self.tags_seviri = get_seviri_tags(self.calibration_modes,
                                           self.channels)
        # self.tags_clmask = get_clmask_tags()
        self.tags = self.tags_seviri  # + self.tags_clmask

        self.freq = freq

        if not grid:
            self.grid = AreaGrid(44, -9.5, 35.5, 4.5, 0.125)
        else:
            self.grid = grid

        if not latlons:
            latlons = self.grid.get_lats_lons()

        self.cols_seviri = self.query_cols(latlons=latlons,
                                           tags=self.tags_seviri)
        # self.cols_clmask = self.query_cols(latlons=latlons,grid=grid,
        #                                    tags=self.tags_clmask)

        if os.path.isfile(self.file) and ifexists:
            data = np.load(self.file)
            index = data[:, 0].astype(int)  # self.query_index(fechaini, date)
            data = data[:, 1:]
            cols = list(self.cols_seviri)  # + list(self.cols_clmask)
            self.dataMatrix = pd.DataFrame(data, index=index, columns=cols)
        elif create:
            self.dataMatrix = False
            self.generate_matrix(ncoord, delta, freq)
        elif not create and dm is not False:
            self.dataMatrix = dm

    def generate_matrix(self, ncoord, delta, freq):
        '''
        Generates the matrix itself, receives the same parameters as
        __init__, plus delta, which indicates how many days backwards
        we shall look to build the matrix. after reading the base variables,
        which are stored in the base files, it computes the modules
        of the velocity.

        This method doesn\'t return anything, but the resulting matrix
        is stored in the dataMatrix attribute of the object.
        '''
        lon_l, lon_r = self.grid.get_lons()
        lat_l, lat_r = self.grid.get_lats()
        nsteps = 1

        for date in pd.date_range(self.date - timedelta(days=delta),
                                  # Avoid duplicating xx:00 timestamps
                                  self.date - timedelta(minutes=15),
                                  freq=freq):
            fecha = datetime.strftime(date, self.date_format)
            files = self.path + fecha + ".seviri.myp.npy"
            # filec = self.path + fecha + ".clmask.myp.npy"
            print(files)

            if not os.path.isfile(files):  # or not os.path.isfile(filec):
                continue

            datas = np.load(files)
            # datac = np.load(filec)

            datas = shape_data(datas, lon_l, lon_r, lat_l, lat_r, 22, nsteps)
            # datac = shape_data(datac, lon_l, lon_r, lat_l, lat_r, 1, nsteps)

            index = self.query_index(date, date, freq)

            df_seviri = pd.DataFrame(datas, index=index,
                                     columns=self.cols_seviri)
            # df_clmask = pd.DataFrame(datac, index=index,
            #                          columns=self.cols_clmask)
            df = df_seviri  # pd.concat([df_seviri, df_clmask], axis=1)

            if isinstance(self.dataMatrix, bool):
                self.dataMatrix = df
            else:
                self.dataMatrix = pd.concat([self.dataMatrix, df],
                                            join='outer', axis=0)

    def data_matrix_from_data_frame(self, df):
        '''
        Build a new DataMatrix copying the properties of self but with the
        inner DataFrame being replaced by df.
        '''
        return DataMatrix(self.date, self.path, self.out,
                          grid=self.grid, create=False, dm=df)

    def query_index(self, start_date, end_date, freq):
        '''
        Generates a list of indexes to address the pandas DataFrame.

        This indexes are generated by checking the meteorology files
        we have inside the matrix. for each of this files, it calculates
        the timestamps by using the nsteps parameter, which indicates
        how many three-hourly intervals the file contains.
        '''
        index = []
        for date in pd.date_range(start_date, end_date, freq=freq):
            files = self.path + \
                date.strftime(self.date_format) + '.seviri.myp.npy'
            # filec = self.path + date.strftime(self.date_format) +
            # '.clmask.myp.npy'
            if os.path.isfile(files):  # and os.path.isfile(filec):
                ts = date.strftime(self.date_format)
                index.append(int(ts))

        return sorted(index)

    def query_cols(self, grid=None, latlons=None, tags=[]):
        '''
        Generates a list of the tags used to address each of the columns
        of the pandas DataFrame.

        Each tag is the string composition of the coordinates at the
        point and the name of the variable it refers to.
        '''
        if grid:
            lon_l, lon_r = grid.get_lons()
            lat_l, lat_r = grid.get_lats()
            res = grid.get_res()
            latlons = [(i, j) for i in np.arange(lat_l, lat_r + res, res)
                       for j in np.arange(lon_l, lon_r + res, res)]

        return np.array([['({0}, {1}) {2}'.format(j, i, tag)
                          for tag in tags]
                         for i, j in latlons]).flatten()

    def query_subgrid(self, grid=None, latlons=None, inplace=False):
        '''
        By giving a grid object, extracts the submatrix referring to it. If
        grid is none, the complete matrix is returned.

        If inplace is True, the inner dataMatrix is replaced by the
        resulting submatrix.
        '''
        if grid is not None or latlons is not None:
            result = self.dataMatrix.loc[:, self.query_cols(grid=grid,
                                                            latlons=latlons,
                                                            tags=self.tags)]
        else:
            result = self.dataMatrix

        if inplace:
            self.dataMatrix = result
        return result

    def query_subgrid_dates(self, start_date, end_date,
                            grid=None, latlons=None, inplace=False):
        '''
        This method returns a subset of the general matrix following the
        above method, for a farm and dates between start_date and
        end_date.

        If inplace is True, the inner dataMatrix is replaced by the
        resulting submatrix.
        '''
        if grid or latlons:
            submatrix = self.query_subgrid(grid=grid, latlons=latlons)
        else:
            submatrix = self.dataMatrix

        result = submatrix.loc[self.query_index(
            start_date, end_date, self.freq), :]
        if inplace:
            self.dataMatrix = result
        return result

    def query_subset_channels(self, channels=channels, inplace=False):
        '''
        Return a subset of the channels for the whole grid.

        If inplace is True, the inner dataMatrix is replaced by the
        resulting submatrix.
        '''
        result = self.dataMatrix.loc[:, self.query_cols(grid=self.grid,
                                                        tags=channels)]
        if inplace:
            self.dataMatrix = result
        return result

    def query_subset_channels_grid(self, grid=None, latlons=None,
                                   channels=channels, inplace=False):
        '''
        Return a subset of the channels for a given grid. If the grid is
        None, the complete grid is taken.

        If inplace is True, the inner dataMatrix is replaced by the
        resulting submatrix.
        '''
        if grid is None and latlons is None:
            grid = self.grid

        result = self.dataMatrix.loc[:, self.query_cols(grid=grid,
                                                        latlons=latlons,
                                                        tags=channels)]
        if inplace:
            self.dataMatrix = result
        return result

    def query_subset_channels_grid_date_range(self, start_date, end_date,
                                              grid=None, channels=channels,
                                              inplace=False):
        '''
        Return a subset of the total channels within a given date range
        and grid. If grid is None, the complete grid is taken.

        If inplace is True, the inner dataMatrix is replaced by the
        resulting submatrix.
        '''
        if grid is None:
            grid = self.grid

        result = self.dataMatrix.loc[self.query_index(start_date, end_date, self.freq),
                                     self.query_cols(grid=grid, tags=self.tags)]
        if inplace:
            self.dataMatrix = result
        return result

    def group_by_hour(self):
        def group(x):
            ts = datetime.strptime(str(x), "%Y%m%d%H%M")
            return ts.year, ts.month, ts.day, ts.hour

        grp = self.dataMatrix.groupby(group)
        new_df = pd.DataFrame(columns=self.dataMatrix.columns)

        for (year, month, day, hour), indices in grp.indices.items():
            new_ts = datetime(year, month, day, hour).strftime("%Y%m%d%H")
            new_df.loc[int(new_ts)] = self.dataMatrix.iloc[indices].mean()

        self.dataMatrix = new_df.sort_index()
        return new_df

    def group_by_channel(self, variable):
        '''
        Averages the value of channel (with calibration calib) on the
        whole grid.
        '''
        regexp = '(\w{6}[ ]?[A-Z%]?)'

        def keyf(col):
            return re.search(regexp, col).group(0)

        grid = {}
        for t in self.tags_seviri:  # + self.tags_clmask:
            grid[t] = []

        cols = sorted(self.dataMatrix.columns)
        groups = map(lambda x: [x[0], list(x[1])[0]], groupby(cols, key=keyf))

        for k, v in groups:
            grid[k].append(v)

        return reduce(lambda acc, c: acc + self.dataMatrix[c].fillna(0),
                      grid[variable], 0) / len(grid[variable])

    def group_channels(self):
        '''
        Return a DataFrame with all the aggregated values for each
        column in the dataMatrix.
        '''
        data = {var: self.group_by_channel(var) for var in self.tags}
        df_agg = pd.concat([data[k] for k in sorted(data.keys())], axis=1)
        df_agg.columns = sorted(data.keys())
        return df_agg

    def combine_channels(self, f, variables=None, init=0):
        '''
        Combines the given variables by f with an initial value init.

        F should by a function taking an accumulated value and a
        grouped DataFrame. Init is the initial value for this accumulation.

        '''
        variables = variables or self.tags
        return reduce(f, [self.group_by_channel(var) for var in variables],
                      init)

    def save_matrix(self, suffix=""):
        '''
        Stores the matrix as a np binary format file, so we reduce
        slightly the read times, but also has a counterpart, as long
        as the resulting file is quite big.

        This is intended to use within local environments with big
        storage capacities, and mainly to run on benchmarking mode for
        a whole year or so, when a lot of dataMatrix are to be
        created.
        '''
        file = self.out + \
            self.date.strftime(self.date_format) + '.mdata' + suffix
        data = np.concatenate(
            (self.dataMatrix.index, self.dataMatrix.values), axis=1)
        np.save(file, data)
