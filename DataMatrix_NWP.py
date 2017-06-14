import pandas as pd
from itertools import groupby
from functools import reduce
import numpy as np
from datetime import timedelta, datetime
import os.path
import re

nwp_deterministic_tags = [
    'FDIR', 'CDIR', 'TCC', 'U10', 'V10', 'T2M', 'SSRD', 'SSR', 'SSRC', 'v10'
]

nwp_ensembles_tags = [
    'FDIR', 'CDIR', 'TCC', 'U10', 'V10', 'T2M', 'SSRD', 'SSR'
]


def shape_data(data, lon_l, lon_r, lat_l, lat_r, variables, nsteps):
    idxlon = np.logical_and(data[:, 0] >= lon_l, data[:, 0] <= lon_r)
    idxlat = np.logical_and(data[:, 1] >= lat_l, data[:, 1] <= lat_r)
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
    cols = pd.read_csv("/scratch/gaa/alecat/dev/cols_ext.txt", sep=';').columns
    land_grid = []
    for col in cols:
        m = re.search('([-]*\d+[.]\d*), ([-]*\d+[.]\d*)', col)
        lon, lat = float(m.group(1)), float(m.group(2))
        if [lon, lat] not in land_grid:
            land_grid.append([lat, lon])
    return sorted({tuple(x) for x in land_grid})


def create_dataframe(year,
                     pathin,
                     pathout,
                     hourly=True,
                     model='deterministic',
                     tags=nwp_deterministic_tags,
                     n_ens=50):
    """Creates a dataframe out of a DataMatrix containing 1 year of hourly
aggregated data (source files are 15 minute)."""
    dfs = []
    # pathin = '/scratch/gaa/alecat/data/eumetsat/processed/myp/15min/'
    # pathout = '/scratch/gaa/alecat/data/eumetsat/processed/'
    for t in range(365):
        m = DataMatrix(
            datetime(year + 1, 1, 1) - timedelta(days=t),
            pathin,
            pathout,
            delta=1,
            tags=tags,
            model=model,
            n_ens=n_ens)
        dfs.append(
            m.dataMatrix
            if not (type(m.dataMatrix) is bool) else pd.DataFrame())
    if not hourly:
        return remove_dups(pd.concat(map(to_hourly, dfs))), dfs[0].columns
    return remove_dups(pd.concat(dfs)), dfs[0].columns


def select_grid(df, latlons):
    """Extract a set of grid points out of a dataframe (DataMatix format) using
latlons."""
    pathin = '/scratch/gaa/alecat/data/eumetsat/processed/myp/15min/'
    pathout = '/scratch/gaa/alecat/data/eumetsat/processed/'
    m = DataMatrix(
        datetime.strptime(str(df.index.max()), "%Y%m%d%H"),
        pathin,
        pathout,
        delta=0)
    m = m.data_matrix_from_data_frame(df)
    return m.query_subgrid(latlons=latlons)


def remove_dups(df):
    idx = np.unique(df.index, return_index=True)[1]
    data = df.iloc[idx]
    return np.hstack((np.matrix(data.index).T, data.values))


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


# Sotavento grid 0.25 resolution
stv_grid = AreaGrid(44.00, -9.5, 42.25, -6.00, 0.25)


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
    date_format = '%Y%m%d%H'
    file_format = '%Y%m%d'

    def __init__(self,
                 date,
                 pathin,
                 pathout,
                 ncoord=2,
                 delta=31,
                 n_ens=50,
                 grid=False,
                 latlons=False,
                 ifexists=False,
                 create=True,
                 dm=False,
                 suffix="",
                 freq='D',
                 tags=nwp_deterministic_tags,
                 model='deterministic'):
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

        self.tags = tags
        if model == 'deterministic':
            nsteps = 24
        elif model == 'ensembles':
            self.n_ens = n_ens
            nsteps = 9

        self.freq = freq

        if not grid:
            if model == 'deterministic':
                self.grid = AreaGrid(44, -9.5, 35.5, 4.5, 0.125)
            else:
                self.grid = AreaGrid(44, -9.5, 35.5, 4.5, 0.25)
        else:
            self.grid = grid

        if not latlons:
            latlons = self.grid.get_lats_lons()

        self.cols = self.query_cols(latlons=latlons, tags=self.tags)

        if os.path.isfile(self.file) and ifexists:
            data = np.load(self.file)
            index = data[:, 0].astype(int)  # self.query_index(fechaini, date)
            data = data[:, 1:]
            if model == 'deterministic':
                cols = list(self.cols)
            else:
                cols = list(
                    np.array([[c + ' {}'.format(ens) for c in self.cols]
                              for ens in range(self.n_ens)]).flatten())
            self.dataMatrix = pd.DataFrame(data, index=index, columns=cols)
        elif create:
            self.dataMatrix = False
            if model == 'deterministic':
                self.generate_matrix(delta, freq, nsteps)
            else:
                self.generate_matrix_ensembles(delta, freq, nsteps)
        elif not create and dm is not False:
            self.dataMatrix = dm

    def generate_matrix(self, delta, freq, nsteps):
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

        for date in pd.date_range(
                self.date - timedelta(days=delta), self.date, freq=freq):
            fecha = datetime.strftime(date, self.file_format)
            files = self.path + fecha + ".myp.npy"
            print(files)

            if not os.path.isfile(files):  # or not os.path.isfile(filec):
                continue

            datas = np.load(files)

            datas = shape_data(datas, lon_l, lon_r, lat_l, lat_r,
                               len(self.tags), nsteps)

            index = self.query_index(date, date, 'H')

            df = pd.DataFrame(datas, index=index, columns=self.cols)

            if isinstance(self.dataMatrix, bool):
                self.dataMatrix = df
            else:
                self.dataMatrix = pd.concat(
                    [self.dataMatrix, df], join='outer', axis=0)

    def generate_matrix_ensembles(self, delta, freq, nsteps):
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

        for date in pd.date_range(
                self.date - timedelta(days=delta), self.date, freq=freq):
            df_ens = False
            for ens in range(self.n_ens):
                fecha = datetime.strftime(date, self.file_format)
                files = self.path + fecha
                files += "_{}.myp.npy".format(ens) if self.n_ens > 1 else ".myp.npy"
                print(files)

                if not os.path.isfile(files):
                    continue

                datas = np.load(files)

                datas = shape_data(datas, lon_l, lon_r, lat_l, lat_r,
                                   len(self.tags), nsteps)

                index = self.query_index(date, date, '3H', ens)
                cols = [c + ' {}'.format(ens) for c in self.cols]

                df = pd.DataFrame(datas, index=index, columns=cols)

                if isinstance(df_ens, bool):
                    df_ens = df
                else:
                    df_ens = pd.concat([df_ens, df], join='outer', axis=1)
            if isinstance(self.dataMatrix, bool):
                self.dataMatrix = df_ens
            else:
                if not isinstance(df_ens, bool):
                    self.dataMatrix = pd.concat(
                        [self.dataMatrix, df_ens], join='outer', axis=0)

    def data_matrix_from_data_frame(self, df):
        '''
        Build a new DataMatrix copying the properties of self but with the
        inner DataFrame being replaced by df.
        '''
        return DataMatrix(
            self.date,
            self.path,
            self.out,
            grid=self.grid,
            create=False,
            dm=df)

    def query_index(self, start_date, end_date, freq, ens=False):
        '''
        Generates a list of indexes to address the pandas DataFrame.

        This indexes are generated by checking the meteorology files
        we have inside the matrix. for each of this files, it calculates
        the timestamps by using the nsteps parameter, which indicates
        how many three-hourly intervals the file contains.
        '''
        index = []
        for date in pd.date_range(start_date, end_date, freq=freq):
            if isinstance(ens, bool):
                files = self.path + \
                    date.strftime(self.file_format) + '.myp.npy'
                if os.path.isfile(files):
                    for hour in pd.date_range(
                            start_date,
                            end_date + timedelta(days=1) - timedelta(hours=1),
                            freq=freq):
                        ts = hour.strftime(self.date_format)
                        index.append(int(ts))
            else:
                files = self.path + date.strftime(self.file_format)
                files += '_0.myp.npy' if self.n_ens > 1 else ".myp.npy"
                if os.path.isfile(files):
                    for hour in pd.date_range(
                            start_date, end_date + timedelta(days=1),
                            freq=freq):
                        ts = hour.strftime(self.date_format)
                        index.append(int(ts))
                # index = index[:-1]

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
            latlons = [(i, j)
                       for i in np.arange(lat_l, lat_r + res, res)
                       for j in np.arange(lon_l, lon_r + res, res)]

        return np.array([['({0}, {1}) {2}'.format(j, i, tag) for tag in tags]
                         for i, j in latlons]).flatten()

    def query_subgrid(self, grid=None, latlons=None, inplace=False):
        '''
        By giving a grid object, extracts the submatrix referring to it. If
        grid is none, the complete matrix is returned.

        If inplace is True, the inner dataMatrix is replaced by the
        resulting submatrix.
        '''
        if grid is not None or latlons is not None:
            result = self.dataMatrix.loc[:, self.query_cols(
                grid=grid, latlons=latlons, tags=self.tags)]
        else:
            result = self.dataMatrix

        if inplace:
            self.dataMatrix = result
        return result

    def query_subgrid_dates(self,
                            start_date,
                            end_date,
                            grid=None,
                            latlons=None,
                            inplace=False):
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

        result = submatrix.loc[self.query_index(start_date, end_date,
                                                self.freq), :]
        if inplace:
            self.dataMatrix = result
        return result

    def group_by_variable(self, variable):
        '''
        Averages the value of variable on the whole grid.
        '''

        def keyf(col):
            return col.rsplit()[-1]

        grid = {}
        for t in self.tags:
            grid[t] = []

        cols = sorted(self.dataMatrix.columns)
        groups = map(lambda x: [x[0], list(x[1])[0]], groupby(cols, key=keyf))

        for k, v in groups:
            grid[k].append(v)

        return reduce(lambda acc, c: acc + self.dataMatrix[c].fillna(0),
                      grid[variable], 0) / len(grid[variable])

    def group_by_hour(self):
        def group(x):
            ts = datetime.strptime(str(x), "%Y%m%d%H")
            return ts.year, ts.month, ts.day, ts.hour

        grp = self.dataMatrix.groupby(group)
        new_df = pd.DataFrame(columns=self.dataMatrix.columns)

        for (year, month, day, hour), indices in grp.indices.items():
            new_ts = datetime(year, month, day, hour).strftime("%Y%m%d%H")
            new_df.loc[int(new_ts)] = self.dataMatrix.iloc[indices].mean()

        self.dataMatrix = new_df.sort_index()
        return new_df

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
        data = np.hstack(
            (self.dataMatrix.index[:, np.newaxis], self.dataMatrix.values))
        np.save(file, data)
