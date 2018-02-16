import netCDF4 as nc
import pandas as pd
import numpy as np
import os
import sys

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

variables = ['fdir', 'cdir', 'tcc', 'u10', 'v10', 't2m', 'ssrd', 'ssr']

# variables = ['U10', 'V10', 'SP', 'T2M', 'U100', 'V100']

# variables_ens = ['FDIR', 'CDIR', 'TCC', 'U10', 'V10', 'T2M', 'SSRD', 'SSR']


def convert_deterministic(src, dst):
    src_nc = nc.Dataset(src)
    d = src_nc.filepath()[-11:-3]
    date = datetime.strptime(d, "%Y%m%d")
    #next_month = date + relativedelta(months=1) - timedelta(days=1)
    n_steps = src_nc.dimensions['time'].size
    off_day = 0

    for day in pd.date_range(date, date + relativedelta(hours=+23)):
        output = dst + '{}.myp.npy'.format(day.strftime("%Y%m%d"))

        if os.path.isfile(output):
            print("{} already converted".format(output))
            continue

        print(day)
        myp = np.zeros((0, len(variables) * n_steps + 2))
        for i, lat in enumerate(src_nc.variables['latitude']):
            partial = np.zeros((len(src_nc.variables['longitude']),
                                len(variables) * n_steps + 2))
            for j, lon in enumerate(src_nc.variables['longitude']):
                row = [lon, lat]
                for step in range(n_steps):
                    for var in variables:
                        row.append(src_nc.variables[var][step + off_day, i, j])
                partial[j, :] = row
            myp = np.vstack([partial, myp])
        off_day += n_steps
        np.save(dst + '{}.myp'.format(day.strftime("%Y%m%d")), myp)


def convert_ensembles(src, dst):
    src_nc = nc.Dataset(src)
    d = src_nc.filepath()[-11:-3]
    date = datetime.strptime(d, "%Y%m%d")
    n_steps = src_nc.dimensions['time'].size
    n_ens = src_nc.dimensions['number'].size

    print(date)
    for ens in range(n_ens):
        output = dst + '{}_{}.myp.npy'.format(date.strftime("%Y%m%d"), ens)

        if os.path.isfile(output):
            print("{} already converted".format(output))
            continue

        myp = np.array([]).reshape((0, len(variables) * n_steps + 2))
        for i, lat in enumerate(src_nc.variables['latitude'][::-1]):
            for j, lon in enumerate(src_nc.variables['longitude']):
                row = [lon, lat]
                for step in range(n_steps):
                    for var in variables:
                        row.append(src_nc.variables[var][step, ens, i, j])
                myp = np.vstack([myp, row])
        np.save(dst + '{}_{}.myp'.format(date.strftime("%Y%m%d"), ens), myp)
        print("Ensemble {} done".format(ens))


if __name__ == '__main__':
    f = sys.argv[1]
    output = '/gaa/home/edcastil/scripts/conversion2015/myp/'
    convert_deterministic(f, output)
