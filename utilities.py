#!/usr/bin/python

import pandas as pd
import numpy as np
import pickle
import sklearn.preprocessing as skpp
import re

from sklearn.linear_model import LassoCV, Lasso
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import DataMatrix as dm
import DataMatrix_NWP as dnwp
import sunrise as sr


def load_data(pathin,
              pathout,
              shp,
              prodsfile,
              year,
              land_grid=False,
              prods=True):
    if not land_grid:
        land_grid = dm.select_pen_grid()

    m = dm.DataMatrix(
        datetime(year + 1, 1, 1),
        pathin,
        pathout,
        latlons=land_grid,
        ifexists=True,
        suffix=".15minH")

    if prods:
        prods = pd.read_csv(prodsfile, header=0, index_col=0)

    return m, prods, land_grid


def load_nwp_data(pathin, pathout, year):
    m = dnwp.DataMatrix(
        datetime(year, 12, 31),
        pathin,
        pathout,
        suffix='.det_noacc_vmodule',
        ifexists=True)
    return m


def load_nwp_df(nwp, land_grid, shft):
    df = nwp.query_subgrid(latlons=land_grid)
    return df


def load_df(m, land_grid, prods, variables, shft):
    df = m.query_subset_channels_grid(latlons=land_grid, channels=variables)
    df = shift_df(df, shft).join(prods, how='inner')
    return df


def merge_satellite_nwp_prods(m, nwp, prods, land_grid, variables,
                              variables_nwp, shft):
    satellite_df = m.query_subset_channels_grid(
        latlons=land_grid, channels=variables)
    satellite_df = shift_df(satellite_df, shft)
    nwp_cols = nwp.query_cols(latlons=land_grid, tags=variables_nwp)
    nwp_df = nwp.dataMatrix[nwp_cols]
    nwp_prods_df = nwp_df.join(prods, how='inner')
    df = satellite_df.join(nwp_prods_df, how='inner')
    return df.sort_index(axis=1)


def clip_df_daylight(df):
    ix = sr.filter_daylight_hours(df.index)
    return df[ix]


def clip_hours(df, hours):
    ix = sr.filter_hours(df.index, hours)
    return df.loc[ix]


def shift_df(df, shft=0):
    df_copy = df.copy()
    ts = [datetime.strptime(str(ix), "%Y%m%d%H") for ix in df.index]
    df_copy.index = [
        int((t + timedelta(hours=shft)).strftime("%Y%m%d%H")) for t in ts
    ]
    return df_copy


def load_X_Y(df, y_col, out_scaler=None, scaler=None, standard=True):
    if not scaler:
        if standard:
            scalerX = skpp.StandardScaler()
        else:
            scalerX = skpp.MinMaxScaler((-1, 1))
    else:
        scalerX = scaler

    dfX = df.loc[:, df.columns != y_col]
    X = dfX.fillna(0).values

    if not scaler:
        Xsc = scalerX.fit_transform(X)
    else:
        Xsc = scalerX.transform(X)

    Y = df[y_col].values

    if out_scaler:
        fileScaler = open(out_scaler, 'wb')
        pickle.dump(scalerX, fileScaler)

    return Xsc, Y, scalerX


def fit_model(X, Y, model=None, out_model=None):
    if not model:
        lCV = LassoCV(max_iter=1000000)  # finds best model
    else:
        lCV = Lasso(alpha=model.alpha_, max_iter=1000000)
    lCV.fit(X, Y)

    if out_model:
        fileLasso = open(out_model, 'wb')
        pickle.dump(lCV, fileLasso)

    return lCV


def load_store_X_Y(df, hours, output, scaler=None, standard=True):
    df_t = clip_hours(df, hours)
    X, Y, scalerX = load_X_Y(df_t, 'Prod', scaler=scaler, standard=standard)
    matrix = np.concatenate((Y[:, None], X), axis=1)
    np.save(output, matrix.astype(np.float32))

    return scalerX


def load_preds(dfs_test, path, suff, columns, models, prefix=None):
    dfs = {}
    for shft in sorted(models.keys()):
        m = 0
        seen = np.array([])
        dfs[shft] = {}
        dfs[shft][m] = pd.DataFrame()

        for model in sorted(models[shft]):
            if any([x in seen for x in models[shft][model]]):
                seen = np.array([])
                dfs[shft][m] = dfs[shft][m].sort_index()
                dfs[shft][m + 1] = pd.DataFrame()
                m += 1

            df_t = clip_hours(dfs_test[shft], models[shft][model])
            if prefix:
                data = np.load(
                    "{0}{}h{1}{2}.{3}".format(path, prefix, shft, model, suff))
            else:
                data = np.load(
                    "{0}h{1}{2}.{3}".format(path, shft, model, suff))
            if shft in dfs:
                dfs[shft][m] = pd.concat([
                    dfs[shft][m], pd.DataFrame(
                        data, index=df_t.index, columns=columns)
                ])
            else:
                dfs[shft][m] = pd.DataFrame(
                    data, index=df_t.index, columns=columns)

            seen = np.append(seen, models[shft][model])

    return dfs


def calc_error(model, X, Y, out_mae):
    mae = -model_selection.cross_val_score(
        model, X, Y, cv=10, scoring='mean_absolute_error')

    with open(out_mae, "w") as f:
        f.write("mae : " + str(mae.mean()))

    return mae.mean(), mae


def load_prods_preds(path, models):
    dfs = {}
    for shft in sorted(models.keys()):
        m = 0
        dfs[shft] = {}
        dfs[shft][m] = pd.DataFrame()

        for model in sorted(models[shft]):
            dfs[shft][model] = pd.read_csv(
                path + 'df_prods_preds_h{}0.csv'.format(model), index_col=0)

    return dfs


def load_CS(df, shft=1):
    cs_file = '/scratch/gaa/alecat/data/clear_sky/cs_15min.npy'
    cs_columns_file = '/scratch/gaa/alecat/data/clear_sky/cs_15min_cols.npy'
    data = np.load(cs_file)
    columns = np.load(cs_columns_file)
    index = data[:, 0].astype(int)
    df_cs = pd.DataFrame(data[:, 1:], columns=columns, index=index)
    if shft > 0:
        df_cs_k = shift_df(df_cs, shft)
        df_cs_k.columns = list(map(lambda c: c + '-K', df_cs_k.columns))
        df_cs_total = pd.concat([df_cs, df_cs_k], axis=1, join='inner')
    else:
        df_cs_total = df_cs

    if not df.empty:
        return df.join(df_cs_total).sort_index(axis=1)
    return df_cs_total.sort_index(axis=1)  # sort_index will put variables at
    # the same coordinate alongside


def get_month_day_range(date):
    last_day = date + relativedelta(day=1, months=+1, days=-1)
    first_day = date + relativedelta(day=1)
    return first_day, last_day


def maes_month_hourly_generic(a,
                              b,
                              date,
                              daylight=True,
                              hours=None,
                              normalize=False):
    df = {}
    meses_list = [
        'January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December'
    ]
    if hours:
        index = sr.filter_hours(a.index, hours)
    elif daylight:
        index = sr.filter_daylight_hours(a.index)
    else:
        index = a.index

    fmt = "%Y%m%d%H"
    maes = abs(a - b).T.mean()
    for month in pd.date_range(date, periods=12, freq='M'):
        init, end = get_month_day_range(month)
        ixx = pd.date_range(
            start=init, end=end, freq='H').map(lambda x: int(x.strftime(fmt)))
        ix = np.intersect1d(ixx, index)
        hh = [sr.filter_hours(ix, [h]) for h in hours]
        maes_month = [maes.loc[h].mean() for h in hh]
        df[month.strftime("%B")] = maes_month

    return pd.DataFrame.from_dict(df, orient='index').loc[meses_list, hours]


def maes_month_hourly(df_prods_preds,
                      date,
                      daylight=True,
                      hours=None,
                      normalize=False):
    df = {}
    meses_list = [
        'January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December'
    ]
    if hours:
        index = sr.filter_hours(df_prods_preds.index, hours)
    elif daylight:
        index = sr.filter_daylight_hours(df_prods_preds.index)
    else:
        index = df_prods_preds.index

    fmt = "%Y%m%d%H"
    for month in pd.date_range(date, periods=12, freq='M'):
        init, end = get_month_day_range(month)
        ixx = pd.date_range(
            start=init, end=end, freq='H').map(lambda x: int(x.strftime(fmt)))
        ix = np.intersect1d(ixx, index)
        hh = [sr.filter_hours(ix, [h]) for h in hours]
        maes = [
            mean_absolute_error(df_prods_preds.loc[h, 'Prod'],
                                df_prods_preds.loc[h, 'Pred']) for h in hh
        ]
        values = maes
        if normalize:
            d = sr.filter_daylight_hours(ix)
            mae_norm = 100. * maes / df_prods_preds.loc[d, 'Prod'].mean()
            values = [maes, mae_norm]
        df[month.strftime("%B")] = values

    return pd.DataFrame.from_dict(df, orient='index').loc[meses_list]


def error_df(df_prods, df_preds, metric):
    return metric(np.clip(df_preds, 0, 105), df_prods)


def error_daylight(df_prods_preds):
    ix = sr.filter_daylight_hours(df_prods_preds.index)
    preds = df_prods_preds.loc[ix, 'Pred']
    prods = df_prods_preds.loc[ix, 'Prod']
    return error_df(prods, preds, mean_absolute_error)


def error_hours(df_prods_preds, hours):
    ix = sr.filter_hours(df_prods_preds.index, hours)
    preds = df_prods_preds.loc[ix, 'Pred']
    prods = df_prods_preds.loc[ix, 'Prod']
    return error_df(prods, preds, mean_absolute_error)


def group_by_hour(df):
    return df.groupby(
        df.index.map(lambda x: datetime.strptime(str(x), "%Y%m%d%H").hour))


def group_by_day(df):
    return df.groupby(
        df.index.map(lambda x:
                     (datetime.strptime(str(x), "%Y%m%d%H").day,
                      datetime.strptime(str(x), "%Y%m%d%H").month)))


def three_hourly(x):
    d = datetime.strptime(str(x), "%Y%m%d%H")
    hour = d.hour
    if hour in range(1, 4):
        d = d.replace(hour=3)
    elif hour in range(4, 7):
        d = d.replace(hour=6)
    elif hour in range(7, 10):
        d = d.replace(hour=9)
    elif hour in range(10, 13):
        d = d.replace(hour=12)
    elif hour in range(13, 16):
        d = d.replace(hour=15)
    elif hour in range(16, 19):
        d = d.replace(hour=18)
    elif hour in range(19, 22):
        d = d.replace(hour=21)
    else:
        d = d + timedelta(days=1)
        d = d.replace(hour=0)
    return int(d.strftime("%Y%m%d%H"))


def group_threehourly(df):
    return df.groupby(df.index.map(three_hourly))


def group_by_coordinate(df):
    regexp = '\([-]*[0-9]*.[0-9]*, [0-9]*.[0-9]*\)'
    return df.groupby(
        df.columns.map(lambda x: re.search(regexp, x).group(0)), axis=1)


def get_coordinate(column):
    regexp = '\([-]*[0-9]*.[0-9]*, [0-9]*.[0-9]*\)'
    return re.search(regexp, column).group(0)


def disaggregate(dm, variables):
    '''
    Disaggregate accumulated three hourly variables, such as radiation
    variables in ensembles datasets.

       `dm` has to be a DataMatrix_NWP object.
    '''
    df = dm.dataMatrix
    variables_cols = dm.query_cols(grid=dm.grid, tags=variables)
    df.loc[:, variables_cols] = df.loc[:, variables_cols].diff()
    df.loc[df[variables_cols] < 0] = 0
    return df


def accumulate_CS(cs_df):
    cs_3h_groups = dict(list(group_threehourly(cs_df)))
    cs_3h_df = pd.DataFrame(
        index=sorted(cs_3h_groups.keys()), columns=cs_df.columns)
    for t in cs_3h_groups:
        cs_3h_df.loc[t] = cs_3h_groups[t].sum()
    return cs_3h_df


def interpolate_by_clearsky(vars_df, cs_df, cs_acc_df):
    vars_coords = dict(list(group_by_coordinate(vars_df)))
    cs_coords = dict(list(group_by_coordinate(cs_df)))
    cs_acc_coords = dict(list(group_by_coordinate(cs_acc_df)))
    tmp_dfs = []

    # To parallelize this just put the body of this loop in a function that
    # receives as argument a certain coordinate `i` and returns `tmp` and use
    # multiprocessing library.
    for i in vars_coords:
        vars_i_df = vars_coords[i]
        cs_i_df = cs_coords[i]
        cs_acc_i_df = cs_acc_coords[i]
        tmp = pd.DataFrame(index=cs_df.index, columns=vars_df.columns)
        for t in cs_df.index:
            t_3h = three_hourly(t)
            # specifically to avoid 2016010100
            if t_3h in cs_i_df.index:
                # if any CS value is 0 the interpolation will be 0
                if cs_i_df.loc[t].any() or cs_acc_i_df.loc[t_3h].any():
                    cs_coef = cs_i_df.loc[t] / cs_acc_i_df.loc[t_3h]
                    tmp.loc[t] = cs_coef.values * vars_i_df.loc[t_3h]
                else:
                    tmp.loc[t] = 0
        print(i)
        tmp_dfs.append(tmp)
    return pd.concat(tmp_dfs, axis=1)


def compute_modules(m, res=0.125, v100=False):
    '''
    Computes the modules of the velocity and adds them
    to the dataMatrix already stored in the dataMatrix
    attribute.

    This is domain-related to weather forecast. Remove
    it for unrelated use.
    '''
    lat_l, lat_r = m.grid.get_lats()
    lon_l, lon_r = m.grid.get_lons()

    for i in np.arange(lat_l, lat_r + res, res):
        for j in np.arange(lon_l, lon_r + res, res):
            colv10 = '({0}, {1}) v10'.format(j, i)
            colU10 = '({0}, {1}) U10'.format(j, i)
            colV10 = '({0}, {1}) V10'.format(j, i)
            m.dataMatrix[colv10] = np.sqrt(m.dataMatrix[colU10]**2 +
                                           m.dataMatrix[colV10]**2)
            if v100:
                colv100 = '({0}, {1}) v100'.format(j, i)
                colU100 = '({0}, {1}) U100'.format(j, i)
                colV100 = '({0}, {1}) V100'.format(j, i)
                m.dataMatrix[colv100] = np.sqrt(m.dataMatrix[colU100]**2 +
                                                m.dataMatrix[colV100]**2)
    tags = m.tags + ['v10']
    if v100:
        tags += ['v100']
    m.dataMatrix = m.dataMatrix[m.query_cols(grid=m.grid, tags=tags)]


# main
pathin = "/scratch/gaa/alecat/data/eumetsat/processed/myp/15min/"
pathout = "/scratch/gaa/alecat/data/eumetsat/processed/myp/15min/"
prodsfile = "/scratch/gaa/alecat/data/prodsTotal.csv"
shp = "/scratch/gaa/alecat/dev/shapefiles/ESP_adm1.shp"
variables = ['IR_016 R']
variables_ext = ['VIS008 R', 'IR_016 R', 'IR_039 R', 'IR_039 K']
variables_nwp = ['SSRD', 'TCC', 'T2M', 'SSRC', 'v10']

# load data
m1, prods, land_grid = load_data(pathin, pathout, shp, prodsfile, 2013)
m2, _, _ = load_data(
    pathin, pathout, shp, prodsfile, 2014, land_grid=land_grid, prods=False)
m3, _, _ = load_data(
    pathin, pathout, shp, prodsfile, 2015, land_grid=land_grid, prods=False)

nwp1 = load_nwp_data('/scratch/gaa/alecat/data/ecmwf/',
                     '/scratch/gaa/alecat/data/ecmwf/', 2013)
nwp2 = load_nwp_data('/scratch/gaa/alecat/data/ecmwf/',
                     '/scratch/gaa/alecat/data/ecmwf/', 2014)
nwp3 = load_nwp_data('/scratch/gaa/alecat/data/ecmwf/',
                     '/scratch/gaa/alecat/data/ecmwf/', 2015)

dfs_train = {}
dfs_val = {}
dfs_test = {}

for shft in range(4):
    dfs_train[shft] = merge_satellite_nwp_prods(
        m1, nwp1, prods, land_grid, variables_ext, variables_nwp, shft)
    dfs_val[shft] = merge_satellite_nwp_prods(
        m2, nwp2, prods, land_grid, variables_ext, variables_nwp, shft)
    dfs_test[shft] = merge_satellite_nwp_prods(
        m3, nwp3, prods, land_grid, variables_ext, variables_nwp, shft)
    # dfs_train[shft] = load_df(m1, land_grid, prods, variables_ext, shft)
    # dfs_val[shft] = load_df(m2, land_grid, prods, variables_ext, shft)
    # dfs_test[shft] = load_df(m3, land_grid, prods, variables_ext, shft)

# CS model
models = {
    0: {
        0: list(range(8, 21))
    },
    1: {
        0: list(range(8, 21))
    },
    2: {
        0: list(range(8, 21))
    },
    3: {
        0: list(range(8, 21))
    }
}

for shft in models:
    for model in models[shft]:
        scalerX = None
        for dfs, suff in zip([dfs_train, dfs_val, dfs_test],
                             ["train", "val", "test"]):
            # path = "/scratch/gaa/data/dare/dareCSh{0}{1}.{2}"
            path = "/scratch/gaa/alecat/data/satellite_nwp_joined/{0}{1}_extended.{2}"
            output = path.format(shft, model, suff)
            dfs[shft] = load_CS(dfs[shft], shft)
            scalerX = load_store_X_Y(
                dfs[shft], models[shft][model], output, scalerX, standard=True)

# building the prods preds dataframe
dfs_prods_preds = load_preds(
    dfs_test, "/home/proyectos/ada2/alecat/experiments/results/lasso/", "npy",
    ["Pred"], models)
# dfs_coefs = load_preds(dfs_train,
#                        "/scratch/gaa/alecat/org/data/results/lasso/15minH/raw/",
#                        "coef", ["Coef"], models)

# set dark hours to zero
for shft in sorted(dfs_prods_preds.keys()):
    for model in sorted(dfs_prods_preds[shft]):
        ix = sr.filter_daylight_hours(dfs_prods_preds[shft][model].index)
        for index in sorted(set(dfs_test[shft].index) - set(ix)):
            dfs_prods_preds[shft][model].loc[index] = 0

# save and store dataframes
for shft in sorted(dfs_prods_preds.keys()):
    for model in sorted(dfs_prods_preds[shft]):
        dfs_prods_preds[shft][model].sort_index(inplace=True)
        dfs_prods_preds[shft][model].to_csv(
            "/home/proyectos/ada2/alecat/experiments/results/lasso/df/df_preds_h{0}{1}.csv".
            format(shft, model))
        dfs_prods_preds[shft][model] = dfs_prods_preds[shft][model].join(
            prods, how='inner')
        dfs_prods_preds[shft][model] = dfs_prods_preds[shft][model].clip(0,
                                                                         105)
        dfs_prods_preds[shft][model].to_csv(
            "/home/proyectos/ada2/alecat/experiments/results/lasso/df/df_prods_preds_h{0}{1}.csv".
            format(shft, model))

# report errors
for h in models:
    init = 8
    end = 21
    print("Errors h{}0 hours {}-{}: {}".format(
        h, init, end - 1,
        error_hours(dfs_prods_preds[h][0], hours=range(init, end))))

# import pandas as pd
# import glob
# import matplotlib.pyplot as plt
# from matplotlib import cm

# for f in glob.glob('/home/alex/org/*log*.csv'):
#     log = pd.read_csv(f, index_col=0)

#     cs = log['C'].reshape((10, 10))
#     values = log['value'].astype(float).reshape((10, 10))
#     gs = log['gamma'].reshape((10, 10))
#     es = log['epsilon'].reshape((10, 10))
#     levels = np.arange(values.min(), values[values < 3].max(), 0.01)

#     for p1, p2, t1, t2 in [[cs, gs, 'log C', 'log gamma'],
#                            [cs, es, 'log C', 'log epsilon'],
#                            [gs, es, 'log gamma', 'log epsilon']]:
#         plt.clf()
#         CS = plt.contourf(p1, p2, values, levels=levels, cmap=cm.jet)
#         plt.colorbar(CS)
#         plt.title('Contours of SVM tuning response surface')
#         plt.xlabel(t1)
#         plt.ylabel(t2)
