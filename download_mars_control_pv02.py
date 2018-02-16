from ecmwfapi import ECMWFService
import os
import pandas as pd
from multiprocessing import Pool


def download_mars(d):
    print(d)
    if not os.path.isfile(
            "pv_control_ensembles_{}.nc".format(d.strftime("%Y%m%d"))):
        server.execute(
            {
                "class": "od",
                "date": "{}".format(d.strftime("%Y-%m-%d")),
                "expver": "1",
                "levtype": "sfc",
                "param": "21.228/22.228/164.128/165.128/166.128/167.128/169.128/176.128/210.128",
                "step": "0/to/21/by/3",
                "stream": "ef",
                "time": "00:00:00",
                "area": "44/-9.5/35/4.5",
                "grid": "0.5/0.5",
                "format": "netcdf",
                "type": "control forecast"
            },
            "pv_control_ensembles_{}.nc".format(d.strftime("%Y%m%d")), )
    return d


if __name__ == '__main__':
    server = ECMWFService('mars')
    dates = pd.date_range('20130101', '20140101', freq='D')
    p = Pool(1)
    p.map(download_mars, dates)
