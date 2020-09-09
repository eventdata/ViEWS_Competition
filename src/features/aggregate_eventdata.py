import pandas as pd
import numpy as np
import json

def data_concatenation(years):
    filepaths = ['data/processed/ICEWS' + year + '_agg.json' for year in years]
    df = pd.concat(map(pd.read_json, filepaths))
    df.reset_index(inplace=True, drop=True)
    # for year in years:
    #     _dir = 'data/processed/ICEWS' + year + '_agg.json'
    #     df = pd.read_json(_dir)
    return df




if __name__ == '__main__':
    years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
    df = data_concatenation(years)
    df['month'] = (df['year'] - 1980) * 12 + df['month']
    df.to_csv('data/processed/pgm_africa_utd.csv', index=False)
    print('here')