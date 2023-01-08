import numpy
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings('ignore')

# TODO walidacja
def download_and_validate(name_file):
    open(name_file, "wb").write(requests.get("https://stooq.pl/q/d/l/?s=btc.v&i=d").content)
    df = pd.read_csv(name_file)

    for x in df['Data']:
        if len(x) == 10 and x[4] == '-' and x[7] == '-':
            pass
        else:
            raise ValueError('Date column does not contain only date in yyyy-mm-dd format')

    return df


def show_stats(df, connection):
    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 6)
    print('Zbiór zawiera {} obserwacji i {} zmiennych.'.format(df.shape[0], df.shape[1]))
    print(df.describe())
    print('Estymator niobciążony wariancji: \n', numpy.var(df, ddof=1))
    print(df.agg(['kurtosis', 'skew']).T)


def show_stats_for_selected_data(df, date_start,date_end,connection):
    selected = df[(df['Data'] >= f'{date_start.year}-{date_start.month}-{date_start.day}') & (df['Data'] <= f'{date_end.year}-{date_end.month}-{date_end.day}')]
    show_stats(selected,connection)