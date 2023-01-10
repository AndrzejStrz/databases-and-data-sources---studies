from datetime import date

import numpy
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from scipy.stats import shapiro
import warnings
from scipy.stats import ttest_rel, f_oneway, ttest_ind

warnings.filterwarnings('ignore')


# TODO walidacja
def download_and_validate(name_file):
    open(name_file, "wb").write(requests.get("https://stooq.pl/q/d/l/?s=btc.v&i=d").content)
    df = pd.read_csv(name_file)

    for x in df['Data']:
        if len(x) == 10 and x[4] == '-' and x[7] == '-':
            pass
        else:
            raise ValueError('Data nie jest w formacie yyyy-mm-dd')

    for x in df['Data']:
        if int(x[0:4]) < 2010:
            raise ValueError('Dane pochodza z czasow kiedy nie bylo bitcoina')
        if int(x[5:7]) > 12:
            raise ValueError('Miesiac ma wartosc wieksza niz 12')
        if int(x[8:10]) > 31:
            raise ValueError('Dzien ma wartosc wieksza niz 31')

    for column in [df['Otwarcie'], df['Najwyzszy'], df['Najnizszy'], df['Zamkniecie']]:
        for value in column:
            try:
                float(value)
            except ValueError or TypeError:
                raise ValueError('Kolumna', column, ' nie zawiera samych float')

    try:
        for item in df['Wolumen']:
            if item is not None:
                float(item)
    except ValueError or TypeError:
        raise ValueError('Kolumna Wolumen nie zawiera tylko float lub null ')

    return df


def show_stats(df):
    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 6)
    print('Zbiór zawiera {} obserwacji i {} zmiennych.'.format(df.shape[0], df.shape[1]))
    print(df.describe())
    print('Estymator niobciążony wariancji: \n', numpy.var(df, ddof=1))
    print(df.agg(['kurtosis', 'skew']).T)


def show_stats_for_selected_data(df, date_start, date_end):
    selected = df[(df['Data'] >= f'{date_start.year}-{date_start.month}-{date_start.day}') & (
            df['Data'] <= f'{date_end.year}-{date_end.month}-{date_end.day}')]
    show_stats(selected)


def hipoteza(connection):
    df = pd.read_sql_table('bitcoin.csv' + " " + str(date.today()), connection)

    df[['rok', 'miesiac', 'dzien']] = df.Data.str.split('-', expand=True)
    del df['index']
    df['rok'] = df['rok'].astype(int)
    df['miesiac'] = df['miesiac'].astype(int)
    df['dzien'] = df['dzien'].astype(int)

    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 6)
    print(df)

    # Hipoteza
    # Średnia najwyższych cena bitcoina z każdego dnia rośnie co roku

    mean_najwyzszy = df.groupby(['rok'], as_index=False).agg({'Najwyzszy': ['mean']}).round(2)

    plt.plot(mean_najwyzszy['rok'], mean_najwyzszy['Najwyzszy'])
    plt.show()
    stat1, p1 = shapiro(df['Najwyzszy'])
    stat2, p2 = shapiro(df['rok'])

    if p1 > 0.05 and p2 > 0.05:
        print('Dane pochodzą z rozkładu normalnego')
    else:
        print('Dane nie pochodzą z rozkładu normalnego')

    stat, p = ttest_rel(df['rok'], df['Najwyzszy'])

    p = round(p, 6)
    print(p)
    if p < 0.05:
        print('Różnice są istotne statystycznie')
    else:
        print('Różnice nie są istotne statystycznie')

    stat, p = ttest_ind(df['Otwarcie'], df['Najnizszy'])
    print(p)
    if p < 0.05:
        print('Różnice są istotne statystycznie')
    else:
        print('Różnice nie są istotne statystycznie')

    stat, p = f_oneway(df['Najnizszy'], df['Zamkniecie'])
    print(p)
    if p < 0.05:
        print('Różnice są istotne statystycznie')
    else:
        print('Różnice nie są istotne statystycznie')
