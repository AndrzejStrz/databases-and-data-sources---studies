from datetime import date, datetime
from sklearn.linear_model import LinearRegression, Ridge

import numpy
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from scipy.stats import shapiro
import warnings
from scipy.stats import ttest_rel, f_oneway, ttest_ind
from sklearn import metrics



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


from sklearn.model_selection import train_test_split
def regresja(df_old, df_now):

    df_old['Data'] = pd.to_datetime(df_old['Data'])
    df_old = df_old.reset_index(drop=True)

    x = df_old[['Otwarcie', 'Najwyzszy', 'Najnizszy']]
    y = df_old['Zamkniecie'].shift(-1)
    y = y[:-1]
    x = x[:-1]

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.7, shuffle=False, random_state=0)
    regression = LinearRegression()
    regression.fit(train_x.values, train_y.values)
    print("regression coefficient", regression.coef_)
    print("regression intercept", regression.intercept_)
    regression_confidence = regression.score(test_x.values, test_y.values)

    print("linear regression confidence: ", regression_confidence)
    predicted = regression.predict(test_x.values)

    dfr = pd.DataFrame({'Actual_Price': test_y, 'Predicted_Price': predicted})
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(test_y, predicted))
    print('Mean Squared Error (MSE) :', metrics.mean_squared_error(test_y, predicted))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(test_y, predicted)))
    plt.scatter(dfr.Actual_Price, dfr.Predicted_Price, color='Darkblue')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.show()

    print('Tomorrow predict:', regression.predict(df_now[['Otwarcie', 'Najwyzszy', 'Najnizszy']].tail(1)))

    print('-----')
    ridgeModelChosen = Ridge(alpha=3000, copy_X=False, random_state=32)
    ridgeModelChosen.fit(train_x, train_y)

    print("regression coefficient", ridgeModelChosen.coef_)
    print("regression intercept", ridgeModelChosen.intercept_)
    regression_confidence = ridgeModelChosen.score(test_x.values, test_y.values)

    print("linear regression confidence: ", regression_confidence)
    predicted = ridgeModelChosen.predict(test_x.values)

    dfr = pd.DataFrame({'Actual_Price': test_y, 'Predicted_Price': predicted})
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(test_y, predicted))
    print('Mean Squared Error (MSE) :', metrics.mean_squared_error(test_y, predicted))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(test_y, predicted)))
    plt.scatter(dfr.Actual_Price, dfr.Predicted_Price, color='Darkblue')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.show()

    print('Tomorrow predict:', ridgeModelChosen.predict(df_now[['Otwarcie', 'Najwyzszy', 'Najnizszy']].tail(1)))

def dodaj_rekord(engine, table, data):
    df = pd.DataFrame(data, columns = ['Data', 'Otwarcie', 'Najwyzszy', 'Najnizszy', 'Zamkniecie', 'Wolumen'])
    df_2 = pd.read_sql_table(table, engine.connect())['index']
    df.insert(0,'index', df_2[len(df_2)-1]+1, True)

    logi = [datetime.now(), 'Do tabeli '+str(table)+' dodano wiersz o indeksie '+ str(df_2[len(df_2)-1]+1)]
    df_logi =pd.DataFrame(logi)
    df_logi.T.to_sql('logi', engine, if_exists='append', index=False)

    df.to_sql(table, engine, if_exists='append', index=False)

def usun_rekord(engine, table, index_to_drop):
    df = pd.read_sql_table(table, engine.connect())
    df[df['index'] == index_to_drop].to_sql('usuniete', engine, if_exists='append', index=False)
    df = df[df['index'] != index_to_drop]
    df.to_sql(table, engine, if_exists='replace', index=False)

    logi = [datetime.now(), 'Z tabeli ' + str(table) + ' usunięto wiersz o indeksie ' + str(index_to_drop)]
    df_logi = pd.DataFrame(logi)
    df_logi.T.to_sql('logi', engine, if_exists='append', index=False)

def update_rekord(engine, table, index_to_update, dane):
    df = pd.read_sql_table(table, engine.connect())
    df[df['index'] == index_to_update] = dane
    df.to_sql(table, engine, if_exists='replace', index=False)

    logi = [datetime.now(), 'Na tabeli ' + str(table) + ' zmieniono wiersz o indeksie ' + str(index_to_update)]
    df_logi = pd.DataFrame(logi)
    df_logi.T.to_sql('logi', engine, if_exists='append', index=False)
