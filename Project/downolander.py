from datetime import date

import pandas as pd

import validate
from sqlalchemy import create_engine

name_file = 'bitcoin.csv'

df = validate.download_and_validate(name_file)

engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres")
connection = engine.connect()

df = df.to_sql(name_file, connection, if_exists='replace')
# df = df.to_sql(name_file + " " + str(date.today()), connection, if_exists='replace')

# validate.show_stats(df)
# validate.show_stats_for_selected_data(df, date(2011, 11, 1), date(2011, 11, 23))
# validate.hipoteza(connection)
# validate.regresja(pd.read_sql_table('bitcoin.csv', connection), df)

# validate.dodaj_rekord(engine,'bitcoin.csv',['2011-11-23',12,1,1,1,1])
# validate.usun_rekord(engine,'bitcoin.csv',4569)
validate.update_rekord(engine,'bitcoin.csv',2,['2013-11-11',12,1,1,1,1])