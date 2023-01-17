from datetime import date

import pandas as pd

import validate
from sqlalchemy import create_engine

name_file = 'bitcoin.csv'

df = validate.download_and_validate(name_file)

engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres")
connection = engine.connect()

if not name_file + " " + str(date.today()) == engine.table_names()[-1]:
    df = df.to_sql(name_file + " " + str(date.today()), connection)

# validate.show_stats(df)
# validate.show_stats_for_selected_data(df, date(2011, 11, 1), date(2011, 11, 23))
# validate.hipoteza(connection)
# validate.regresja(pd.read_sql_table('bitcoin.csv', connection), df)

# validate.dodaj_rekord(engine,'bitcoin.csv',[['1985-11-13',12,1,1,1,1]])