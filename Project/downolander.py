from datetime import date
import validate
from sqlalchemy import create_engine



name_file = 'bitcoin.csv'

df = validate.download_and_validate(name_file)

engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres")
connection = engine.connect()


if not name_file+" "+str(date.today()) == engine.table_names()[-1]:
    df = df.to_sql(name_file+" "+str(date.today()), connection)


validate.show_stats(df, connection)
validate.show_stats_for_selected_data(df, date(2011,11,1), date(2011,11,23),connection)