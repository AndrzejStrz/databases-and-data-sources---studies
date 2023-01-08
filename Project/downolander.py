import requests
from validate import validate
from sqlalchemy import create_engine
import pandas as pd

name_file = 'bitcoin.csv'
open(name_file, "wb").write(requests.get("https://stooq.pl/q/d/l/?s=btc.v&i=d").content)

validate(name_file)

engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres")
connection = engine.connect()

df = pd.read_csv(name_file, sep=',')
df = df.to_sql(name_file, connection)
