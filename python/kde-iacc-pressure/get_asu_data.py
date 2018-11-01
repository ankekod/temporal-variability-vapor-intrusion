import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

db_dir = 'C://Users/jstroem/lib/vapor-intrusion-dbs/'

db_asu = sqlite3.connect(db_dir + 'hill-afb.db')

"""asu = pd.read_sql_query(
    "SELECT \
        date(C.time) as t, \
        AVG(C.concentration) AS c, \
        P.pressure AS p \
    FROM \
        \"td-basement\" AS C, \
        daily_average_pressure AS P \
    GROUP BY \
        date(C.time);",
    db_asu,
)"""



asu = pd.read_sql_query(
    "SELECT \
        date(C.time) as t, \
        AVG(C.concentration) AS c, \
        AVG(P.pressure) AS p \
    FROM \
        \"td-basement\" AS C \
    INNER JOIN daily_average_pressure on date(C.time) == date(daily_average_pressure.time);",
    db_asu,
)

print(asu)


asu.plot(x='t',y=['c','p'])
plt.show()
