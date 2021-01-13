using DrWatson
using CSV
using DataFrames
using Dates

path = datadir("exp_raw/UNT330A_20200918_115833.csv")

data = CSV.read(path)

processed_data = data[:, [2, 3]]
dropmissing!(processed_data)
colnames = ["DateTime", "Temperature"]
rename!(processed_data, Symbol.(colnames))

t_date = processed_data.DateTime[1]

dtfmt = dateformat"yyyy-mm-dd HH:MM:SS"

t_date_f = DateTime(t_date, dtfmt)
d_col = DateTime.(processed_data.DateTime, dtfmt)

processed_data.DateTime = d_col


CSV.write(datadir("exp_pro/dta_1.csv"), processed_data;)
