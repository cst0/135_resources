import pandas
import numpy as np


def print_info(df):
    print("---")
    print(f"frame:\n{df}")
    print(f"types:   {df.dtypes}")
    print(f"shape:   {df.shape}")
    # print(f"columns: {df.columns}")
    print("---")


print(
    """
In basics.py, we saw how to make dataframes and some basic manipulations we
could perform. But, often our data will be more complicated, and will require
more complicated manipulations.

To explore these, we'll make some new data. This time, we'll make two
dataframes that are time-series data. To make things more complicated, they
might operate at different frequencies (maybe one is every second and the other
is every minute), and they might not have the same number of entries (sometimes
we failed to record data).
"""
)

# make some random data, masking out 30% of the data
random_data_points_a = np.ma.masked_array(np.random.randn(20), np.random.rand(20) < 0.3)
random_data_points_b = np.ma.masked_array(np.random.randn(20), np.random.rand(20) < 0.3)
random_data_points_c = np.ma.masked_array(np.random.randn(20), np.random.rand(20) < 0.3)

# now let's pick 3 random starting times (within a little bit of each other)
#fmt: off
random_start_times = np.random.randint(0, 10, 3)
random_time_a = pandas.date_range(start=f"2024-01-01 00:00:00", periods=20, freq="S") + pandas.Timedelta(seconds=random_start_times[0])
random_time_b = pandas.date_range(start=f"2024-01-01 00:00:00", periods=20, freq="S") + pandas.Timedelta(seconds=random_start_times[1])
random_time_c = pandas.date_range(start=f"2024-01-01 00:00:00", periods=20, freq="S") + pandas.Timedelta(seconds=random_start_times[2])
#fmt: on

# now we can make our dataframes, indexed on time
df_a = pandas.DataFrame({"a": random_data_points_a}, index=random_time_a)
df_b = pandas.DataFrame({"b": random_data_points_b}, index=random_time_b)
df_c = pandas.DataFrame({"c": random_data_points_c}, index=random_time_c)

print("""
Now we've got three dataframes, each with a single column, and each indexed on
time.

We could try to concatenate these together, but that won't work, because they
have different indices. Instead, we'll use the pandas merge function, which
allows us to specify how we want to merge the dataframes together. In this
case, we'll use the "outer" method, which will keep all the data, and fill in
NaN's where there isn't any data.
""")
df_merged = pandas.merge(df_a, df_b, how="outer", left_index=True, right_index=True)
df_merged = pandas.merge(df_merged, df_c, how="outer", left_index=True, right_index=True)

print_info(df_merged)

print("""
Now we've got a single dataframe with all the data, but we've got a lot of
NaN's in there. We could use the pandas fillna function to fill in those NaN's
with whatever we want. In this case, though, we can use a function called
interpolate, which will fill in the NaN's with interpolated values. This is
useful for time-series data, because we can fill in the NaN's with values that
are likely to be close to the actual values. This doesn't work for the
datapoints at the beginning and end of the data, though, since we don't have
any data to interpolate from. So, we'll just drop those rows.
""")
df_merged = df_merged.interpolate(method="time")
df_merged = df_merged.dropna()

print_info(df_merged)

print("""
So we've seen that Pandas gives us some great tools for manipulating our
data, but what if we need to use methods that aren't available there?

We can do that using the 'apply' function. This function takes a function as
an argument, and applies that function to each row or column of the dataframe.
You can use existing functions, like those from Numpy, or you can create
your own.

For example, let's say we want
to take the square root of each value in column 'a', but only if the value in
column 'b' is greater than 0, and if so, we want to multiply the value in
column 'c' by 2. We just need to define a function that does that, and
then Pandas will handle it once we provide that function to 'apply'.
""")
def complicated_function(row):
    if row["b"] > 0:
        return np.sqrt(abs(row["a"])) * row["c"]
    else:
        return row["a"]

df_merged["new_a"] = df_merged.apply(complicated_function, axis=1)
print_info(df_merged)

print("""
We can also use apply to access data outside of the current row. For example,
let's say we want to calculate the mean of the values in column 'a' for the
previous 5 rows. We can do that by using the rolling function, which will
return a rolling window of the data. Then, we can use apply to calculate the
mean of that window.
""")

df_merged["rolling_mean"] = df_merged["a"].rolling(5).apply(np.mean)
print_info(df_merged)

print("""
And, since at the end of the day these are just custom-defined functions, we
can access data outside of them, too. For example, let's say we have an array
hat has time-series data on the same indexing, but it has boolean
values. If the values are true we can leave the data alone, but if it's false,
we need to multiply the value in column 'a' by 2. We can do that by
defining a function that accesses the external dataframe, and then using
apply to apply that function to each row.
""")

external_boolean_values = np.random.rand(20) > 0.5
def access_external_df(row):
    target_date = row.name
    target_integer_index = df_merged.index.get_loc(target_date)
    if external_boolean_values[target_integer_index]:
        return row["a"]
    else:
        return row["a"] * 2
df_merged["new_a"] = df_merged.apply(access_external_df, axis=1)
print_info(df_merged)

