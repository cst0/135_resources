import numpy as np
import pandas

print(
    """
Pandas is a library for working with dataframes. Dataframes are a way of
storing data in a tabular format. They're a lot like numpy arrays, but they
have some extra features that make them more useful for storing data.

Let's start by making a dataframe from some random data, which we'll be able to
print out and look at.
"""
)
random_data_a = np.random.randn(10)
random_data_b = np.random.randn(10)
my_dataframe = pandas.DataFrame({"a": random_data_a, "b": random_data_b})


# we can get all kinds of basic information about it:
def print_info(df):
    print("---")
    print(f"frame:\n{df}")
    print(f"types:   {df.dtypes}")
    print(f"shape:   {df.shape}")
    print(f"columns: {df.columns}")
    print("---")


print_info(my_dataframe)

print(
    """
Now let's say we've got some more data we want to add to the dataframe.
Maybe it's a numpy array, or a list. Notice how it's not the same length as
the other entries in the dataframe: That's not going to be okay!
"""
)
new_data = np.random.randn(15)
try:
    my_dataframe["c"] = new_data
except ValueError as e:
    print(f"Error: {e}")

print(
    """
Instead, we need to make sure that the new data is the same length as the
existing data. To do this without dropping any new data, let's make a new
dataframe for this data. Then, we'll be able to concatenate the two dataframes
together.
"""
)
new_dataframe = pandas.DataFrame({"c": new_data})
my_dataframe = pandas.concat([my_dataframe, new_dataframe], axis=1)
print_info(my_dataframe)

print(
    """
Notice that we've got some NaN's in there now! That's because the new data
didn't have any values for the final 5 rows, so it just filled them in with
NaN's. We can deal with this later.

With all these columns, now we can start to access the data in different ways.
The simplest is to just grab a column, or an index, but we can also grab
multiple columns, or multiple indices, or even a subset of the data.
"""  # fmt: off
)
col_a = my_dataframe["a"]  # select the column named "a"
index_0 = my_dataframe.index[0]  # select the index at position 0
col_a_index_0 = my_dataframe["a"][0]  # select the val of column a at 0
col_a_first_five = my_dataframe["a"][:5]  # select the first 5 vals of a
col_a_b_first_five = my_dataframe[["a", "b"]][:5]  # select first 5 of a+b
# fmt: on

print(
    """
We can also select rows based on some condition, which is really useful! For
example, let's say we want to select all the values of a where the value in
column "b" is greater than 0.

We can do this by using our regular python comparison operators, but inside the
[]. This will return a boolean array, which we can then use to select the rows
we want.

We could also deal with our NaN's that way, but we don't need to, since pandas
has a built-in function for that: using dropna() will drop all rows that have
NaN's in them. Another option would be to use this same indexing technique to
select all the rows with NaN's in them, and then replace them with some other
value.
"""
)

b_gt_zero = [my_dataframe["b"] > 0]
col_a_where_b_gt_zero = my_dataframe["a"][my_dataframe["b"] > 0]

my_dataframe_no_nans = my_dataframe.dropna()
my_dataframe_replaced_nans = my_dataframe.fillna(0)

print(
    """
There are also convenient functions for doing some pretty common things. The
'sort_values' function is handy for sorting the dataframe based on some column,
but there's also a 'sort_index' function for sorting based on the index.

Pandas also has a bunch of built-in functions for doing things like calculating
the mean, median, and standard deviation of the data. These functions will
ignore NaN's by default, but you can change that by passing in the argument
'skipna=False'.
"""
)
my_dataframe_sorted = my_dataframe.sort_values(by="a")
print_info(my_dataframe_sorted)
print(f"Notice how our index is scrambled now:\n{my_dataframe_sorted.index}")

my_dataframe_mean = my_dataframe.mean()
my_dataframe_std = my_dataframe.std()
my_dataframe_median = my_dataframe.median()
print(f"mean:\n{my_dataframe_mean}")
print(f"std:\n{my_dataframe_std}")
print(f"median:\n{my_dataframe_median}")
print("---")

print(
    """
We're also not limited to numbers! We can store strings, or even other objects,
in a dataframe. This is really useful for storing metadata about the data in
the dataframe, like a classification label.
    """
)
classifications = np.random.choice(["red", "green", "blue", "indigo"], size=15)
my_dataframe["classifications"] = classifications
print_info(my_dataframe)

print(
    """
Finally, we can also save our dataframes to a file, and load them back in later.
"""
)
my_dataframe.to_csv("my_dataframe.csv")
my_dataframe_loaded = pandas.read_csv("my_dataframe.csv")
print_info(my_dataframe_loaded)
print(
    """
We've only scratched the surface of what pandas can do, but this should
provide a good starting point for working with dataframes. For more
information, check out the pandas documentation at
https://pandas.pydata.org/pandas-docs/stable/index.html
"""
)
