import unittest
import numpy as np
import pandas

# if you're looking to test your knowledge of what's been covered
# so far, try filling out the functions of the class below.
# then, you can run this file, and the tests at the bottom
# of this file will run. If you pass all the tests, you've
# got it!

class BasicChallenges:
    def constructs_dataframe(self, data_a, data_b, data_c):
        # data_a, data_b, and data_c are lists of equal length
        # return a pandas dataframe with three columns, one for each
        # list above
        pass

    def returns_range_from_dataframe(self, df, r):
        # df is a pandas dataframe with a column called 'data_a'
        # return the range of values "r" in that column
        pass

    def returns_all_greater_than(self, df, val):
        # df is a pandas dataframe with a column called 'data_b'
        # return all the values in that column that are greater than val
        pass

class AdvancedChallenges:
    def returns_dataframe_with_new_column(self, df, col_name, col_data):
        # df is a pandas dataframe
        # col_name is a string
        # col_data is a list
        # return a new dataframe with the new column added
        pass

    def returns_dataframe_with_new_column_from_function(self, df, col_name, func):
        # df is a pandas dataframe
        # col_name is a string
        # func is a function that takes a single argument
        # return a new dataframe with the new column added
        pass

    def returns_nans_replaced_with_zeros(self, df):
        # df is a pandas dataframe
        # return a new dataframe with all the NaNs replaced with zeros
        pass

    def returns_dataframe_with_new_column_summing_columns(self, df):
        # df is a pandas dataframe with columns 'data_a' and 'data_b'
        # return a new dataframe with a new column 'data_c' that is the sum
        # of the other two columns
        pass


class BasicChallengesTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BasicChallengesTests, self).__init__(*args, **kwargs)
        self.bc = BasicChallenges()

    def test_constructs_dataframe(self):
        data_a = [1, 2, 3]
        data_b = [4, 5, 6]
        data_c = [7, 8, 9]
        df = self.bc.constructs_dataframe(data_a, data_b, data_c)
        self.assertIsInstance(df, pandas.DataFrame)
        self.assertEqual(len(df.columns), 3)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.columns[0], 'data_a')
        self.assertEqual(df.columns[1], 'data_b')
        self.assertEqual(df.columns[2], 'data_c')
        self.assertEqual(df['data_a'].tolist(), data_a)
        self.assertEqual(df['data_b'].tolist(), data_b)
        self.assertEqual(df['data_c'].tolist(), data_c)

    def test_returns_range_from_dataframe(self):
        df = pandas.DataFrame({'data_a': [1, 2, 3, 4, 5]})
        r = 3
        self.assertEqual(self.bc.returns_range_from_dataframe(df, r), [1, 2, 3])

    def test_returns_all_greater_than(self):
        df = pandas.DataFrame({'data_b': [1, 2, 3, 4, 5]})
        val = 3
        self.assertEqual(self.bc.returns_all_greater_than(df, val), [4, 5])

class AdvancedChallengesTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(AdvancedChallengesTests, self).__init__(*args, **kwargs)
        self.ac = AdvancedChallenges()

    def test_returns_dataframe_with_new_column(self):
        df = pandas.DataFrame({'data_a': [1, 2, 3, 4, 5]})
        col_name = 'data_b'
        col_data = [6, 7, 8, 9, 10]
        new_df = self.ac.returns_dataframe_with_new_column(df, col_name, col_data)
        self.assertIsInstance(new_df, pandas.DataFrame)
        self.assertEqual(len(new_df.columns), 2)
        self.assertEqual(len(new_df), 5)
        self.assertEqual(new_df.columns[0], 'data_a')
        self.assertEqual(new_df.columns[1], 'data_b')
        self.assertEqual(new_df['data_a'].tolist(), df['data_a'].tolist())
        self.assertEqual(new_df['data_b'].tolist(), col_data)

    def test_returns_dataframe_with_new_column_from_function(self):
        df = pandas.DataFrame({'data_a': [1, 2, 3, 4, 5]})
        col_name = 'data_b'
        func = lambda x: x * 2
        new_df = self.ac.returns_dataframe_with_new_column_from_function(df, col_name, func)
        self.assertIsInstance(new_df, pandas.DataFrame)
        self.assertEqual(len(new_df.columns), 2)
        self.assertEqual(len(new_df), 5)
        self.assertEqual(new_df.columns[0], 'data_a')
        self.assertEqual(new_df.columns[1], 'data_b')
        self.assertEqual(new_df['data_a'].tolist(), df['data_a'].tolist())
        self.assertEqual(new_df['data_b'].tolist(), [2, 4, 6, 8, 10])

    def test_returns_nans_replaced_with_zeros(self):
        df = pandas.DataFrame({'data_a': [1, 2, 3, 4, 5], 'data_b': [6, 7, np.nan, 9, 10]})
        new_df = self.ac.returns_nans_replaced_with_zeros(df)
        self.assertIsInstance(new_df, pandas.DataFrame)
        self.assertEqual(len(new_df.columns), 2)
        self.assertEqual(len(new_df), 5)
        self.assertEqual(new_df.columns[0], 'data_a')
        self.assertEqual(new_df.columns[1], 'data_b')
        self.assertEqual(new_df['data_a'].tolist(), df['data_a'].tolist())
        self.assertEqual(new_df['data_b'].tolist(), [6, 7, 0, 9, 10])

    def test_returns_dataframe_with_new_column_summing_columns(self):
        df = pandas.DataFrame({'data_a': [1, 2, 3, 4, 5], 'data_b': [6, 7, 8, 9, 10]})
        new_df = self.ac.returns_dataframe_with_new_column_summing_columns(df)
        self.assertIsInstance(new_df, pandas.DataFrame)
        self.assertEqual(len(new_df.columns), 3)
        self.assertEqual(len(new_df), 5)
        self.assertEqual(new_df.columns[0], 'data_a')
        self.assertEqual(new_df.columns[1], 'data_b')
        self.assertEqual(new_df.columns[2], 'data_c')
        self.assertEqual(new_df['data_a'].tolist(), df['data_a'].tolist())
        self.assertEqual(new_df['data_b'].tolist(), df['data_b'].tolist())
        self.assertEqual(new_df['data_c'].tolist(), [7, 9, 11, 13, 15])

if __name__ == '__main__':
    # run the tests
    unittest.main()
