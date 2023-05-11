# python_portfolio
This is the portfolio of python code that I learned during MSNT 510.

## using_a_jupyter_notebook
What is Jupyter Notebook?
```python
%matplotlib inline
```


```python
import pandas as pd
```


```python
import matplotlib.pyplot as plt
```


```python
import seaborn as sns
```


```python
sns.set(
    style = "darkgrid"
)
```


```python
df = pd.read_csv(
    'fortune500.csv'
)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25495</td>
      <td>2005</td>
      <td>496</td>
      <td>Wm. Wrigley Jr.</td>
      <td>3648.6</td>
      <td>493</td>
    </tr>
    <tr>
      <td>25496</td>
      <td>2005</td>
      <td>497</td>
      <td>Peabody Energy</td>
      <td>3631.6</td>
      <td>175.4</td>
    </tr>
    <tr>
      <td>25497</td>
      <td>2005</td>
      <td>498</td>
      <td>Wendy's International</td>
      <td>3630.4</td>
      <td>57.8</td>
    </tr>
    <tr>
      <td>25498</td>
      <td>2005</td>
      <td>499</td>
      <td>Kindred Healthcare</td>
      <td>3616.6</td>
      <td>70.6</td>
    </tr>
    <tr>
      <td>25499</td>
      <td>2005</td>
      <td>500</td>
      <td>Cincinnati Financial</td>
      <td>3614.0</td>
      <td>584</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = [
    'year', 
    'rank', 
    'company', 
    'revenue', 
    'profit'
]
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(
    df
)
```




    25500




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit      object
    dtype: object




```python
non_numberic_profits = df.profit.str.contains(
    '[^0-9.-]'
)
```


```python
df.loc[
    non_numberic_profits
].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>228</td>
      <td>1955</td>
      <td>229</td>
      <td>Norton</td>
      <td>135.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>290</td>
      <td>1955</td>
      <td>291</td>
      <td>Schlitz Brewing</td>
      <td>100.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>294</td>
      <td>1955</td>
      <td>295</td>
      <td>Pacific Vegetable Oil</td>
      <td>97.9</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>296</td>
      <td>1955</td>
      <td>297</td>
      <td>Liebmann Breweries</td>
      <td>96.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>352</td>
      <td>1955</td>
      <td>353</td>
      <td>Minneapolis-Moline</td>
      <td>77.4</td>
      <td>N.A.</td>
    </tr>
  </tbody>
</table>
</div>




```python
set(
    df.profit[
        non_numberic_profits
    ]
)
```




    {'N.A.'}




```python
len(
    df.profit[
        non_numberic_profits
    ]
)
```




    369




```python
bin_sizes, _, _ = plt.hist(
    df.year[
        non_numberic_profits
    ], 
    bins = range(
        1955, 
        2006
    )
)
```


![output_16_0](https://github.com/dalemajorjr/python_portfolio/assets/132947457/ee69c7d2-17e1-4b7d-be4b-c31fd902ecf1)




```python
df = df.loc[
    ~non_numberic_profits
]
```


```python
df.profit = df.profit.apply(
    pd.to_numeric
)
```


```python
len(
    df
)
```




    25131




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit     float64
    dtype: object




```python
group_by_year = df.loc[
    :, 
    [
        'year', 
        'revenue', 
        'profit'
    ]
].groupby(
    'year'
)
```


```python
avgs = group_by_year.mean()
```


```python
x = avgs.index
```


```python
y1 = avgs.profit
```


```python
def plot(
    x, 
    y, 
    ax, 
    title, 
    y_label
):
    ax.set_title(
        title
    )
    ax.set_ylabel(
        y_label
    )
    ax.plot(
        x, 
        y
    )
    ax.margins(
        x = 0, 
        y = 0
    )
```


```python
fig, ax = plt.subplots()

plot(
    x, 
    y1, 
    ax, 
    'Increase in mean Fortune 500 company profits from 1955 to 2005', 
    'Profit (millions)'
)
```


![output_26_0](https://github.com/dalemajorjr/python_portfolio/assets/132947457/82a51f18-a5c5-4955-a349-cafd3e5b45d2)


```python
y2 = avgs.revenue
```


```python
fig, ax = plt.subplots()

plot(
    x, 
    y2, 
    ax, 
    'Increase in mean Fortune 500 company revenues from 1955 to 2005', 
    'Revenue (millions)'
)
```


![output_28_0](https://github.com/dalemajorjr/python_portfolio/assets/132947457/9be6a697-547f-47b7-aeb7-b4305580a339)




```python
def plot_with_std(
    x, 
    y, 
    stds, 
    ax, 
    title, 
    y_label
):
    ax.fill_between(
        x, 
        y - stds, 
        y + stds, 
        alpha = 0.2
    )
    plot(
        x, 
        y, 
        ax, 
        title, 
        y_label
    )
```


```python
fig, (
    ax1, 
    ax2
) = plt.subplots(
    ncols= 2
)

title = 'Increase in mean and std Fortune 500 company %s from 1955 to 2005'

stds1 = group_by_year.std().profit.values

stds2 = group_by_year.std().revenue.values

plot_with_std(
    x, 
    y1.values, 
    stds1, 
    ax1, 
    title % 'profits', 
    'Profit (millions)'
)

plot_with_std(
    x, 
    y2.values, 
    stds2, 
    ax2, 
    title % 'revenues', 
    'Revenue (millions)'
)

fig.set_size_inches(
    14, 
    4
)

fig.tight_layout()
```


![output_30_0](https://github.com/dalemajorjr/python_portfolio/assets/132947457/2208264c-2d16-4dc4-9954-fb68e1f2cb92)

## analyzing_data 
How can I process tabular data files in Python? 
```python
import numpy
```


```python
numpy.loadtxt(
    fname = 'inflammation-01.csv', 
    delimiter = ','
)
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
data = numpy.loadtxt(
    fname = 'inflammation-01.csv', 
    delimiter = ','
)
```


```python
print(
    data
)
```

    [[0. 0. 1. ... 3. 0. 0.]
     [0. 1. 2. ... 1. 0. 1.]
     [0. 1. 1. ... 2. 1. 1.]
     ...
     [0. 1. 1. ... 1. 1. 1.]
     [0. 0. 0. ... 0. 2. 0.]
     [0. 0. 1. ... 1. 1. 0.]]



```python
print(
    type(
        data
    )
)
```

    <class 'numpy.ndarray'>



```python
print(
    data.dtype
)
```

    float64



```python
print(
    data.shape
)
```

    (60, 40)



```python
print(
    'first value in data:', 
    data[
        0, 
        0
    ]
)
```

    first value in data: 0.0



```python
print(
    'middle value in data:', 
    data[
        29, 
        19
    ]
)
```

    middle value in data: 16.0



```python
print(
    data[
        0:4, 
        0:10
    ]
)
```

    [[0. 0. 1. 3. 1. 2. 4. 7. 8. 3.]
     [0. 1. 2. 1. 2. 1. 3. 2. 2. 6.]
     [0. 1. 1. 3. 3. 2. 6. 2. 5. 9.]
     [0. 0. 2. 0. 4. 2. 2. 1. 6. 7.]]



```python
print(
    data[
        5:10, 
        0:10
    ]
)
```

    [[0. 0. 1. 2. 2. 4. 2. 1. 6. 4.]
     [0. 0. 2. 2. 4. 2. 2. 5. 5. 8.]
     [0. 0. 1. 2. 3. 1. 2. 3. 5. 3.]
     [0. 0. 0. 3. 1. 5. 6. 5. 5. 8.]
     [0. 1. 1. 2. 1. 3. 5. 3. 5. 8.]]



```python
small = data[
    :3, 
    36:
]
```


```python
print(
    'small is:'
)
```

    small is:



```python
print(
    small
)
```

    [[2. 3. 0. 0.]
     [1. 1. 0. 1.]
     [2. 2. 1. 1.]]



```python
print(
    numpy.mean(
        data
    )
)
```

    6.14875



```python
import time
```


```python
print(
    time.ctime()
)
```

    Wed May 10 19:45:12 2023



```python
maxval, minval, stdval = numpy.amax(
    data
), numpy.amin(
    data
), numpy.std(
    data
)
```


```python
print(
    'maximum inflammation:', 
    maxval
)
```

    maximum inflammation: 20.0



```python
print(
    'minimum inflammation:', 
    minval
)
```

    minimum inflammation: 0.0



```python
print(
    'standard deviation:', 
    stdval
)
```

    standard deviation: 4.613833197118566



```python
patient_0 = data[
    0, 
    :
]
```


```python
print(
    'maximum inflammation for patient 0:', 
    numpy.amax(
        patient_0
    )
)
```

    maximum inflammation for patient 0: 18.0



```python
print(
    'maximum inflammation for patient 2:', 
    numpy.amax(
        data[
            2, 
            :
        ]
    )
)
```

    maximum inflammation for patient 2: 19.0



```python
print(
    numpy.mean(
        data, 
        axis = 0
    )
)
```

    [ 0.          0.45        1.11666667  1.75        2.43333333  3.15
      3.8         3.88333333  5.23333333  5.51666667  5.95        5.9
      8.35        7.73333333  8.36666667  9.5         9.58333333 10.63333333
     11.56666667 12.35       13.25       11.96666667 11.03333333 10.16666667
     10.          8.66666667  9.15        7.25        7.33333333  6.58333333
      6.06666667  5.95        5.11666667  3.6         3.3         3.56666667
      2.48333333  1.5         1.13333333  0.56666667]



```python
print(
    numpy.mean(
        data, 
        axis = 0
    ).shape
)
```

    (40,)



```python
print(
    numpy.mean(
        data, 
        axis = 1
    )
)
```

    [5.45  5.425 6.1   5.9   5.55  6.225 5.975 6.65  6.625 6.525 6.775 5.8
     6.225 5.75  5.225 6.3   6.55  5.7   5.85  6.55  5.775 5.825 6.175 6.1
     5.8   6.425 6.05  6.025 6.175 6.55  6.175 6.35  6.725 6.125 7.075 5.725
     5.925 6.15  6.075 5.75  5.975 5.725 6.3   5.9   6.75  5.925 7.225 6.15
     5.95  6.275 5.7   6.1   6.825 5.975 6.725 5.7   6.25  6.4   7.05  5.9  ]



```python
element = 'oxygen'
```


```python
print(
    'first three characters:', 
    element[
        0:3
    ]
)
```

    first three characters: oxy



```python
print(
    'last three characters:', 
    element[
        3:6
    ]
)
```

    last three characters: gen



```python
import numpy
```


```python
A = numpy.array(
    [
        [
            1,
            2,
            3
        ], 
        [
            4,
            5,
            6
        ], 
        [
            7, 
            8, 
            9
        ]
    ]
)
```


```python
print(
    'A = '
)

print(
    A
)
```

    A = 
    [[1 2 3]
     [4 5 6]
     [7 8 9]]



```python
B = numpy.hstack(
    [
        A, 
        A
    ]
)
```


```python
print(
    'B = '
)

print(
    B
)
```

    B = 
    [[1 2 3 1 2 3]
     [4 5 6 4 5 6]
     [7 8 9 7 8 9]]



```python
C = numpy.vstack(
    [
        A, 
        A
    ]
)
```


```python
print(
    'C = '
)

print(
    C
)

```

    C = 
    [[1 2 3]
     [4 5 6]
     [7 8 9]
     [1 2 3]
     [4 5 6]
     [7 8 9]]



```python
patient3_week1 = data[
    3, 
    :7
]
```


```python
print(
    patient3_week1
)
```

    [0. 0. 2. 0. 4. 2. 2.]



```python
numpy.diff(
    patient3_week1
)
```




    array([ 0.,  2., -2.,  4., -2.,  0.])
    
## storing_values_in_lists
How can I store many values together?
```python
odds = [
    1, 
    3, 
    5, 
    7
]
```


```python
print(
    'odds are:', 
    odds
)
```

    odds are: [1, 3, 5, 7]



```python
print(
    'first element:', 
    odds[
        0
    ]
)
```

    first element: 1



```python
print(
    'last element:', 
    odds[
        3
    ]
)
```

    last element: 7



```python
print(
    '"-1" element:', 
    odds[
        -1
    ]
)
```

    "-1" element: 7



```python
names = ['Curie', 'Darwing', 'Turing']
```


```python
print(
    'names is originally:', 
    names
)
```

    names is originally: ['Curie', 'Darwing', 'Turing']



```python
names[
    1
] = 'Darwin'
```


```python
print(
    'final value of names:', 
    names
)
```

    final value of names: ['Curie', 'Darwin', 'Turing']



```python
name = 'Darwin'
```


```python
name[
    0
] = 'd'
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-150-44f62d478bea> in <module>
          1 name[
          2     0
    ----> 3 ] = 'd'
    

    TypeError: 'str' object does not support item assignment



```python
mild_salsa = [
    'peppers', 
    'onions', 
    'cilantro', 
    'tomatoes'
]
```


```python
hot_salsa = mild_salsa 
```


```python
hot_salsa[
    0
] = 'hot peppers'
```


```python
print(
    'Ingredients in mild salsa:', 
    mild_salsa
)
```

    Ingredients in mild salsa: ['hot peppers', 'onions', 'cilantro', 'tomatoes']



```python
print(
    'Ingredients in hot salsa:', 
    hot_salsa
)
```

    Ingredients in hot salsa: ['hot peppers', 'onions', 'cilantro', 'tomatoes']



```python
mild_salsa = [
    'peppers', 
    'onions', 
    'cilantro', 
    'tomatoes'
]
```


```python
hot_salsa = list(
    mild_salsa
)
```


```python
hot_salsa[
    0
] = 'hot peppers'
```


```python
print(
    'Ingredients in mild salsa:', 
    mild_salsa
)
```

    Ingredients in mild salsa: ['peppers', 'onions', 'cilantro', 'tomatoes']



```python
print(
    'Ingredients in hot salsa:', 
    hot_salsa
)
```

    Ingredients in hot salsa: ['hot peppers', 'onions', 'cilantro', 'tomatoes']



```python
veg = [
    [
        'lettuce', 
        'lettuce', 
        'peppers', 
        'zucchini'
    ],
     [
         'lettuce', 
         'lettuce', 
         'peppers', 
         'zucchini'
     ],
     [
         'lettuce', 
         'cilantro', 
         'peppers', 
         'zucchini'
     ]
]
```


```python
print(
    veg[
        2
    ]
)
```

    ['lettuce', 'cilantro', 'peppers', 'zucchini']



```python
print(
    veg[
        0
    ]
)
```

    ['lettuce', 'lettuce', 'peppers', 'zucchini']



```python
print(
    veg[
        0
    ][
        0
    ]
)
```

    lettuce



```python
print(
    veg[
        1
    ][
        2
    ]
)
```

    peppers



```python
sample_ages = [
    10, 
    12.5, 
    'Unknown'
]
```


```python
print(
    sample_ages
)
```

    [10, 12.5, 'Unknown']



```python
odds.append(
    11
)
```


```python
print(
    'odds after adding a value:', 
    odds
)
```

    odds after adding a value: [1, 3, 5, 7, 11]



```python
removed_element = odds.pop(
    0
)
```


```python
print(
    'odds after removing the first element:', 
    odds
)
```

    odds after removing the first element: [3, 5, 7, 11]



```python
print(
    'removed_element:', 
    removed_element
)
```

    removed_element: 1



```python
odds.reverse()
```


```python
print(
    'odds after reversing:', 
    odds
)
```

    odds after reversing: [11, 7, 5, 3]



```python
odds = [
    3, 
    5, 
    7
]
```


```python
primes = odds
```


```python
primes.append(
    2
)
```


```python
print(
    'primes:', 
    primes
)
```

    primes: [3, 5, 7, 2]



```python
print(
    'odds:', 
    odds
)
```

    odds: [3, 5, 7, 2]



```python
binomial_name = 'Drosophila melanogaster'
```


```python
group = binomial_name[
    0:10
]
```


```python
print(
    'group:', 
    group
)
```

    group: Drosophila



```python
species = binomial_name[
    11:23
]
```


```python
print(
    'species:', 
    species
)
```

    species: melanogaster



```python
chromosomes = [
    'X', 
    'Y', 
    '2', 
    '3', 
    '4'
]
```


```python
autosomes = chromosomes[
    2:5
]
```


```python
print(
    'autosomes:', 
    autosomes
)
```

    autosomes: ['2', '3', '4']



```python
last = chromosomes[
    -1
]
```


```python
print(
    'last:', 
    last
)
```

    last: 4



```python
string_for_slicing = 'Observation date: 02-Feb-2013'
```


```python
list_for_slicing = [
    [
        'fluorine', 
        'F'
    ],
    [
        'chlorine', 
        'Cl'
    ],
    [
        'bromine', 
        'Br'
    ],
    [
        'iodine', 
        'I'
    ],
    [
        'astatine',
        'At'
    ]
]
```


```python
print(
    string_for_slicing
)
```

    Observation date: 02-Feb-2013



```python
print(
    string_for_slicing
)
```

    Observation date: 02-Feb-2013



```python
primes = [
    2, 
    3, 
    5, 
    7, 
    11, 
    13, 
    17, 
    19, 
    23, 
    29, 
    31, 
    37
]
```


```python
subset = primes[
    0:12:3
]
```


```python
print(
    'subset', 
    subset
)
```

    subset [2, 7, 17, 29]



```python
beatles = "In an octopus's garden in the shade"
```


```python
print(
    beatles
)
```

    In an octopus's garden in the shade



```python
date = 'Monday 4 January 2016'
```


```python
day = date[
    0:6
]
```


```python
print(
    'Using 0 to begin range:', 
    day
)
```

    Using 0 to begin range: Monday



```python
day = date[
    :6
]
```


```python
print(
    'Omitting beginning index:', 
    day
)
```

    Omitting beginning index: Monday



```python
months = [
    'jan', 
    'feb', 
    'mar', 
    'apr', 
    'may', 
    'jun', 
    'jul', 
    'aug', 
    'sep', 
    'oct', 
    'nov', 
    'dec'
]
```


```python
sond = months[
    8:12
]
```


```python
sond = months[
    8:12
]
```


```python
sond = months[
    8:len(
        months
    )
]
```


```python
print(
    'Using len() to get last entry:', 
    sond
)
```

    Using len() to get last entry: ['sep', 'oct', 'nov', 'dec']



```python
sond = months[
    8:
]
```


```python
print(
    'Omitting ending index:', 
    sond
)
```

    Omitting ending index: ['sep', 'oct', 'nov', 'dec']



```python
counts = [
    2, 
    4, 
    6, 
    8, 
    10
]
```


```python
repeats = counts * 2
```


```python
print(
    repeats
)
```

    [2, 4, 6, 8, 10, 2, 4, 6, 8, 10]
    
 ## using_loops
 How can I do the same operations on many different values?
 ```python
odds = [
    1, 
    3, 
    5, 
    7
]
```


```python
print(
    odds[
        0
    ]
)
```

    1



```python
print(
    odds[
        1
    ]
)
```

    3



```python
print(
    odds[
        2
    ]
)
```

    5



```python
print(
    odds[
        3
    ]
)
```

    7



```python
odds = [
    1, 
    3, 
    5
]
```


```python
print(
    odds[
        0
    ]
)
```

    1



```python
print(
    odds[
        1
    ]
)
```

    3



```python
print(
    odds[
        2
    ]
)
```

    5



```python
print(
    odds[
        3
    ]
)
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-10-3392a80ec81a> in <module>
          1 print(
          2     odds[
    ----> 3         3
          4     ]
          5 )


    IndexError: list index out of range



```python
odds = [
    1, 
    3, 
    5, 
    7
]
```


```python
for num in odds:
    print(
        num
    )
```

    1
    3
    5
    7



```python
odds = [
    1, 
    3, 
    5, 
    7,
    9,
    11
]
```


```python
for num in odds:
    print(
        num
    )
```

    1
    3
    5
    7
    9
    11



```python
odds = [
    1,
    3,
    5,
    7,
    9,
    11
]
```


```python
for banana in odds:
    print(
        banana
    )
```

    1
    3
    5
    7
    9
    11



```python
length = 0
```


```python
names = [
    'Curie',
    'Darwin',
    'Turing'
]
```


```python
for value in names:
    length = length + 1
```


```python
print(
    'There are',
    length,
    'names in the list.'
)
```

    There are 3 names in the list.



```python
name = 'Rosalind'
```


```python
for name in [
    'Curie', 
    'Darwin', 
    'Turing'
]:
    print(
        name
    )
```

    Curie
    Darwin
    Turing



```python
print(
    'after the loop, name is', 
    name
)
```

    after the loop, name is Turing



```python
print(
    len(
        [
            0, 
            1, 
            2, 
            3
        ]
    )
)
```

    4



```python
word = 'oxygen'
```


```python
for letter in word:
    print(
        letter
    )
```

    o
    x
    y
    g
    e
    n



```python
print(
    5 ** 3
)
```

    125



```python
x = 5
```


```python
coefs = [
    2, 
    4,
    3
]
```


```python
y = coefs[
    0
] * x**0 + coefs[
    1
] * x**1 + coefs[
    2
] * x**2
```


```python
print(
    y
)
```

    97
    
## using_multiple_files
How can I do the same operations on many different files?
```python
import glob
```


```python
print(
    glob.glob(
        'inflammation*.csv'
    )
)
```

    ['inflammation-05.csv', 'inflammation-12.csv', 'inflammation-04.csv', 'inflammation-08.csv', 'inflammation-10.csv', 'inflammation-06.csv', 'inflammation-09.csv', 'inflammation-01.csv', 'inflammation-07.csv', 'inflammation-11.csv', 'inflammation-03.csv', 'inflammation-02.csv']



```python
import glob
```


```python
import numpy
```


```python
import matplotlib.pyplot
```


```python
filenames = sorted(
    glob.glob(
        'inflammation*.csv'
    )
)
```


```python
filenames = filenames[
    0:3
]
```


```python
for filename in filenames:
    print(
        filename
    )

    data = numpy.loadtxt(
        fname = filename, 
        delimiter = ','
    )

    fig = matplotlib.pyplot.figure(
        figsize = (
            10.0,
            3.0
        )
    )

    axes1 = fig.add_subplot(
        1,
        3,
        1
    )
    
    axes2 = fig.add_subplot(
        1,
        3,
        2
    )
    
    axes3 = fig.add_subplot(
        1,
        3,
        3
    )

    axes1.set_ylabel(
        'average'
    )
    
    axes1.plot(
        numpy.mean(
            data,
            axis = 0
        )
    )

    axes2.set_ylabel(
        'max'
    )
    
    axes2.plot(
        numpy.amax(
            data,
            axis = 0
        )
    )

    axes3.set_ylabel(
        'min'
    )
    
    axes3.plot(
        numpy.amin(
            data,
            axis = 0
        )
    )

    fig.tight_layout()
    
    matplotlib.pyplot.show()
```

    inflammation-01.csv



![output_7_1](https://github.com/dalemajorjr/python_portfolio/assets/132947457/007fe4f6-cd8f-41bb-b298-cdcc2da55408)



    inflammation-02.csv



![output_7_3](https://github.com/dalemajorjr/python_portfolio/assets/132947457/1dc98c52-e4be-4013-ab58-95637a616ab3)



    inflammation-03.csv



![output_7_5](https://github.com/dalemajorjr/python_portfolio/assets/132947457/0b072290-316d-4465-aa0b-f2da6003cc36)




```python
import glob
```


```python
import numpy
```


```python
import matplotlib.pyplot
```


```python
filenames = glob.glob(
    'inflammation*.csv'
)
```


```python
composite_data = numpy.zeros(
    (
        60,
        40
    )
)
```


```python
for filename in filenames:
    data = numpy.loadtxt(
        fname = filename, delimiter=','
    )
```


```python
composite_data = composite_data + data
```


```python
composite_data = composite_data / len(
    filenames
)
```


```python
fig = matplotlib.pyplot.figure(
    figsize = (
        10.0,
        3.0
    )
)

axes1 = fig.add_subplot(
    1,
    3,
    1
)

axes2 = fig.add_subplot(
    1,
    3,
    2
)

axes3 = fig.add_subplot(
    1,
    3,
    3
)

axes1.set_ylabel(
    'average'
)

axes1.plot(
    numpy.mean(
        composite_data,
        axis = 0
    )
)

axes2.set_ylabel(
    'max'
)

axes2.plot(
    numpy.amax(
        composite_data, 
        axis = 0
    )
)

axes3.set_ylabel(
    'min'
)

axes3.plot(
    numpy.amin(
        composite_data, 
        axis=0
    )
)

fig.tight_layout()

matplotlib.pyplot.show()
```


![output_16_0](https://github.com/dalemajorjr/python_portfolio/assets/132947457/c3d8f0d7-c10a-4e7b-835e-6dcc5cf6dec6)

## making_chocies
How can my programs do different things based on data values?
```python
num = 37
```


```python
if num > 100:
    print(
        'greater'
    )
    
else:
    print(
        'not greater'
    )
```

    not greater



```python
print(
    'done'
)
```

    done



```python
num = 53
```


```python
print(
    'before conditional...'
)
```

    before conditional...



```python
if num > 100:
    print(
        num,
        'is greater than 100'
    )
```


```python
print(
    '...after conditional'
)
```

    ...after conditional



```python
num = -3
```


```python
if num > 0:
    print(
        num, 
        'is positive'
    )
    
elif num == 0:
    print(
        num, 
        'is zero'
    )
    
else:
    print(
        num, 
        'is negative'
    )
```

    -3 is negative



```python
if (
    1 > 0
) and (
    -1 >= 0
):
    print(
        'both parts are true'
    )
    
else:
    print(
        'at least one part is false'
    )
```

    at least one part is false



```python
if (
    1 < 0
) or (
    1 >= 0
):
    print(
        'at least one test is true'
    )
```

    at least one test is true

```python
import numpy
```


```python
data = numpy.loadtxt(
    fname = 'inflammation-01.csv', 
    delimiter = ','
)
```


```python
max_inflammation_0 = numpy.amax(
    data, 
    axis = 0
)[
    0
]
```


```python
max_inflammation_20 = numpy.amax(
    data,
    axis = 0
)[
    20
]
```


```python
if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print(
        'Suspicious looking maxima!'
    )
    
elif numpy.sum(
    numpy.amin(
        data,
        axis = 0
    )
) == 0:
    print(
        'Minima add up to zero!'
    )
    
else:
    print(
        'Seems OK!'
    )
```

    Suspicious looking maxima!



```python
data = numpy.loadtxt(
    fname = 'inflammation-03.csv',
    delimiter = ','
)
```


```python
max_inflammation_0 = numpy.amax(
    data,
    axis = 0
)[
    0
]
```


```python
max_inflammation_20 = numpy.amax(
    data,
    axis = 0
)[
    20
]
```


```python
if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print(
        'Suspicious looking maxima!'
    )
    
elif numpy.sum(
    numpy.amin(
        data,
        axis = 0
    )
) == 0:
    print(
        'Minima add up to zero!'
    )
    
else:
    print(
        'Seems OK!'
    )
```

    Minima add up to zero!



```python
if 4 > 5:
    print(
        'A'
    )
    
elif 4 == 5:
    print(
        'B'
    )
    
elif 4 < 5:
    print(
        'C'
    )
```

    C



```python
if '':
    print(
        'empty string is true'
    )
    
if 'word':
    print(
        'word is true'
    )
    
if []:
    print(
        'empty list is true'
    )
    
if [
    1,
    2,
    3
]:
    print(
        'non-empty list is true'
    )
    
if 0:
    print(
        'zero is true'
    )
    
if 1:
    print(
        'one is true'
    )
```

    word is true
    non-empty list is true
    one is true



```python
if not '':
    print(
        'empty string is not true'
    )
    
if not 'word':
    print(
        'word is not true'
    )
    
if not not True:
    print(
        'not not True is true'
    )
```

    empty string is not true
    not not True is true



```python
x = 1 
```


```python
x += 1 
```


```python
x *= 3 
```


```python
print(
    x
)
```

    6



```python
'String'.startswith(
    'Str'
)
```




    True




```python
'String'.startswith(
    'str'
)
```




    False




```python
filenames = [
    'inflammation-01.csv',
    'myscript.py',
    'inflammation-02.csv',
    'small-01.csv',
    'small-02.csv'
]
```


```python
large_files = []
```


```python
small_files = []
```


```python
other_files = []
```


```python
for filename in filenames:
    if filename.startswith(
        'inflammation-'
    ):
        large_files.append(
            filename
        )
    elif filename.startswith(
        'small-'
    ):
        small_files.append(
            filename
        )
    else:
        other_files.append(
            filename
        )
```


```python
print(
    'large_files:',
    large_files
)
```

    large_files: ['inflammation-01.csv', 'inflammation-02.csv']



```python
print(
    'small_files:',
    small_files
)
```

    small_files: ['small-01.csv', 'small-02.csv']



```python
print(
    'other_files:',
    other_files
)
```

    other_files: ['myscript.py']
    
## functions
How can I define new functions?
What’s the difference between defining and calling a function?
What happens when I call a function?
```python
fahrenheit_val = 99
```


```python
celsius_val = (
    (
        fahrenheit_val - 32
    ) * (
        5/9
    )
)
```


```python
print(
    celsius_val
)
```

    37.22222222222222



```python
fahrenheit_val2 = 43
```


```python
celsius_val2 = (
    (
        fahrenheit_val2 - 32
    ) * (
        5/9
    )
)
```


```python
print(
    celsius_val2
)
```

    6.111111111111112



```python
def explicit_fahr_to_celsius(
    temp
):
    converted = (
        (
            temp - 32
        ) * (
            5/9
        )
    )
    return converted
```


```python
def fahr_to_celsius(
    temp
):
    return(
        (
            temp - 32
        ) * (
            5/9
        )
    )
```


```python
fahr_to_celsius(
    32
)
```




    0.0




```python
print(
    'freezing point of water:', 
    fahr_to_celsius(
        32
    ),
    'C'
)
```

    freezing point of water: 0.0 C



```python
print(
    'boiling point of water:', 
    fahr_to_celsius(
        212
    ),
    'C'
)
```

    boiling point of water: 100.0 C



```python
def celsius_to_kelvin(
    temp_c
):
    return temp_c + 273.15
```


```python
print(
    'freezing point of water in Kelvin:',
    celsius_to_kelvin(
        0.
    )
)
```

    freezing point of water in Kelvin: 273.15



```python
def fahr_to_kelvin(
    temp_f
):
    temp_c = fahr_to_celsius(
        temp_f
    )
    temp_k = celsius_to_kelvin(
        temp_c
    )
    return temp_k
```


```python
print(
    'boiling point of water in Kelvin:', 
    fahr_to_kelvin(
        212.0
    )
)
```

    boiling point of water in Kelvin: 373.15



```python
print(
    'Again, temperature in Kelvin was:',
    temp_k
)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-16-d41fa953a2d5> in <module>
          1 print(
          2     'Again, temperature in Kelvin was:',
    ----> 3     temp_k
          4 )


    NameError: name 'temp_k' is not defined



```python
temp_kelvin = fahr_to_kelvin(
    212.0
)
```


```python
print(
    'temperature in Kelvin was:',
    temp_kelvin
)
```

    temperature in Kelvin was: 373.15



```python
def print_temperatures():
    print(
        'temperature in Fahrenheit was:', 
        temp_fahr
    )
    print(
        'temperature in Kelvin was:',
        temp_kelvin
    )
```


```python
temp_fahr = 212.0
```


```python
temp_kelvin = fahr_to_kelvin(
    temp_fahr
)
```


```python
print_temperatures()
```

    temperature in Fahrenheit was: 212.0
    temperature in Kelvin was: 373.15

```python
import numpy
```


```python
import glob
```


```python
import matplotlib
```


```python
import matplotlib.pyplot
```


```python
def visualize(filename):
    data = numpy.loadtxt(
        fname = filename, 
        delimiter = ','
    )
    fig = matplotlib.pyplot.figure(
        figsize = (
            10.0,
            3.0
        )
    )
    axes1 = fig.add_subplot(
        1,
        3,
        1
    )
    axes2 = fig.add_subplot(
        1,
        3,
        2
    )
    axes3 = fig.add_subplot(
        1,
        3,
        3
    )
    axes1.set_ylabel(
        'average'
    )
    axes1.plot(
        numpy.mean(
            data,
            axis = 0
        )
    )
    axes2.set_ylabel(
        'max'
    )
    axes2.plot(
        numpy.amax(
            data,
            axis = 0
        )
    )
    axes3.set_ylabel(
        'min'
    )
    axes3.plot(
        numpy.amin(
            data,
            axis = 0
        )
    )
    fig.tight_layout()
    matplotlib.pyplot.show()
```


```python
def detect_problems(filename):
    data = numpy.loadtxt(
        fname = filename,
        delimiter = ','
    )
    if numpy.amax(
        data,
        axis = 0
    )[
        0
    ] == 0 and numpy.amax(
        data,
        axis = 0)[
        20
    ] == 20:
        print(
            'Suspicious looking maxima!'
        )        
    elif numpy.sum(
        numpy.amin(
            data,
            axis = 0
        )
    ) == 0:
        print(
            'Minima add up to zero!'
        )        
    else:
        print(
            'Seems OK!'
        )
```


```python
filenames = sorted(
    glob.glob(
        'inflammation*.csv'
    )
)
```


```python
for filename in filenames:
    print(
        filename
    )
    visualize(
        filename
    )
    detect_problems(
        filename
    )
```

    inflammation-01.csv



![output_7_1](https://github.com/dalemajorjr/python_portfolio/assets/132947457/30017b96-5ade-4015-88a3-ab099d94764f)



    Suspicious looking maxima!
    inflammation-02.csv



![output_7_3](https://github.com/dalemajorjr/python_portfolio/assets/132947457/a7cb95d8-b4c7-4f4c-bbcf-166374bae53d)



    Suspicious looking maxima!
    inflammation-03.csv



![output_7_5](https://github.com/dalemajorjr/python_portfolio/assets/132947457/a741d73f-96b6-4c69-81c9-7bcc041009f3)



    Minima add up to zero!
    inflammation-04.csv



![output_7_7](https://github.com/dalemajorjr/python_portfolio/assets/132947457/e24ced56-b7a6-483c-80b8-c95c5359d222)



    Suspicious looking maxima!
    inflammation-05.csv



![output_7_9](https://github.com/dalemajorjr/python_portfolio/assets/132947457/581f0808-da43-4bda-aee3-985afaffd363)



    Suspicious looking maxima!
    inflammation-06.csv



![output_7_11](https://github.com/dalemajorjr/python_portfolio/assets/132947457/e230489e-98e9-413d-8197-c2fb6926da41)



    Suspicious looking maxima!
    inflammation-07.csv



![output_7_13](https://github.com/dalemajorjr/python_portfolio/assets/132947457/691e0e47-8a60-4a72-8b86-088589ae2336)



    Suspicious looking maxima!
    inflammation-08.csv



![output_7_15](https://github.com/dalemajorjr/python_portfolio/assets/132947457/14529946-e016-4407-a7cb-f4f735d2c553)



    Minima add up to zero!
    inflammation-09.csv



![output_7_17](https://github.com/dalemajorjr/python_portfolio/assets/132947457/5de7496c-0618-432d-ac4b-d4847f7834e2)



    Suspicious looking maxima!
    inflammation-10.csv



![output_7_19](https://github.com/dalemajorjr/python_portfolio/assets/132947457/488bb251-30a9-448b-bfcc-e34db6b49913)



    Suspicious looking maxima!
    inflammation-11.csv



![output_7_21](https://github.com/dalemajorjr/python_portfolio/assets/132947457/74bd8e6e-bac5-4789-92af-0f37965ae646)



    Minima add up to zero!
    inflammation-12.csv



![output_7_23](https://github.com/dalemajorjr/python_portfolio/assets/132947457/aed4b99a-8892-4cb1-835d-ca34e0c29bb3)



    Suspicious looking maxima!



```python
def offset_mean(
    data, 
    target_mean_value
):
    return(
        data - numpy.mean(
            data
        )
    ) + target_mean_value
```


```python
z = numpy.zeros(
    (
        2,
        2
    )
)
```


```python
print(
    offset_mean(
        z,
        3
    )
)
```

    [[3. 3.]
     [3. 3.]]



```python
data = numpy.loadtxt(
    fname = 'inflammation-01.csv', 
    delimiter = ','
)
```


```python
print(
    offset_mean(
        data,
        0
    )
)
```

    [[-6.14875 -6.14875 -5.14875 ... -3.14875 -6.14875 -6.14875]
     [-6.14875 -5.14875 -4.14875 ... -5.14875 -6.14875 -5.14875]
     [-6.14875 -5.14875 -5.14875 ... -4.14875 -5.14875 -5.14875]
     ...
     [-6.14875 -5.14875 -5.14875 ... -5.14875 -5.14875 -5.14875]
     [-6.14875 -6.14875 -6.14875 ... -6.14875 -4.14875 -6.14875]
     [-6.14875 -6.14875 -5.14875 ... -5.14875 -5.14875 -6.14875]]



```python
print(
    'original min, mean, and max are:', 
    numpy.amin(
        data
    ),
    numpy.mean(
        data
    ),
    numpy.amax(
        data
    )
)
```

    original min, mean, and max are: 0.0 6.14875 20.0



```python
offset_data = offset_mean(
    data,
    0
)
```


```python
print(
    'min, mean, and max of offset data are:',
    numpy.amin(
        offset_data
    ),
    numpy.mean(
        offset_data
    ),
    numpy.amax(
        offset_data
    )
)
```

    min, mean, and max of offset data are: -6.14875 2.842170943040401e-16 13.85125



```python
print(
    'std dev before and after:', 
    numpy.std(
        data
    ), 
    numpy.std(
        offset_data
    )
)
```

    std dev before and after: 4.613833197118566 4.613833197118566



```python
print(
    'difference in standard deviations before and after:',
    numpy.std(
        data
    ) - numpy.std(
        offset_data
    )
)
```

    difference in standard deviations before and after: 0.0



```python
def offset_mean(
    data, 
    target_mean_value
):
    return(
        data - numpy.mean(
            data
        )
    ) + target_mean_value
```


```python
def offset_mean(
    data, target_mean_value
):
    """Return a new array containing the original data
       with its mean offset to match the desired value."""
    return(
        data - numpy.mean(
            data
        )
    ) + target_mean_value
```


```python
help(
    offset_mean
)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array containing the original data
        with its mean offset to match the desired value.
    



```python
def offset_mean(
    data, target_mean_value
):
    """Return a new array containing the original data
       with its mean offset to match the desired value.

    Examples
    --------
    >>> offset_mean([1, 2, 3], 0)
    array([-1.,  0.,  1.])
    """
    return(
        data - numpy.mean(
            data
        )
    ) + target_mean_value
```


```python
help(
    offset_mean
)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array containing the original data
           with its mean offset to match the desired value.
        
        Examples
        --------
        >>> offset_mean([1, 2, 3], 0)
        array([-1.,  0.,  1.])
    



```python
numpy.loadtxt(
    'inflammation-01.csv',
    delimiter = ','
)
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
numpy.loadtxt(
    'inflammation-01.csv',
    ','
)
```


    Traceback (most recent call last):


      File "/home/student/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 3326, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)


      File "<ipython-input-25-2984fb556f6c>", line 3, in <module>
        ','


      File "/home/student/anaconda3/lib/python3.7/site-packages/numpy/lib/npyio.py", line 1087, in loadtxt
        dtype = np.dtype(dtype)


      File "/home/student/anaconda3/lib/python3.7/site-packages/numpy/core/_internal.py", line 201, in _commastring
        newitem = (dtype, eval(repeats))


      File "<string>", line 1
        ,
        ^
    SyntaxError: unexpected EOF while parsing




```python
def offset_mean(
    data,
    target_mean_value = 0.0
):
    """Return a new array containing the original data
       with its mean offset to match the desired value, (0 by default).

    Examples
    --------
    >>> offset_mean([1, 2, 3])
    array([-1.,  0.,  1.])
    """
    return(
        data - numpy.mean(
            data
        )
    ) + target_mean_value
```


```python
test_data = numpy.zeros(
    (
        2,
        2
    )
)
```


```python
print(
    offset_mean(
        test_data,
        3
    )
)
```

    [[3. 3.]
     [3. 3.]]



```python
more_data = 5 + numpy.zeros(
    (
        2,
        2
    )
)
```


```python
print(
    'data before mean offset:'
)
```

    data before mean offset:



```python
print(
    more_data
)
```

    [[5. 5.]
     [5. 5.]]



```python
print(
    'offset data:'
)
```

    offset data:



```python
print(
    offset_mean(
        more_data
    )
)
```

    [[0. 0.]
     [0. 0.]]



```python
def display(
    a = 1,
    b = 2,
    c = 3
):
    print(
        'a:',
        a,
        'b:',
        b,
        'c:',
        c
    )
```


```python
print(
    'no parameters:'
)
```

    no parameters:



```python
display()
```

    a: 1 b: 2 c: 3



```python
print(
    'one parameter:'
)
```

    one parameter:



```python
display(
    55
)
```

    a: 55 b: 2 c: 3



```python
print(
    'two parameters:'
)
```

    two parameters:



```python
display(
    55,
    66
)
```

    a: 55 b: 66 c: 3



```python
print(
    'only setting the value of c'
)
```

    only setting the value of c



```python
display(
    c = 77
)
```

    a: 1 b: 2 c: 77



```python
help(
    numpy.loadtxt
)
```

    Help on function loadtxt in module numpy:
    
    loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None)
        Load data from a text file.
        
        Each row in the text file must have the same number of values.
        
        Parameters
        ----------
        fname : file, str, or pathlib.Path
            File, filename, or generator to read.  If the filename extension is
            ``.gz`` or ``.bz2``, the file is first decompressed. Note that
            generators should return byte strings for Python 3k.
        dtype : data-type, optional
            Data-type of the resulting array; default: float.  If this is a
            structured data-type, the resulting array will be 1-dimensional, and
            each row will be interpreted as an element of the array.  In this
            case, the number of columns used must match the number of fields in
            the data-type.
        comments : str or sequence of str, optional
            The characters or list of characters used to indicate the start of a
            comment. None implies no comments. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is '#'.
        delimiter : str, optional
            The string used to separate values. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is whitespace.
        converters : dict, optional
            A dictionary mapping column number to a function that will parse the
            column string into the desired value.  E.g., if column 0 is a date
            string: ``converters = {0: datestr2num}``.  Converters can also be
            used to provide a default value for missing data (but see also
            `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
            Default: None.
        skiprows : int, optional
            Skip the first `skiprows` lines, including comments; default: 0.
        usecols : int or sequence, optional
            Which columns to read, with 0 being the first. For example,
            ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
            The default, None, results in all columns being read.
        
            .. versionchanged:: 1.11.0
                When a single column has to be read it is possible to use
                an integer instead of a tuple. E.g ``usecols = 3`` reads the
                fourth column the same way as ``usecols = (3,)`` would.
        unpack : bool, optional
            If True, the returned array is transposed, so that arguments may be
            unpacked using ``x, y, z = loadtxt(...)``.  When used with a structured
            data-type, arrays are returned for each field.  Default is False.
        ndmin : int, optional
            The returned array will have at least `ndmin` dimensions.
            Otherwise mono-dimensional axes will be squeezed.
            Legal values: 0 (default), 1 or 2.
        
            .. versionadded:: 1.6.0
        encoding : str, optional
            Encoding used to decode the inputfile. Does not apply to input streams.
            The special value 'bytes' enables backward compatibility workarounds
            that ensures you receive byte arrays as results if possible and passes
            'latin1' encoded strings to converters. Override this value to receive
            unicode arrays and pass strings as input to converters.  If set to None
            the system default is used. The default value is 'bytes'.
        
            .. versionadded:: 1.14.0
        max_rows : int, optional
            Read `max_rows` lines of content after `skiprows` lines. The default
            is to read all the lines.
        
            .. versionadded:: 1.16.0
        
        Returns
        -------
        out : ndarray
            Data read from the text file.
        
        See Also
        --------
        load, fromstring, fromregex
        genfromtxt : Load data with missing values handled as specified.
        scipy.io.loadmat : reads MATLAB data files
        
        Notes
        -----
        This function aims to be a fast reader for simply formatted files.  The
        `genfromtxt` function provides more sophisticated handling of, e.g.,
        lines with missing values.
        
        .. versionadded:: 1.10.0
        
        The strings produced by the Python float.hex method can be used as
        input for floats.
        
        Examples
        --------
        >>> from io import StringIO   # StringIO behaves like a file object
        >>> c = StringIO(u"0 1\n2 3")
        >>> np.loadtxt(c)
        array([[0., 1.],
               [2., 3.]])
        
        >>> d = StringIO(u"M 21 72\nF 35 58")
        >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
        ...                      'formats': ('S1', 'i4', 'f4')})
        array([(b'M', 21, 72.), (b'F', 35, 58.)],
              dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])
        
        >>> c = StringIO(u"1,0,2\n3,0,4")
        >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
        >>> x
        array([1., 3.])
        >>> y
        array([2., 4.])
    



```python
numpy.loadtxt(
    'inflammation-01.csv',
    delimiter = ','
)
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
def s(
    p
):
    a = 0
    for v in p:
        a += v
    m = a/len(
        p
    )
    d = 0
    for v in p:
        d += (
            v - m
        ) * (
            v - m
        )
    return numpy.sqrt(
        d/(
            len(
                p
            ) - 1
        )
    )
```


```python
def std_dev(
    sample
):
    sample_sum = 0
    for value in sample:
        sample_sum += value
    sample_mean = sample_sum/len(
        sample
    )
    sum_squared_devs = 0
    for value in sample:
        sum_squared_devs += (
            value - sample_mean
        ) * (
            value - sample_mean
        )
    return numpy.sqrt(
        sum_squared_devs/(
            len(
                sample
            ) - 1
        )
    )
```


```python
def fence(
    original,
    wrapper
):
    return wrapper + original + wrapper
```


```python
def add(
    a,
    b
):
    print(
        a + b
    )
```


```python
A = add(
    7,
    3
)
```

    10



```python
print(
    A
)
```

    None



```python
print(
    fence(
        'name',
        '*'
    )
)
```

    *name*



```python
def add(
    a,
    b
):
    print(
        a + b
    )
```


```python
def outer(
    input_string
):
    return input_string[
        0
    ] + input_string[
        -1
    ]
```


```python
print(
    outer(
        'helium'
    )
)
```

    hm



```python
def rescale(
    input_array
):
    L = numpy.amin(
        input_array
    )
    H = numpy.amax(
        input_array
    )
    output_array = (
        input_array - L
    )/(
        H - L
    )
    return output_array
```

## defensive_porograming
How can I make my programs more reliable?
```python
numbers = [
    1.5,
    2.3,
    0.7,
    0.001,
    4.4
]
```


```python
total = 0.0
```


```python
for num in numbers:
    assert num > 0.0, 
    'Data should only contain positive values'
    total += num
```


      File "<ipython-input-4-3f92537c24d6>", line 2
        assert num > 0.0,
                          ^
    SyntaxError: invalid syntax




```python
print(
    'total is:', 
    total
)
```

    total is: 0.0



```python
def normalize_rectangle(
    rect
):
    """Normalizes a rectangle so that it is at the origin and 1.0 units long on its longest axis.
    Input should be of the format (x0, y0, x1, y1).
    (x0, y0) and (x1, y1) define the lower left and upper right corners
    of the rectangle, respectively."""
    assert len(
        rect
    ) == 4, 
    'Rectangles must contain 4 coordinates'
    x0, 
    y0, 
    x1, 
    y1 = rect
    assert x0 < x1,
    'Invalid X coordinates'
    assert y0 < y1, 
    'Invalid Y coordinates'
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy:
        scaled = dx/dy
        upper_x, 
        upper_y = 1.0,
        scaled
    else:
        scaled = dx/dy
        upper_x, 
        upper_y = scaled, 
        1.0
    assert 0 < upper_x <= 1.0, 
    'Calculated upper X coordinate invalid'
    assert 0 < upper_y <= 1.0, 
    'Calculated upper Y coordinate invalid'
```


      File "<ipython-input-6-8ac046e3dcd2>", line 10
        ) == 4,
                ^
    SyntaxError: invalid syntax




```python
print(
    normalize_rectangle(
        (
            0.0,
            1.0,
            2.0
        )
    )
)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-7-5d94b06c1e15> in <module>
          1 print(
    ----> 2     normalize_rectangle(
          3         (
          4             0.0,
          5             1.0,


    NameError: name 'normalize_rectangle' is not defined



```python
print(
    normalize_rectangle(
        (
            0.0,
            0.0,
            1.0,
            5.0
        )
    )
)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-8-7af076429a90> in <module>
          1 print(
    ----> 2     normalize_rectangle(
          3         (
          4             0.0,
          5             0.0,


    NameError: name 'normalize_rectangle' is not defined



```python
print(
    normalize_rectangle(
        (
            0.0,
            0.0,
            5.0,
            1.0
        )
    )
)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-9-926e451f0aa7> in <module>
          1 print(
    ----> 2     normalize_rectangle(
          3         (
          4             0.0,
          5             0.0,


    NameError: name 'normalize_rectangle' is not defined

