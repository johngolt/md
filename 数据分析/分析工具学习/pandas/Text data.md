##### Working with text data

Pandas includes features to address both this need for vectorized string operations and for correctly handling missing data via the ``str`` attribute of Pandas Series and Index objects containing strings.

Series and Index are equipped with a set of string processing methods that make it easy to operate on each element of the array. Perhaps most importantly, these methods exclude missing/NA values automatically. These are accessed via the `str` attribute and generally have names matching the equivalent built-in string methods.

If you do want literal replacement of a string, you can set the optional `regex` parameter to `False`, rather than escaping each character. In this case both `pat` and `repl` must be strings.

 ```python
dollars = pd.Series(['12', '-$10', '$10,000'])
dollars = pd.Series(['12', '-$10', '$10,000'])
dollars.str.replace('-$', '-', regex=False)
 ```

The `replace` method can also take a callable as replacement. It is called on every `pat` using `re.sub()`. The callable should expect one positional argument and return a string. 

```python
pat = r'[a-z]+'
def repl(m):
     return m.group(0)[::-1]
pd.Series(['foo 123', 'bar baz', np.nan]).str.replace(pat, repl)
pat = r"(?P<one>\w+) (?P<two>\w+) (?P<three>\w+)"
def repl(m):
     return m.group('two').swapcase()
pd.Series(['Foo Bar Baz', np.nan]).str.replace(pat, repl)
```

The `replace` method also accepts a compiled regular expression object from `re.compile()` as a pattern. All flags should be included in the compiled regular expression object.

```python
regex_pat = re.compile(r'^.a|dog', flags=re.IGNORECASE)
s3.str.replace(regex_pat, 'XX-XX ')
```

There are several ways to concatenate a Series or Index, either with itself or others, all based on `cat()`

```python
pd.Series(['a', 'b', 'c', 'd']).str.cat(sep=',')
s.str.cat(['A', 'B', 'C', 'D'])
v = pd.Series(['z', 'a', 'b', 'd', 'e'], index=[-1, 0, 1, 3, 4])
s.str.cat(v, join='left', na_rep='-')
```

By default, missing values are ignored. Using `na_rep`, they can be given a representation. The first argument to `cat()` can be a list-like object, provided that it matches the length of the calling Series. Missing values on either side will result in missing values in the result as well, unless `na_rep` is specified.
The parameter `others` can also be two-dimensional. In this case, the number or rows must match the lengths of the calling Series. For concatenation with a Series or DataFrame, it is possible to align the indexes before concatenation by setting the `join` keyword. The usual options are available for join (one of 'left', 'outer', 'inner', 'right'). In particular, alignment also means that the different lengths do not need to coincide anymore.
The same alignment can be used when `others` is a DataFrame. Several array-like items can be combined in a list-like container. All elements without an index within the passed list-like must match in length to the calling Series, but Series and Index may have arbitrary length (as long as alignment is not disabled with join=None). If using join='right' on a list-like of `others` that contains different indexes, the union of these indexes will be used as the basis for the final concatenation. 

You can use `[]` notation to directly index by position locations. If you index past the end of the string, the result will be a `NaN`.

###### Extracting substrings

The `extract` method accepts a regular expression with at least one capture group. Extracting a regular expression with more than one group returns a DataFrame with one column per group.

```python
pd.Series(['a1', 'b2', 'c3']).str.extract(r'([ab])(\d)', expand=False)
```

Elements that do not match return a row filled with NaN. Thus, a Series of messy strings can be “converted” into a like-indexed Series or DataFrame of cleaned-up or more useful strings, without necessitating `get()` to access tuples or `re.match` objects. The `dtype` of the result is always object, even if no match is found and the result only contains NaN. Note that any capture group names in the regular expression will be used for column names; otherwise capture group numbers will be used. 

Extracting a regular expression with one group returns a DataFrame with one column if `expand=True`.It returns a Series if `expand=False`. Calling on an Index with a regex with more than one capture group returns a DataFrame if `expand=True`. It raises ValueError if `expand=False`.

```python
pd.Series(['a1', 'b2', 'c3']).str.extract(r'(?P<letter>[ab])(?P<digit>\d)',expand=False)
```

the `extractall` method returns every match. The result of `extractall` is always a DataFrame with a `MultiIndex` on its rows. The last level of the `MultiIndex` is named match and indicates the order in the subject. When each subject string in the Series has exactly one match,then `extractall(pat).xs(0, level='match')` gives the same result as `extract(pat)`. Index also supports `.str.extractall`. It returns a DataFrame which has the same result as a `Series.str.extractall` with a default index. 

The distinction between `match` and `contains` is strictness: `match` relies on strict `re.match`, while `contains` relies on `re.search`. Methods like `match`, `contains, startswith`, and `endswith` take an extra `na` argument so missing values can be considered `True` or `False`.

| Method            | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `wrap()`          | Split long strings into lines with length less than a given width |
| `slice()`         | Slice each string in the Series                              |
| `slice_replace()` | Replace slice in each string with passed value               |
| `findall()`       | Compute list of all occurrences of pattern/regex for each string |
| `count()`         | Count occurrences of pattern                                 |

Nearly all Python's built-in string methods are mirrored by a Pandas vectorized string method. Here is a list of Pandas ``str`` methods that mirror Python string methods:

|              |                  |                  |                  |
| ------------ | ---------------- | ---------------- | ---------------- |
| ``len()``    | ``lower()``      | ``translate()``  | ``islower()``    |
| ``ljust()``  | ``upper()``      | ``startswith()`` | ``isupper()``    |
| ``rjust()``  | ``find()``       | ``endswith()``   | ``isnumeric()``  |
| ``center()`` | ``rfind()``      | ``isalnum()``    | ``isdecimal()``  |
| ``zfill()``  | ``index()``      | ``isalpha()``    | ``split()``      |
| ``strip()``  | ``rindex()``     | ``isdigit()``    | ``rsplit()``     |
| ``rstrip()`` | ``capitalize()`` | ``isspace()``    | ``partition()``  |
| ``lstrip()`` | ``swapcase()``   | ``istitle()``    | ``rpartition()`` |

In addition, there are several methods that accept regular expressions to examine the content of each string element, and follow some of the API conventions of Python's built-in ``re`` module:

| Method         | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| ``match()``    | Call ``re.match()`` on each element, returning a boolean.    |
| ``extract()``  | Call ``re.match()`` on each element, returning matched groups as strings. |
| ``findall()``  | Call ``re.findall()`` on each element                        |
| ``replace()``  | Replace occurrences of pattern with some other string        |
| ``contains()`` | Call ``re.search()`` on each element, returning a boolean    |
| ``count()``    | Count occurrences of pattern                                 |
| ``split()``    | Equivalent to ``str.split()``, but accepts regexps           |
| ``rsplit()``   | Equivalent to ``str.rsplit()``, but accepts regexps          |

Finally, there are some miscellaneous methods that enable other convenient operations:

| Method              | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| ``get()``           | Index each element                                           |
| ``slice()``         | Slice each element                                           |
| ``slice_replace()`` | Replace slice in each element with passed value              |
| ``cat()``           | Concatenate strings                                          |
| ``repeat()``        | Repeat values                                                |
| ``normalize()``     | Return Unicode form of string                                |
| ``pad()``           | Add whitespace to left, right, or both sides of strings      |
| ``wrap()``          | Split long strings into lines with length less than a given width |
| ``join()``          | Join strings in each element of the Series with passed separator |
| ``get_dummies()``   | extract dummy variables as a dataframe                       |

##### 