# Pysqlfilter
This package provides a parser for a simplified SQL where clause conditional grammer, along with basic typechecking against a set of columns and the ability to emit clean SQL code for use in a query. It is intended to help support users in creating custom conditional filter statements in applications such as an analytics dashboard, with the knowledge that those filter statements are free from SQL injections or go beyond the capabilities of the grammer.

## Usage
```python
import pysqlfilter

columns = {'s': 'text', 'b': 'bool', 'i': 'int', 'f': 'real'}
pysqlfilter.parser.build_sql('s + \'lit\' > 1 and not b', columns=columns)
# (((s + \'lit\') > 1) and (not b))

pysqlfilter.parser.build_sql('1 + i / f > -0.1', columns=columns)
# ((1 + (i / f)) > (- 0.1))'

pysqlfilter.parser.build_sql('1 - s = f', columns=columns)
# throws: pysqlfilter.TypecheckError: expected integer, got varchar
```

## Development
For now all code lives in the `sqlfilter.py` file, and you can run a simple set of tests with `python sqlfilter.py`. You should see `----- Passed -----` at the end of the output if the tests work. (I'll switch this to pytest or something eventually)

Running `python sqlfilter.py` will also regenerate the `railroad.svg` file with any changes.

## Grammer
[Railroad diagram](./railroad.svg)
