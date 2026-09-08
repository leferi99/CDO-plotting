# The field schema

`cdo.schema` maps a canonical field name to every HDF5 path that quantity has lived at, and reads it from a file regardless of the DREAM version or run setup. The whole compatibility layer is this one table plus the resolver that walks it.

## The alias table

`FIELDS` maps each canonical name to a `Field`. A `Field` records the paths and how the field behaves.

```python
"major_radius": _f("grid/R0", "settings/radialgrid/R0", required=True, unit="m"),
"gammaHottail": _fluid("other/fluid/gammaFhot", "other/fluid/gammaHottail"),
```

The `Field` attributes:

- `paths`: the HDF5 paths to try, in order. The first that exists as a dataset is used.
- `time`: how the leading axis relates to `grid/t`. `FULL` for `eqsys` datasets, which carry every time step and have their initial step dropped on concatenation. `STEPS` for `other` datasets, which carry one step fewer and are already aligned. `NONE` for grids and scalars.
- `required`: reading an absent required field raises `MissingFieldError`. Others return `None`.
- `grid`: `"hottail"` or `"runaway"` when the field only exists with that momentum grid enabled.
- `group`: the `settings/other/include` entry the field belongs to, `"fluid"` or `"scalar"`, for `other/` quantities.
- `unit`, `note`: for display and documentation.

`_f` builds a general field. `_fluid` sets `time=STEPS` and `group="fluid"` for `other/fluid` quantities.

## Reading

`Resolver(handle)` reads canonical names from one open file. `Run` uses it internally; use it directly for one file.

```python
r = cdo.Resolver(cdo.open_output(path))
r.read("major_radius")        # array, or None if absent and not required
r.read_string_list("ion_names")
r.report()                    # which fields are present, relocated, or absent
```

Absence is classified by cause:

- `GATED`: the momentum grid was disabled, or the `other/` group was excluded. Normal, silent.
- `NOT_PRODUCED`: an `other/` field whose group was included but which the run did not compute. Normal.
- `MISSING`: an `eqsys` or `grid` field absent with every gate open. The only category worth attention.

`cdo.survey(handle)` classifies every field without reading the data, cheap on large files. `cdo.describe(path)` prints the result.

When several listed paths for one quantity exist in a file, the first wins and a `DuplicatePathWarning` is raised only if the copies disagree.

## Adding a DREAM era

Adding support for a new output layout is a table edit.

- A field moved to a new path: add the new path to that field's `paths`, before the old ones.
- A field renamed: add the new name as the first path. A rename inside `other/` cannot be distinguished from an inactive module by the file alone, so both names stay in the table.
- A new quantity: add a `Field` entry with its path, its `time` axis, and any `grid` or `group` gate.

Run the resolver against a file from the new era with `cdo.describe(path)` and check that nothing needed lands in the `MISSING` category.
