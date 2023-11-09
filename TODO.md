* add df_refresh_pset_hash(df, columns=None) to be applied after we
  manually add a column to the database that should be part of the pset;
  columns default should be all which don't start w/ "_"

* in future versions, add a prefix to all pset variables such that we
  have a clear distinction in the db:

  ```
  book keeping
    _pset_id
    _run_id
    _pset_sha1
    ...

  pset content
    pset_foo
    pset_bar
    ...

  results added by worker (run_local()) and/or eval scripts (mostly after
  prep_batch())
    baz
    boing
    ...
  ```
