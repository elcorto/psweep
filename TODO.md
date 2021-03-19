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

* Check for existing run_id and pset_id in db. It may be very
  unlikely but we don't check for that!
  pset_id: serial: pass df or list of existing pset_ids to worker??

* Add git workflow also to run_local() if we need it, for now only in
  prep_batch()

* make test_run_all_examples run in parallel
