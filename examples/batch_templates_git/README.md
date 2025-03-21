Same as `batch_templates/`, but additionally using git to track changes. Also
we use the older template "dollar" syntax here for demonstration, but recommend
to use jinja instead, as we do in `batch_templates/`.

Do NOT run the scripts, e.g. `10input.py`, in this dir directly because that
will create git commits in psweep's repo (here)!

Instead, copy this dir to a safe location (e.g. `/tmp/batch_templates_git`) and then
run it, or better yet use `run_example.sh`.

If you by mistake did run `10input.py` here, just

```sh
$ git reset --hard HEAD~
```

Example:

```sh
$ cp -r batch_templates_git /tmp/
$ cd /tmp/batch_templates_git
$ ./run_example.sh

# clear and repeat
$ rm -rf calc .git*; cp -r /path/to/psweep/examples/batch_templates_git/* .; ./run_example.sh
```
