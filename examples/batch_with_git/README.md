Do NOT run the scripts, e.g. `10input.py`, in this dir directly because that
will create git commits in psweep's repo (here)!

Instead, copy this dir to a safe location (e.g. `/tmp/batch_with_git`) and then
run it, or better yet use `run_example.sh`.

If you by mistake did run `10input.py` here, just

```sh
$ git reset --hard HEAD~
```

Example:

```sh
$ cp -r batch_with_git /tmp/
$ cd /tmp/batch_with_git
$ ./run_example.sh

# clear and repeat
$ rm -rf calc .git*; cp -r /path/to/psweep/examples/batch_with_git/* .; ./run_example.sh
```
