# Adding Modules

**TODO** Awaiting documentation will cover...

- Adding a Python module under `src/afmslicer/<module>.py`
- Incorporating it into `src/afmslicer/run_modules.py::process()` so that the module will run as part of the end to end
  processing.
- Adding additional command line argument options to the `process` sub-command.
- Adding an additional sub-command (aka `sub-parser`) and writing functions in `src/afmslicer/run_modules.py` to wrap
  functionality.
- Writing a `src/afmslicer/processing.py::<function>_scan()` to run the module in isolation in parallel.
- Add configuration options to `src/afmslicer/default_config.yaml`.
- Update `src/afmslicer/validation.py` with new configuration options that have been added.
