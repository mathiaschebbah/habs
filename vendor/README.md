# Unmodified analysis executor

`run_cells.py` is deployed byte-for-byte from the installed
`audit-ml-pipeline/scripts/run_cells.py` skill. The HABS container cannot access
the desktop skill directory. This deployment copy is not a second implementation.

The entry point also checks the text for failed cells and the expected final
summary, because the original executor continues after a cell error.
