## Torch Backend Fixes

- replaced `uop.st` usage with lightweight view tracking (`_view_ops` + `_view_st`) recorded at every view op
- taught `_as_strided` to rebuild views purely from recorded movement ops and a tiny local movement-op converter
- made `wrap`/C++ glue consume the tracked `ShapeTracker`, not `uop.st`
- added slice/step-aware metadata recording + `ShapeTracker/View.slice` so we can replay nested slices/in-place ops
- reran `PYTHONPATH=. .venv/bin/python extra/torch_backend/test.py` (local cache dirs); all functional tests pass, only `test_mnist_index` fails due to offline MNIST download
