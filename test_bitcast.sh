METAL=1 python3 -m pytest -n=auto test/external/external_test_onnx_backend.py
METAL=1 python3 -m pytest -n=auto test/test_dtype.py::TestBitCast::test_float32_bitcast_to_int32
METAL=1 python3 load_llama.py
