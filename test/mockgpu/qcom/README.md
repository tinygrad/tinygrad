# QCOM A630 MockGPU

Run tinygrad's QCOM runtime and compiled A630 instructions on a CPU, without a Qualcomm device. The mock implements KGSL requests, compute command packets, instruction execution, workgroup barriers, and float32/float16 2D images. It uses the existing Mesa instruction decoder.

Install the test and Mesa dependencies and a C compiler:

```sh
python -m pip install -e '.[testing_minimal,mesa]'
DEV=MOCK+QCOM:IR3 python -c 'from tinygrad import Tensor; print((Tensor([1., 2.]) + 1).tolist())'
```

QCOM MockGPU uses the compiled CPU host runtime because command submission calls `ioctl`. The emulated device and its mapped buffers share CPU memory. Other MockGPU backends retain their existing host-runtime defaults.

Run the focused tests:

```sh
DEV=MOCK+QCOM:IR3 python -m pytest -n12 test/mockgpu/qcom
DEV=MOCK+QCOM:IR3 IMAGE=1 python -m pytest -n12 test/mockgpu/qcom/test_qcom.py -k image
```

Image support must be enabled before device initialization. `IMAGE=1` adds the image alignment requirement to the QCOM target; setting it later does not configure the same target.

The OpenCL path uses the existing QCOM compiler support, including its qemu or Docker compiler server on non-aarch64 hosts:

```sh
DEV=MOCK+QCOM:CL python -m pytest -n12 test/mockgpu/qcom
DEV=MOCK+QCOM:CL IMAGE=1 FLOAT16=1 python -m pytest -n12 test/mockgpu/qcom/test_qcom.py -k image
```

Run the backend operation tests with the same settings as the MockGPU CI job:

```sh
DEV=MOCK+QCOM:IR3 TRANSCENDENTAL=2 FORWARD_ONLY=1 SKIP_SLOW_TEST=1 python -m pytest -n12 test/backend/test_ops.py
DEV=MOCK+QCOM:CL FORWARD_ONLY=1 SKIP_SLOW_TEST=1 python -m pytest -n12 test/backend/test_ops.py
```

The IR3 invocation uses software transcendental lowering because native NIR large-angle trigonometry has limited precision. These commands exercise forward calculations and retain the existing slow-test skips. The interpreter models functional results rather than instruction timing, caches, or graphics operations. Unsupported instruction/addressing modes raise errors. GPU execution failures are retained across the C callback and reported on synchronization, with the original exception in the cause chain.
