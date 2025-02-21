#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/extension.h>
#include <ATen/OpaqueTensorImpl.h>

// register guard
namespace at {
namespace detail {
C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);
}
}

// code from chatgpt
struct GILSafeDeleter {
  void operator()(PyObject* ptr) const {
    if (ptr) {
      py::gil_scoped_acquire gil;
      Py_DECREF(ptr);
    }
  }
};

class TinyTensor {
private:
  // We wrap the PyObject* inside a shared_ptr so the GILSafeDeleter runs on destruction.
  std::shared_ptr<PyObject> obj_;

public:
  TinyTensor() : obj_(nullptr, GILSafeDeleter()) {}

  // From a py::object
  TinyTensor(const py::object& o)
  : obj_(o.inc_ref().ptr(), GILSafeDeleter()) {
    // o.inc_ref() bumps the PyObject reference count; we store the pointer in shared_ptr
  }

  // Optional move or copy ctors if needed:
  TinyTensor(const TinyTensor &other) = default;
  TinyTensor(TinyTensor &&other) = default;
  TinyTensor& operator=(const TinyTensor &other) = default;
  TinyTensor& operator=(TinyTensor &&other) = default;

  py::object get_py_obj() const {
    if (!obj_) {
      return py::none();
    }
    // Safely borrow as a py::object (we must hold the GIL).
    py::gil_scoped_acquire gil;
    return py::reinterpret_borrow<py::object>(obj_.get());
  }
};

at::Tensor wrap_tensor(py::object &py_obj) {
  // TODO: we have to get the dtype and the shape from the tinygrad Tensor
  std::vector<int64_t> sizes = py_obj.attr("shape").cast<std::vector<int64_t>>();

  return at::detail::make_tensor<at::OpaqueTensorImpl<TinyTensor>>(
    at::DispatchKeySet(at::DispatchKey::PrivateUse1),
    caffe2::TypeMeta::Make<float>(),
    at::Device(at::kPrivateUse1),
    TinyTensor(py_obj),
    sizes);
}

py::object unwrap_tensor(const at::Tensor &tensor) {
  auto* impl = tensor.unsafeGetTensorImpl();
  auto* opaque_impl = static_cast<at::OpaqueTensorImpl<TinyTensor>*>(impl);
  const TinyTensor &tiny = opaque_impl->opaque_handle();
  return tiny.get_py_obj();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wrap", &wrap_tensor);
  m.def("unwrap", &unwrap_tensor);
}
