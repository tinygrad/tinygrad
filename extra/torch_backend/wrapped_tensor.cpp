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

static caffe2::TypeMeta dtypeFromName(const std::string &dtype_name) {
  if (dtype_name == "float") { return caffe2::TypeMeta::Make<float>();
  } else if (dtype_name == "double") { return caffe2::TypeMeta::Make<double>();
  } else if (dtype_name == "int") { return caffe2::TypeMeta::Make<int32_t>();
  } else if (dtype_name == "long") { return caffe2::TypeMeta::Make<int64_t>();
  } else if (dtype_name == "bool") { return caffe2::TypeMeta::Make<bool>();
  } else if (dtype_name == "char") { return caffe2::TypeMeta::Make<char>();
  } else if (dtype_name == "unsigned char") { return caffe2::TypeMeta::Make<unsigned char>();
  }
  throw std::runtime_error("Unsupported dtype: " + dtype_name);
}

at::Tensor wrap_tensor(py::object &py_obj) {
  // TODO: we have to get the dtype and the shape from the tinygrad Tensor
  std::vector<int64_t> sizes = py_obj.attr("shape").cast<std::vector<int64_t>>();
  std::string dtype_name = py_obj.attr("dtype").attr("name").cast<std::string>();

  return at::detail::make_tensor<at::OpaqueTensorImpl<TinyTensor>>(
    at::DispatchKeySet(at::DispatchKey::PrivateUse1),
    dtypeFromName(dtype_name),
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
