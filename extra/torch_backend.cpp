#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <torch/extension.h>

// Python factory function where real implementations can be found
PyObject* py_factory;

// Setter for the python dictionary with implementations
void set_impl_factory(PyObject* factory) {
  py_factory = factory;
}

py::function get_method(const char* name) {
  auto factory = py::cast<py::function>(py_factory);
  return factory(name);
}


namespace {

static c10::DeviceIndex device_count() {
  py::gil_scoped_acquire acquire;
  return get_method("deviceCount")().cast<c10::DeviceIndex>();
}

struct OpenRegGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  c10::DeviceType type() const override {
    return static_type;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    py::gil_scoped_acquire acquire;
    auto old_device_index = get_method("exchangeDevice")(d.index()).cast<c10::DeviceIndex>();
    return c10::Device(static_type, old_device_index);
  }

  c10::Device getDevice() const override {
    return c10::Device(static_type, 0);
  }

  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    py::gil_scoped_acquire acquire;
    auto device = get_method("setDevice")(d.index());
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    py::gil_scoped_acquire acquire;
    auto device = get_method("uncheckedSetDevice")(d.index());
  }

  c10::Stream getStream(c10::Device d) const noexcept override {
    py::gil_scoped_acquire acquire;
    auto stream_id = get_method("getStream")(d.index()).cast<c10::StreamId>();
    return c10::Stream(c10::Stream::UNSAFE, d, stream_id);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    py::gil_scoped_acquire acquire;
    auto stream_id = get_method("exchangeStream")(s).cast<c10::StreamId>();
    return c10::Stream(c10::Stream::UNSAFE, s.device(), stream_id);
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }
};

// Register our dummy allocator
//static OpenRegAllocator global_openreg_alloc;
//REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_openreg_alloc);

// Empty op needs C++ code and cannot be handled by python side fallback
at::Tensor empty_openreg(
  c10::IntArrayRef size,
  std::optional<c10::ScalarType> dtype_opt,
  std::optional<c10::Layout> layout_opt,
  std::optional<c10::Device> device_opt,
  std::optional<bool> pin_memory_opt,
  std::optional<c10::MemoryFormat> memory_format_opt) {
const auto device = c10::device_or_default(device_opt);
const auto dtype = c10::dtype_or_default(dtype_opt);
TORCH_CHECK(device.is_privateuseone());
TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided, "Non strided layout not supported");
TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt), "Pin memory can only be on CPU");
const c10::DeviceGuard device_guard(device);
constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
return at::detail::empty_generic(
    size, &global_openreg_alloc, pu1_dks, dtype, memory_format_opt);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", empty_openreg);
}

// Register our device guard
C10_REGISTER_GUARD_IMPL(PrivateUse1, OpenRegGuardImpl);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
