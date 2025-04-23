#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/extension.h>
#include <torch/csrc/PyInterpreter.h>
#include <ATen/OpaqueTensorImpl.h>
#include <ATen/core/GeneratorForPrivateuseone.h>

struct TinygradGeneratorImpl : public c10::GeneratorImpl {
  py::object rng;  // Tinygrad RandomGenerator instance

  TinygradGeneratorImpl(c10::DeviceIndex device_index)
    : GeneratorImpl(
        c10::Device(c10::DeviceType::PrivateUse1, device_index),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)
      ) {
    py::gil_scoped_acquire gil;
    py::object rng_class = py::module::import("tinygrad.tensor").attr("RandomGenerator");
    rng = rng_class();
  }
  void set_current_seed(uint64_t seed) override {
    py::gil_scoped_acquire gil;
    rng.attr("manual_seed")(seed);
  }
  uint64_t current_seed() const override {
    py::gil_scoped_acquire gil;
    return rng.attr("seed").cast<uint64_t>();
  }
  uint64_t seed() override {
    auto random = c10::detail::getNonDeterministicRandom(true);
    this->set_current_seed(random);
    return random;
  }
  GeneratorImpl* clone_impl() const override {
    auto gen = new TinygradGeneratorImpl(device().index());
    {
      py::gil_scoped_acquire gil;
      py::object deepcopy = py::module::import("copy").attr("deepcopy");
      gen->rng = deepcopy(rng);
    }
    return gen;
  }
  void set_state(const c10::TensorImpl&) override {}
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override { return nullptr; }
  void set_offset(uint64_t) override {}
  uint64_t get_offset() const override { return 0; }
};

at::Generator make_custom_generator(c10::DeviceIndex device_index) {
  return at::make_generator<TinygradGeneratorImpl>(device_index);
}
REGISTER_GENERATOR_PRIVATEUSE1(make_custom_generator);

py::object get_rng_from_generator(const at::Generator &gen) {
  auto* impl = dynamic_cast<TinygradGeneratorImpl*>(gen.unsafeGetGeneratorImpl());
  if (!impl) throw std::runtime_error("Not a TinygradGeneratorImpl");
  return impl->rng;
}

// register guard
namespace at {
namespace detail {
//C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);
// NOTE: pytorch's no-op class throws error on backwards with events/streams
// TODO: why are there events in autograd?
struct CustomNoOpDeviceGuardImpl : public c10::impl::DeviceGuardImplInterface
{
  static const DeviceType D = DeviceType::PrivateUse1;
  CustomNoOpDeviceGuardImpl() = default;
  DeviceType type() const override {
    return D;
  }
  Device exchangeDevice(Device) const override {
    return Device(D, 0); // no-op
  }
  Device getDevice() const override {
    return Device(D, 0);
  }
  void setDevice(Device) const override {
    // no-op
  }
  void uncheckedSetDevice(Device) const noexcept override {
    // no-op
  }
  Stream getStream(Device) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  Stream getDefaultStream(Device) const override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  Stream getStreamFromGlobalPool(Device, bool isHighPriority = false)
      const override {
    // no-op
    (void)isHighPriority;
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  Stream getNewStream(Device, int priority = 0) const override {
    // no-op
    (void)priority;
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(D, 0));
  }
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }
  // Event-related functions
  void record(
      void** /*event*/,
      const Stream& /*stream*/,
      const DeviceIndex /*device_index*/,
      const EventFlag /*flag*/) const override {
    //TORCH_CHECK(false, D, " backend doesn't support events.");
  }
  void block(void* /*event*/, const Stream& /*stream*/) const override {
    //TORCH_CHECK(false, D, " backend doesn't support events.")
  }
  bool queryEvent(void* /*event*/) const override {
    //TORCH_CHECK(false, D, " backend doesn't support events.")
    return true;
  }
  void destroyEvent(void* /*event*/, const DeviceIndex /*device_index*/)
      const noexcept override {}
  // Stream-related functions
  bool queryStream(const Stream& /*stream*/) const override {
    return true;
  }
  void synchronizeStream(const Stream& /*stream*/) const override {
    // Don't wait for anything.
  }
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, CustomNoOpDeviceGuardImpl);
}

template <typename OpaqueHandle>
struct TinyOpaqueTensorImpl : public OpaqueTensorImpl<OpaqueHandle> {
  TinyOpaqueTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      int64_t storage_offset)
      : OpaqueTensorImpl<OpaqueHandle>(key_set, data_type, device, opaque_handle, sizes) {
    this->sizes_and_strides_.set_strides(strides);
    this->storage_offset_ = storage_offset;
  }
};
}

struct OpenRegHooksInterface : public at::PrivateUse1HooksInterface {
  // NOTE: no idea what this is
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override { return true; }
};

int register_hook() {
  at::RegisterPrivateUse1HooksInterface(new OpenRegHooksInterface());
  return 0;
}
int temp_register_hook = register_hook();

at::Tensor wrap_tensor(py::object &py_obj, c10::ScalarType dtype, c10::DeviceIndex device_index) {
  // TODO: we have to get the dtype and the shape from the tinygrad Tensor
  std::vector<int64_t> sizes = py_obj.attr("shape").cast<std::vector<int64_t>>();

  py::list views = py_obj.attr("lazydata").attr("st").attr("views");
  std::vector<int64_t> strides = views[views.size() - 1].attr("strides").cast<std::vector<int64_t>>();
  int64_t storage_offset = 0;
  for (auto& v: views) {
    storage_offset += v.attr("offset").cast<int64_t>(); // TODO: is this correct?
  }

  return at::detail::make_tensor<at::TinyOpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>>(
    at::DispatchKeySet(at::DispatchKey::PrivateUse1),
    c10::scalarTypeToTypeMeta(dtype),
    at::Device(at::kPrivateUse1, device_index),
    std::make_shared<c10::SafePyObject>(py_obj.release().ptr(), getPyInterpreter()),
    sizes, strides, storage_offset);
}

py::object unwrap_tensor(const at::Tensor &tensor) {
  auto* impl = tensor.unsafeGetTensorImpl();
  auto* opaque_impl = static_cast<at::TinyOpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>*>(impl);
  std::shared_ptr<c10::SafePyObject> tiny = opaque_impl->opaque_handle();
  return py::reinterpret_borrow<py::object>(tiny->ptr(getPyInterpreter()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wrap", &wrap_tensor);
  m.def("unwrap", &unwrap_tensor);
  m.def("get_rng_from_generator", &get_rng_from_generator);
}
