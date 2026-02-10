#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/extension.h>
#include <torch/csrc/PyInterpreter.h>
#include <ATen/OpaqueTensorImpl.h>
#include <ATen/Context.h>
#include <limits>
#include <cstdlib>
#include <vector>

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
      c10::Storage&& storage,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      int64_t storage_offset)
      : OpaqueTensorImpl<OpaqueHandle>(key_set, data_type, device, opaque_handle, sizes) {
    this->storage_ = std::move(storage);
    this->sizes_and_strides_.set_strides(strides);
    this->storage_offset_ = storage_offset;
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<TinyOpaqueTensorImpl<OpaqueHandle>>(
      this->key_set(), this->dtype(), this->device(), this->opaque_handle(), c10::Storage(this->unsafe_storage()),
      this->sizes_and_strides_.sizes_arrayref(), this->sizes_and_strides_.strides_arrayref(), this->storage_offset_);
    copy_tensor_metadata(this, impl.get(), version_counter, allow_tensor_metadata_change);
    impl->refresh_numel();
    return impl;
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<TinyOpaqueTensorImpl<OpaqueHandle>>(
      this->key_set(), this->dtype(), this->device(), this->opaque_handle(), c10::Storage(this->unsafe_storage()),
      this->sizes_and_strides_.sizes_arrayref(), this->sizes_and_strides_.strides_arrayref(), this->storage_offset_);
    copy_tensor_metadata(this, impl.get(), std::move(version_counter), allow_tensor_metadata_change);
    impl->refresh_numel();
    return impl;
  }

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    AT_ASSERT(this->has_compatible_shallow_copy_type(impl->key_set()));
    auto opaque_impl = static_cast<const TinyOpaqueTensorImpl<OpaqueHandle>*>(impl.get());
    copy_tensor_metadata(opaque_impl, this, this->version_counter(), this->allow_tensor_metadata_change());
    this->refresh_numel();
  }

  const c10::Storage& storage() const override { return this->unsafe_storage(); }

#ifdef DEBUG
  bool has_storage() const override { return true; }
#endif

 protected:
  static void copy_tensor_metadata(
      const TinyOpaqueTensorImpl<OpaqueHandle>* src_opaque_impl,
      TinyOpaqueTensorImpl<OpaqueHandle>* dest_opaque_impl,
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(
      src_opaque_impl, dest_opaque_impl, version_counter, allow_tensor_metadata_change);
    dest_opaque_impl->unsafe_opaque_handle() = src_opaque_impl->opaque_handle();
  }

  static void copy_tensor_metadata(
      const TinyOpaqueTensorImpl<OpaqueHandle>* src_opaque_impl,
      TinyOpaqueTensorImpl<OpaqueHandle>* dest_opaque_impl,
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(
      src_opaque_impl, dest_opaque_impl, std::move(version_counter), allow_tensor_metadata_change);
    dest_opaque_impl->unsafe_opaque_handle() = src_opaque_impl->opaque_handle();
  }

 private:
  const char* tensorimpl_type_name() const override { return "TinyOpaqueTensorImpl"; }
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

size_t required_storage_bytes(c10::ScalarType dtype, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, int64_t storage_offset) {
  TORCH_CHECK(sizes.size() == strides.size(), "sizes and strides must have the same rank");
  const size_t itemsize = c10::scalarTypeToTypeMeta(dtype).itemsize();
  bool is_empty = false;
  for (const int64_t s : sizes) {
    TORCH_CHECK(s >= 0, "negative size in tiny tensor metadata");
    if (s == 0) {
      is_empty = true;
      break;
    }
  }
  if (is_empty || itemsize == 0) return 0;
  uint64_t max_index = storage_offset > 0 ? static_cast<uint64_t>(storage_offset) : 0;
  for (size_t i = 0; i < sizes.size(); i++) {
    const uint64_t span = sizes[i] > 0 ? static_cast<uint64_t>(sizes[i] - 1) : 0;
    const uint64_t stride = static_cast<uint64_t>(std::llabs(strides[i]));
    TORCH_CHECK(stride == 0 || span <= (std::numeric_limits<uint64_t>::max() - max_index) / stride, "tiny storage index overflow");
    max_index += span * stride;
  }
  const uint64_t required_elems = max_index + 1;
  TORCH_CHECK(required_elems <= std::numeric_limits<size_t>::max() / itemsize, "tiny storage byte size overflow");
  return static_cast<size_t>(required_elems * itemsize);
}

static c10::Storage make_tiny_storage(size_t size_bytes, c10::DeviceIndex device_index) {
  auto* allocator = at::getCPUAllocator();
  at::DataPtr data_ptr = allocator->allocate(size_bytes);
  data_ptr.unsafe_set_device(at::Device(at::kPrivateUse1, device_index));
  return c10::Storage(c10::Storage::use_byte_size_t(), size_bytes, std::move(data_ptr), allocator, false);
}

at::Tensor wrap_tensor(py::object &py_obj, c10::ScalarType dtype, c10::DeviceIndex device_index) {
  std::vector<int64_t> sizes = py_obj.attr("shape").cast<std::vector<int64_t>>();
  std::vector<int64_t> strides = py_obj.attr("_strides").cast<std::vector<int64_t>>();
  int64_t storage_offset = py_obj.attr("_storage_offset").cast<int64_t>();
  c10::Storage storage = make_tiny_storage(required_storage_bytes(dtype, sizes, strides, storage_offset), device_index);
  return at::detail::make_tensor<at::TinyOpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>>(
    at::DispatchKeySet(at::DispatchKey::PrivateUse1),
    c10::scalarTypeToTypeMeta(dtype),
    at::Device(at::kPrivateUse1, device_index),
    std::make_shared<c10::SafePyObject>(py_obj.release().ptr(), getPyInterpreter()),
    std::move(storage),
    sizes, strides, storage_offset);
}

py::object unwrap_tensor(const at::Tensor &tensor) {
  auto* impl = tensor.unsafeGetTensorImpl();
  auto* opaque_impl = dynamic_cast<at::TinyOpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>*>(impl);
  TORCH_CHECK(opaque_impl != nullptr, "cannot unwrap non-tiny tensor impl");
  std::shared_ptr<c10::SafePyObject> tiny = opaque_impl->opaque_handle();
  return py::reinterpret_borrow<py::object>(tiny->ptr(getPyInterpreter()));
}

void update_metadata(const at::Tensor &tensor, const std::vector<int64_t> &sizes,
                     const std::vector<int64_t> &strides, int64_t storage_offset) {
  auto* impl = tensor.unsafeGetTensorImpl();
  impl->set_allow_tensor_metadata_change(true);
  impl->set_sizes_and_strides(sizes, strides, storage_offset);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wrap", &wrap_tensor);
  m.def("unwrap", &unwrap_tensor);
  m.def("update_metadata", &update_metadata);
}
