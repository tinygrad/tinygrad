#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/extension.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

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
      DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      int64_t storage_offset,
      c10::Storage storage)
      : OpaqueTensorImpl<OpaqueHandle>(key_set, data_type, device, opaque_handle, sizes) {
    this->sizes_and_strides_.set_strides(strides);
    this->storage_offset_ = storage_offset;
    this->set_storage_keep_dtype(std::move(storage));
    this->storage_access_should_throw_ = false;
  }

  void set_size(int64_t dim, int64_t new_size) override {
    TensorImpl::set_size(dim, new_size);
  }

  void set_stride(int64_t dim, int64_t new_stride) override {
    TensorImpl::set_stride(dim, new_stride);
  }

  void set_storage_offset(int64_t storage_offset) override {
    TensorImpl::set_storage_offset(storage_offset);
  }
};

} // namespace at

namespace {

struct TinyStorageContext {
  PyObject* base;
};

void tiny_storage_deleter(void* ctx);

struct TinyStorageCache final {
  std::mutex mutex;
  std::unordered_map<PyObject*, c10::weak_intrusive_ptr<c10::StorageImpl>> cache;

  c10::Storage get_or_create(const py::object& base, size_t nbytes, const c10::Device& device) {
    PyObject* base_ptr = base.ptr();
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto it = cache.find(base_ptr);
      if (it != cache.end()) {
        if (auto existing = it->second.lock()) {
          auto storage = c10::Storage(std::move(existing));
          if (storage.nbytes() < nbytes) storage.set_nbytes(nbytes);
          return storage;
        }
        cache.erase(it);
      }
    }

    auto* context = new TinyStorageContext{base_ptr};
    Py_INCREF(base_ptr);
    auto data_ptr = c10::DataPtr(nullptr, context, &tiny_storage_deleter, device);
    c10::Storage storage(c10::Storage::use_byte_size_t(), nbytes, std::move(data_ptr), nullptr, false);

    {
      std::lock_guard<std::mutex> guard(mutex);
      cache.emplace(base_ptr, storage.getWeakStorageImpl());
    }
    return storage;
  }
};

TinyStorageCache& get_storage_cache() {
  static TinyStorageCache cache;
  return cache;
}

void tiny_storage_deleter(void* ctx) {
  auto* context = static_cast<TinyStorageContext*>(ctx);
  {
    py::gil_scoped_acquire gil;
    Py_DECREF(context->base);
    auto& cache = get_storage_cache();
    std::lock_guard<std::mutex> guard(cache.mutex);
    auto it = cache.cache.find(context->base);
    if (it != cache.cache.end() && it->second.expired()) cache.cache.erase(it);
  }
  delete context;
}

py::object get_storage_base(const py::object& tensor) {
  py::object base = tensor;
  std::unordered_set<PyObject*> seen;
  while (py::hasattr(base, "_view_base")) {
    PyObject* current = base.ptr();
    if (!seen.insert(current).second) break;
    py::object next = base.attr("_view_base");
    if (next.is_none()) break;
    base = next;
  }
  return base;
}

} // namespace

struct OpenRegHooksInterface : public at::PrivateUse1HooksInterface {
  // NOTE: no idea what this is
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override { return true; }
};

int register_hook() {
  at::RegisterPrivateUse1HooksInterface(new OpenRegHooksInterface());
  return 0;
}
int temp_register_hook = register_hook();

at::Tensor wrap_tensor(py::object& py_obj, c10::ScalarType dtype, c10::DeviceIndex device_index) {
  py::object tensor = py_obj;
  py::object base = get_storage_base(tensor);
  std::vector<int64_t> sizes = tensor.attr("shape").cast<std::vector<int64_t>>();
  std::vector<int64_t> strides = tensor.attr("_strides").cast<std::vector<int64_t>>();
  int64_t storage_offset = tensor.attr("_storage_offset").cast<int64_t>();
  int64_t base_numel = base.attr("numel")().cast<int64_t>();
  const auto type_meta = c10::scalarTypeToTypeMeta(dtype);
  size_t nbytes = base_numel * type_meta.itemsize();
  c10::Device device(at::kPrivateUse1, device_index);
  c10::Storage storage = get_storage_cache().get_or_create(base, nbytes, device);

  return at::detail::make_tensor<
      at::TinyOpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>>(
      at::DispatchKeySet(at::DispatchKey::PrivateUse1),
      type_meta,
      device,
      std::make_shared<c10::SafePyObject>(py_obj.release().ptr(), getPyInterpreter()),
      sizes,
      strides,
      storage_offset,
      std::move(storage));
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
}
