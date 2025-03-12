#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/extension.h>
#include <torch/csrc/PyInterpreter.h>
#include <ATen/OpaqueTensorImpl.h>

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
    // TODO: stub
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

class TinyLazyStorageImpl : public c10::StorageImpl {
public:
  TinyLazyStorageImpl() :
    c10::StorageImpl(c10::StorageImpl::use_byte_size_t(), 0, at::DataPtr(), nullptr, /*resizable=*/false) {
  }
};
template <typename OpaqueHandle>
struct TinyOpaqueTensorImpl : public OpaqueTensorImpl<OpaqueHandle> {
  TinyOpaqueTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides)
      : OpaqueTensorImpl<OpaqueHandle>(key_set, data_type, device, opaque_handle, sizes) {
    this->sizes_and_strides_.set_strides(strides);
    this->storage_ = c10::Storage(c10::make_intrusive<TinyLazyStorageImpl>());
    TensorImpl::storage_access_should_throw_ = false;
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

  // Last dimension stride is 1 for contiguous row-major layout
  std::vector<int64_t> strides(sizes.size());
  if (sizes.size() >= 1) {
    strides[sizes.size() - 1] = 1;

    // Compute strides from right to left
    for (int64_t i = sizes.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * sizes[i + 1];
    }
  }

  return at::detail::make_tensor<at::TinyOpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>>(
    at::DispatchKeySet(at::DispatchKey::PrivateUse1),
    c10::scalarTypeToTypeMeta(dtype),
    at::Device(at::kPrivateUse1, device_index),
    std::make_shared<c10::SafePyObject>(py_obj.release().ptr(), getPyInterpreter()),
    sizes, strides);
}

py::object unwrap_tensor(const at::Tensor &tensor) {
  auto* impl = tensor.unsafeGetTensorImpl();
  auto* opaque_impl = static_cast<at::TinyOpaqueTensorImpl<std::shared_ptr<c10::SafePyObject>>*>(impl);
  std::shared_ptr<c10::SafePyObject> tiny = opaque_impl->opaque_handle();
  return py::reinterpret_borrow<py::object>(tiny->ptr(getPyInterpreter()));
}

// NOTE: Distributed pytorch
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <pybind11/chrono.h>

class WorkShim : public c10d::Work {
public:
  WorkShim(c10d::OpType opType, c10::intrusive_ptr<c10::ivalue::Future> future)
    : c10d::Work(-1, opType), future_(std::move(future)) {}
  bool isCompleted() override {
    return future_->completed();
  }
  bool isSuccess() const override {
    return future_->hasValue();
  }
  bool wait(std::chrono::milliseconds timeout = c10d::kUnsetTimeout) override {
    future_->wait();
    return true;
  }
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    return future_;
  }
private:
  c10::intrusive_ptr<c10::ivalue::Future> future_;
};

class PyDistBackend : public c10d::Backend {
public:
  PyDistBackend(int rank, int size, py::object py_backend)
    : c10d::Backend(rank, size), py_backend_(py_backend) {}

  virtual const std::string getBackendName() const override { return "tiny"; }

  c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) override {
    py::gil_scoped_acquire gil;
    py::object py_result = py_backend_.attr("allreduce")(tensors, opts);
    auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
    fut->markCompleted(c10::IValue(tensors));
    return c10::make_intrusive<WorkShim>(c10d::OpType::ALLREDUCE, std::move(fut));
  }

  c10::intrusive_ptr<c10d::Work> broadcast(std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override {
    py::gil_scoped_acquire gil;
    py::object py_result = py_backend_.attr("broadcast")(tensors, opts);
    auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
    fut->markCompleted(c10::IValue(tensors));
    return c10::make_intrusive<WorkShim>(c10d::OpType::BROADCAST, std::move(fut));
  }

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    py::gil_scoped_acquire gil;
    py::object py_result = py_backend_.attr("allgather")(outputTensors, inputTensors, opts);
    auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
    fut->markCompleted(c10::IValue(outputTensors));
    return c10::make_intrusive<WorkShim>(c10d::OpType::ALLGATHER, std::move(fut));
  }
private:
  py::object py_backend_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wrap", &wrap_tensor);
  m.def("unwrap", &unwrap_tensor);
  m.def("register_dist_backend", [](py::object backendcls) {
    auto factory = [backendcls](const c10::intrusive_ptr<c10d::Store>& /*store*/,
                             int rank, int size, const std::chrono::duration<float>& /*timeout*/) -> c10::intrusive_ptr<c10d::Backend> {
        py::object backend = backendcls();
        backend.attr("rank") = rank;
        backend.attr("size") = size;
        return c10::make_intrusive<PyDistBackend>(rank, size, backend);
    };
    py::object torch_distributed = py::module::import("torch.distributed");
    py::object backend_class = torch_distributed.attr("Backend");
    py::list devices; devices.append("tiny");
    backend_class.attr("register_backend")("tiny", py::cpp_function(factory), false, devices);
  });
}

// TODO: do we need autograd for these?
TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("allgather_", torch::CppFunction::makeFallthrough());
  m.impl("broadcast_", torch::CppFunction::makeFallthrough());
  m.impl("allreduce_", torch::CppFunction::makeFallthrough());
}