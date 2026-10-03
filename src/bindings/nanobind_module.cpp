#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/sim_context.cuh"
#include "rocketsim_cuda/types/arena_state.cuh"

namespace nb = nanobind;
using namespace rocketsim_cuda;

// CUDA Asynchronous Hardware Event Wrapper for non-blocking benchmarking & synchronization
class GpuEvent {
public:
    cudaEvent_t m_event = nullptr;
    bool m_enable_timing = true;

    explicit GpuEvent(bool enable_timing = true) : m_enable_timing(enable_timing) {
        unsigned int flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
        cudaEventCreateWithFlags(&m_event, flags);
    }

    ~GpuEvent() {
        if (m_event) {
            cudaEventDestroy(m_event);
            m_event = nullptr;
        }
    }

    // Disable copy, allow move
    GpuEvent(const GpuEvent&) = delete;
    GpuEvent& operator=(const GpuEvent&) = delete;
    GpuEvent(GpuEvent&& o) noexcept : m_event(o.m_event), m_enable_timing(o.m_enable_timing) {
        o.m_event = nullptr;
    }
    GpuEvent& operator=(GpuEvent&& o) noexcept {
        if (this != &o) {
            if (m_event) cudaEventDestroy(m_event);
            m_event = o.m_event;
            m_enable_timing = o.m_enable_timing;
            o.m_event = nullptr;
        }
        return *this;
    }

    void record(nb::object stream = nb::none()) {
        cudaStream_t s = nullptr;
        if (!stream.is_none()) {
            if (nb::isinstance<nb::int_>(stream)) {
                s = reinterpret_cast<cudaStream_t>(nb::cast<uintptr_t>(stream));
            } else if (nb::hasattr(stream, "cuda_stream")) {
                s = reinterpret_cast<cudaStream_t>(nb::cast<uintptr_t>(stream.attr("cuda_stream")));
            }
        }
        cudaEventRecord(m_event, s);
    }

    void wait(nb::object stream = nb::none()) {
        cudaStream_t s = nullptr;
        if (!stream.is_none()) {
            if (nb::isinstance<nb::int_>(stream)) {
                s = reinterpret_cast<cudaStream_t>(nb::cast<uintptr_t>(stream));
            } else if (nb::hasattr(stream, "cuda_stream")) {
                s = reinterpret_cast<cudaStream_t>(nb::cast<uintptr_t>(stream.attr("cuda_stream")));
            }
        }
        cudaStreamWaitEvent(s, m_event, 0);
    }

    void synchronize() {
        cudaEventSynchronize(m_event);
    }

    bool query() const {
        return cudaEventQuery(m_event) == cudaSuccess;
    }

    float elapsed_time(const GpuEvent& end) const {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, m_event, end.m_event);
        return ms;
    }
};

// Zero-copy DLPack-compliant GPU Tensor View structure
struct GpuTensorView {
    void* data = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::string dtype = "float32";
    int device_type = 2; // kDLCUDA
    int device_id = 0;
    nb::object owner;

    size_t get_element_size() const {
        if (dtype == "uint8") return 1;
        if (dtype == "int64") return 8;
        return 4; // float32, int32, uint32
    }

    int64_t total_elements() const {
        int64_t count = 1;
        for (auto s : shape) count *= s;
        return count;
    }

    nb::object getitem(nb::object idx_obj) const {
        std::vector<int64_t> indices;
        if (nb::isinstance<nb::tuple>(idx_obj)) {
            nb::tuple t = nb::cast<nb::tuple>(idx_obj);
            for (size_t i = 0; i < t.size(); ++i) {
                indices.push_back(nb::cast<int64_t>(t[i]));
            }
        } else if (nb::isinstance<nb::int_>(idx_obj)) {
            indices.push_back(nb::cast<int64_t>(idx_obj));
        } else {
            throw std::invalid_argument("Index must be an integer or tuple of integers");
        }

        if (indices.size() > shape.size()) {
            throw std::invalid_argument("Too many indices for tensor dimension");
        }

        int64_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            int64_t idx = indices[i];
            if (idx < 0) idx += shape[i];
            if (idx < 0 || idx >= shape[i]) {
                throw std::out_of_range("Index out of range");
            }
            offset += idx * strides[i];
        }

        if (indices.size() == shape.size()) {
            if (dtype == "float32") {
                float val = 0.0f;
                cudaMemcpy(&val, static_cast<const float*>(data) + offset, sizeof(float), cudaMemcpyDeviceToHost);
                return nb::cast(val);
            } else if (dtype == "uint8") {
                uint8_t val = 0;
                cudaMemcpy(&val, static_cast<const uint8_t*>(data) + offset, sizeof(uint8_t), cudaMemcpyDeviceToHost);
                return nb::cast(val);
            } else if (dtype == "uint32") {
                uint32_t val = 0;
                cudaMemcpy(&val, static_cast<const uint32_t*>(data) + offset, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                return nb::cast(val);
            } else if (dtype == "int32") {
                int32_t val = 0;
                cudaMemcpy(&val, static_cast<const int32_t*>(data) + offset, sizeof(int32_t), cudaMemcpyDeviceToHost);
                return nb::cast(val);
            } else if (dtype == "int64") {
                int64_t val = 0;
                cudaMemcpy(&val, static_cast<const int64_t*>(data) + offset, sizeof(int64_t), cudaMemcpyDeviceToHost);
                return nb::cast(val);
            }
            throw std::runtime_error("Unsupported dtype");
        } else {
            GpuTensorView sub;
            sub.data = static_cast<char*>(data) + offset * get_element_size();
            sub.shape.assign(shape.begin() + indices.size(), shape.end());
            sub.strides.assign(strides.begin() + indices.size(), strides.end());
            sub.dtype = dtype;
            sub.device_type = device_type;
            sub.device_id = device_id;
            sub.owner = owner;
            return nb::cast(sub);
        }
    }

    void setitem(nb::object idx_obj, nb::object val_obj) {
        std::vector<int64_t> indices;
        if (nb::isinstance<nb::tuple>(idx_obj)) {
            nb::tuple t = nb::cast<nb::tuple>(idx_obj);
            for (size_t i = 0; i < t.size(); ++i) {
                indices.push_back(nb::cast<int64_t>(t[i]));
            }
        } else if (nb::isinstance<nb::int_>(idx_obj)) {
            indices.push_back(nb::cast<int64_t>(idx_obj));
        } else {
            throw std::invalid_argument("Index must be an integer or tuple of integers");
        }

        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Assignment requires all dimensional indices");
        }

        int64_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            int64_t idx = indices[i];
            if (idx < 0) idx += shape[i];
            if (idx < 0 || idx >= shape[i]) {
                throw std::out_of_range("Index out of range");
            }
            offset += idx * strides[i];
        }

        if (dtype == "float32") {
            float val = nb::cast<float>(val_obj);
            cudaMemcpy(static_cast<float*>(data) + offset, &val, sizeof(float), cudaMemcpyHostToDevice);
        } else if (dtype == "uint8") {
            uint8_t val = nb::cast<uint8_t>(val_obj);
            cudaMemcpy(static_cast<uint8_t*>(data) + offset, &val, sizeof(uint8_t), cudaMemcpyHostToDevice);
        } else if (dtype == "uint32") {
            uint32_t val = nb::cast<uint32_t>(val_obj);
            cudaMemcpy(static_cast<uint32_t*>(data) + offset, &val, sizeof(uint32_t), cudaMemcpyHostToDevice);
        } else if (dtype == "int32") {
            int32_t val = nb::cast<int32_t>(val_obj);
            cudaMemcpy(static_cast<int32_t*>(data) + offset, &val, sizeof(int32_t), cudaMemcpyHostToDevice);
        } else if (dtype == "int64") {
            int64_t val = nb::cast<int64_t>(val_obj);
            cudaMemcpy(static_cast<int64_t*>(data) + offset, &val, sizeof(int64_t), cudaMemcpyHostToDevice);
        }
    }

    void zero_() {
        int64_t total_bytes = total_elements() * get_element_size();
        cudaMemset(data, 0, total_bytes);
    }

    GpuTensorView clone() const {
        int64_t total = total_elements();
        size_t elem_size = get_element_size();
        size_t total_bytes = total * elem_size;

        void* d_new = nullptr;
        cudaMalloc(&d_new, total_bytes);
        cudaMemcpy(d_new, data, total_bytes, cudaMemcpyDeviceToDevice);

        GpuTensorView c;
        c.data = d_new;
        c.shape = shape;
        c.strides = strides;
        c.dtype = dtype;
        c.device_type = device_type;
        c.device_id = device_id;
        nb::capsule cleanup(d_new, [](void* p) noexcept {
            if (p) cudaFree(p);
        });
        c.owner = cleanup;
        return c;
    }

    nb::object to_dlpack(nb::object /* stream */ = nb::none()) const {
        std::vector<size_t> u_shape(shape.begin(), shape.end());
        nb::dlpack::dtype dt = nanobind::dtype<float>();
        if (dtype == "uint8") {
            dt = nanobind::dtype<uint8_t>();
        } else if (dtype == "int32") {
            dt = nanobind::dtype<int32_t>();
        } else if (dtype == "uint32") {
            dt = nanobind::dtype<uint32_t>();
        } else if (dtype == "int64") {
            dt = nanobind::dtype<int64_t>();
        }

        nb::ndarray<nb::device::cuda> arr(
            data,
            u_shape.size(),
            u_shape.data(),
            owner,
            strides.data(),
            dt,
            device_type,
            device_id
        );
        return nb::cast(arr);
    }
};

NB_MODULE(rocketsim_cuda, m) {
    m.doc() = "RocketSim-CUDA: Ultra-parallel GPU physics simulation engine with zero-copy PyTorch/DLPack tensors";

    // Global Physics & Timing Constants
    m.attr("TICK_RATE") = TICK_RATE;
    m.attr("DELTA_TIME") = DELTA_TIME;
    m.attr("DEFAULT_MAX_ENVS") = DEFAULT_MAX_ENVS;
    m.attr("MAX_CARS_PER_ENV") = MAX_CARS_PER_ENV;
    m.attr("MAX_BOOST_PADS") = MAX_BOOST_PADS;
    m.attr("CAR_MASS") = CAR_MASS;
    m.attr("BALL_MASS") = BALL_MASS;
    m.attr("BALL_RADIUS") = BALL_RADIUS;
    m.attr("BALL_REST_Z") = BALL_REST_Z;
    m.attr("BOOST_MAX") = BOOST_MAX;

    // GpuTensorView: DLPack standard representation exposing shape, strides, and __dlpack__
    nb::class_<GpuTensorView>(m, "GpuTensorView")
        .def_prop_ro("shape", [](const GpuTensorView& v) {
            nb::list l;
            for (auto s : v.shape) l.append(s);
            return nb::tuple(l);
        })
        .def_prop_ro("strides", [](const GpuTensorView& v) {
            nb::list l;
            for (auto s : v.strides) l.append(s);
            return nb::tuple(l);
        })
        .def_prop_ro("dtype", [](const GpuTensorView& v) { return v.dtype; })
        .def_prop_ro("device", [](const GpuTensorView& v) { return "cuda:0"; })
        .def_prop_ro("is_cuda", [](const GpuTensorView&) { return true; })
        .def_prop_ro("data_ptr", [](const GpuTensorView& v) { return reinterpret_cast<uintptr_t>(v.data); })
        .def("get_data_ptr", [](const GpuTensorView& v) { return reinterpret_cast<uintptr_t>(v.data); })
        .def("size", [](const GpuTensorView& v) {
            nb::list l;
            for (auto s : v.shape) l.append(s);
            return nb::tuple(l);
        })
        .def("stride", [](const GpuTensorView& v) {
            nb::list l;
            for (auto s : v.strides) l.append(s);
            return nb::tuple(l);
        })
        .def("dim", [](const GpuTensorView& v) { return v.shape.size(); })
        .def("__len__", [](const GpuTensorView& v) { return v.shape.empty() ? 0 : v.shape[0]; })
        .def("__getitem__", &GpuTensorView::getitem)
        .def("__setitem__", &GpuTensorView::setitem)
        .def("zero_", &GpuTensorView::zero_)
        .def("clone", &GpuTensorView::clone)
        .def("__dlpack__", &GpuTensorView::to_dlpack, nb::arg("stream") = nb::none())
        .def("__dlpack_device__", [](const GpuTensorView& v) {
            return std::make_pair(v.device_type, v.device_id);
        })
        .def("__repr__", [](const GpuTensorView& v) {
            std::string s = "GpuTensorView(shape=[";
            for (size_t i = 0; i < v.shape.size(); ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(v.shape[i]);
            }
            s += "], strides=[";
            for (size_t i = 0; i < v.strides.size(); ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(v.strides[i]);
            }
            s += "], dtype=" + v.dtype + ", device=cuda:0)";
            return s;
        });

    nb::class_<SimContext>(m, "SimContext")
        .def(nb::init<uint32_t, uint32_t>(),
             nb::arg("num_envs") = 1,
             nb::arg("cars_per_env") = 1,
             "Initialize SimContext allocating monolithic GPU VRAM arena")
        .def_prop_ro("num_envs", &SimContext::GetNumEnvs)
        .def_prop_ro("cars_per_env", &SimContext::GetCarsPerEnv)
        .def_prop_ro("total_cars", &SimContext::GetTotalCars)
        .def_prop_ro("allocated_bytes", &SimContext::GetAllocatedBytes)
        .def_prop_ro("ball_pitch_floats", &SimContext::GetBallPitchFloats)
        .def_prop_ro("car_pitch_floats", &SimContext::GetCarPitchFloats)
        .def("get_num_envs", &SimContext::GetNumEnvs)
        .def("get_cars_per_env", &SimContext::GetCarsPerEnv)
        .def("get_total_cars", &SimContext::GetTotalCars)
        .def("get_allocated_bytes", &SimContext::GetAllocatedBytes)
        .def("get_ball_pitch_floats", &SimContext::GetBallPitchFloats)
        .def("get_car_pitch_floats", &SimContext::GetCarPitchFloats)

        // Ball State DLPack Zero-Copy View: shape [num_envs, 13], strides [1, pitch_floats]
        .def("get_ball_observations", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetBallState().pos_x;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), 13 };
            v.strides = { 1, static_cast<int64_t>(ctx.GetBallPitchFloats()) };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA strided tensor view of Ball State (13 rigid body floats) [num_envs, 13]")
        .def("get_ball_state_tensor", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetBallState().pos_x;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), 13 };
            v.strides = { 1, static_cast<int64_t>(ctx.GetBallPitchFloats()) };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Alias for get_ball_observations")
        .def("get_ball_observations_raw", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            size_t shape[2] = { ctx.GetNumEnvs(), 13 };
            int64_t strides[2] = { 1, static_cast<int64_t>(ctx.GetBallPitchFloats()) };
            return nb::ndarray<float, nb::device::cuda>(
                ctx.GetBallState().pos_x,
                2,
                shape,
                self,
                strides
            );
        }, "Get raw DLPack PyCapsule of Ball State")

        // Car State DLPack Zero-Copy View: shape [num_envs, cars_per_env, 14], strides [cars_per_env, 1, pitch_floats]
        .def("get_car_observations", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().pos_x;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()), 14 };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1, static_cast<int64_t>(ctx.GetCarPitchFloats()) };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA strided tensor view of Car State (14 floats) [num_envs, cars_per_env, 14]")
        .def("get_car_state_tensor", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().pos_x;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()), 14 };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1, static_cast<int64_t>(ctx.GetCarPitchFloats()) };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Alias for get_car_observations")
        .def("get_car_observations_raw", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            size_t shape[3] = { ctx.GetNumEnvs(), ctx.GetCarsPerEnv(), 14 };
            int64_t strides[3] = {
                static_cast<int64_t>(ctx.GetCarsPerEnv()),
                1,
                static_cast<int64_t>(ctx.GetCarPitchFloats())
            };
            return nb::ndarray<float, nb::device::cuda>(
                ctx.GetCarState().pos_x,
                3,
                shape,
                self,
                strides
            );
        }, "Get raw DLPack PyCapsule of Car State")

        // Ball Hit State Views: shape [num_envs, cars_per_env]
        .def("get_ball_hit_is_valid", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().ball_hit_is_valid;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Ball Hit Is Valid flags [num_envs, cars_per_env]")

        // Boost Pad States: shape [num_envs, 34], strides [34, 1]
        .def("get_pad_is_active", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetArenaState().pad_is_active;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), MAX_BOOST_PADS };
            v.strides = { MAX_BOOST_PADS, 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Boost Pad Active flags [num_envs, 34]")
        .def("get_pad_cooldown", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetArenaState().pad_cooldown;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), MAX_BOOST_PADS };
            v.strides = { MAX_BOOST_PADS, 1 };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Boost Pad Cooldown timers [num_envs, 34]")

        // Episode & Termination Flags (shape [num_envs])
        .def("get_is_goal", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetArenaState().is_goal;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Goal flags [num_envs]")
        .def("get_scoring_team", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetArenaState().scoring_team;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Scoring Team indices [num_envs]")
        .def("get_is_out_of_bounds", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetArenaState().is_out_of_bounds;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Out of Bounds flags [num_envs]")
        .def("get_tick_count", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetArenaState().tick_count;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Episode Tick Counts [num_envs]")
        .def("get_rewards", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetRewards();
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Episode Rewards [num_envs, cars_per_env]")
        .def("get_terminated", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetTerminated();
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Terminated flags [num_envs]")
        .def("get_truncated", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetTruncated();
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Truncated flags [num_envs]")
        .def("get_stream", [](const SimContext& ctx) {
            return reinterpret_cast<uintptr_t>(ctx.GetStream());
        }, "Get raw CUDA stream pointer (uintptr_t)")

        // Simulation Step
        .def("step", [](SimContext& ctx, uint32_t batch_size) {
            ctx.Step(batch_size, nullptr);
        }, nb::arg("batch_size") = 0, "Execute simulation step using stored controls")
        .def("step", [](SimContext& ctx, nb::object actions) {
            const float* d_actions = nullptr;
            if (nb::isinstance<GpuTensorView>(actions)) {
                d_actions = reinterpret_cast<const float*>(nb::cast<const GpuTensorView&>(actions).data);
            } else if (nb::hasattr(actions, "data_ptr")) {
                uintptr_t ptr = nb::cast<uintptr_t>(actions.attr("data_ptr")());
                d_actions = reinterpret_cast<const float*>(ptr);
            } else {
                auto arr = nb::cast<nb::ndarray<float, nb::device::cuda>>(actions);
                d_actions = arr.data();
            }
            ctx.Step(0, d_actions);
        }, nb::arg("actions"), "Execute simulation step consuming GPU actions tensor with zero host copies")
        .def("step_actions", [](SimContext& ctx, nb::object actions) {
            const float* d_actions = nullptr;
            if (nb::isinstance<GpuTensorView>(actions)) {
                d_actions = reinterpret_cast<const float*>(nb::cast<const GpuTensorView&>(actions).data);
            } else if (nb::hasattr(actions, "data_ptr")) {
                uintptr_t ptr = nb::cast<uintptr_t>(actions.attr("data_ptr")());
                d_actions = reinterpret_cast<const float*>(ptr);
            } else {
                auto arr = nb::cast<nb::ndarray<float, nb::device::cuda>>(actions);
                d_actions = arr.data();
            }
            ctx.Step(0, d_actions);
        }, nb::arg("actions"), "Execute simulation step consuming GPU actions tensor with zero host copies")

        // Selective Resets
        .def("reset_to_default", [](SimContext& ctx) {
            ctx.ResetToDefault();
        }, "Reset all environments to default kickoff states")
        .def("reset_batch", [](SimContext& ctx, nb::object indices) {
            if (indices.is_none()) {
                ctx.ResetToDefault();
                return;
            }
            if (nb::isinstance<GpuTensorView>(indices)) {
                const auto& v = nb::cast<const GpuTensorView&>(indices);
                if (v.dtype == "int64") {
                    ctx.ResetEnvironmentsIndexed(reinterpret_cast<const int64_t*>(v.data), static_cast<uint32_t>(v.shape[0]));
                } else {
                    ctx.ResetEnvironmentsIndexed(reinterpret_cast<const int32_t*>(v.data), static_cast<uint32_t>(v.shape[0]));
                }
                return;
            }
            if (nb::hasattr(indices, "data_ptr")) {
                uintptr_t ptr = nb::cast<uintptr_t>(indices.attr("data_ptr")());
                size_t count = nb::len(indices);
                std::string dt_str = nb::hasattr(indices, "dtype") ? nb::cast<std::string>(indices.attr("dtype").attr("__str__")()) : "";
                if (dt_str.find("int64") != std::string::npos || dt_str.find("long") != std::string::npos) {
                    ctx.ResetEnvironmentsIndexed(reinterpret_cast<const int64_t*>(ptr), static_cast<uint32_t>(count));
                } else {
                    ctx.ResetEnvironmentsIndexed(reinterpret_cast<const int32_t*>(ptr), static_cast<uint32_t>(count));
                }
                return;
            }
            if (nb::isinstance<nb::ndarray<int32_t, nb::device::cuda>>(indices)) {
                auto arr = nb::cast<nb::ndarray<int32_t, nb::device::cuda>>(indices);
                ctx.ResetEnvironmentsIndexed(arr.data(), static_cast<uint32_t>(arr.size()));
                return;
            }
            if (nb::isinstance<nb::ndarray<int64_t, nb::device::cuda>>(indices)) {
                auto arr = nb::cast<nb::ndarray<int64_t, nb::device::cuda>>(indices);
                ctx.ResetEnvironmentsIndexed(arr.data(), static_cast<uint32_t>(arr.size()));
                return;
            }
            if (nb::isinstance<nb::list>(indices)) {
                nb::list l = nb::cast<nb::list>(indices);
                uint32_t count = static_cast<uint32_t>(l.size());
                if (count == 0) return;
                std::vector<int32_t> host_idx(count);
                for (uint32_t i = 0; i < count; ++i) {
                    host_idx[i] = nb::cast<int32_t>(l[i]);
                }
                int32_t* d_idx = nullptr;
                cudaMalloc(&d_idx, count * sizeof(int32_t));
                cudaMemcpyAsync(d_idx, host_idx.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.GetStream());
                ctx.ResetEnvironmentsIndexed(d_idx, count);
                if (ctx.GetStream()) cudaStreamSynchronize(ctx.GetStream()); else cudaDeviceSynchronize();
                cudaFree(d_idx);
                return;
            }
        }, nb::arg("indices") = nb::none(), "Selectively reset environments on GPU asynchronously")
        .def("reset_masked", [](SimContext& ctx, nb::object mask) {
            const uint8_t* d_mask = nullptr;
            if (nb::isinstance<GpuTensorView>(mask)) {
                d_mask = reinterpret_cast<const uint8_t*>(nb::cast<const GpuTensorView&>(mask).data);
            } else if (nb::hasattr(mask, "data_ptr")) {
                uintptr_t ptr = nb::cast<uintptr_t>(mask.attr("data_ptr")());
                d_mask = reinterpret_cast<const uint8_t*>(ptr);
            } else {
                auto arr = nb::cast<nb::ndarray<uint8_t, nb::device::cuda>>(mask);
                d_mask = arr.data();
            }
            ctx.ResetEnvironmentsMasked(d_mask);
        }, nb::arg("mask"), "Reset environments using a GPU boolean/uint8 mask");

    // Asynchronous CUDA Events (for non-blocking latency/throughput benchmarking)
    nb::class_<GpuEvent>(m, "GpuEvent")
        .def(nb::init<bool>(), nb::arg("enable_timing") = true)
        .def("record", &GpuEvent::record, nb::arg("stream") = nb::none())
        .def("wait", &GpuEvent::wait, nb::arg("stream") = nb::none())
        .def("synchronize", &GpuEvent::synchronize)
        .def("query", &GpuEvent::query)
        .def("elapsed_time", &GpuEvent::elapsed_time, nb::arg("end_event"));

    m.attr("Event") = m.attr("GpuEvent");

    // Allocate zeroed GPU tensor in VRAM
    m.def("zeros", [](std::vector<int64_t> shape, const std::string& dtype) {
        int64_t total = 1;
        for (auto s : shape) total *= s;
        size_t elem_size = (dtype == "uint8") ? 1 : ((dtype == "int64") ? 8 : 4);
        size_t total_bytes = total * elem_size;

        void* d_ptr = nullptr;
        cudaMalloc(&d_ptr, total_bytes);
        cudaMemset(d_ptr, 0, total_bytes);

        std::vector<int64_t> strides(shape.size());
        int64_t st = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = st;
            st *= shape[i];
        }

        GpuTensorView v;
        v.data = d_ptr;
        v.shape = shape;
        v.strides = strides;
        v.dtype = dtype;
        v.device_type = 2;
        v.device_id = 0;
        nb::capsule cleanup(d_ptr, [](void* p) noexcept {
            if (p) cudaFree(p);
        });
        v.owner = cleanup;
        return v;
    }, nb::arg("shape"), nb::arg("dtype") = "float32", "Allocate a zeroed GPU tensor in VRAM");

    m.def("get_vram_info", []() {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        return std::make_pair(free_b, total_b);
    }, "Return tuple of (free_vram_bytes, total_vram_bytes)");
}
