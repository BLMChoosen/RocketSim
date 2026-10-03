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

// Zero-copy DLPack-compliant GPU Tensor View structure
struct GpuTensorView {
    void* data = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::string dtype = "float32";
    int device_type = 2; // kDLCUDA
    int device_id = 0;
    nb::object owner;

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
        .def_prop_ro("data_ptr", [](const GpuTensorView& v) { return reinterpret_cast<uintptr_t>(v.data); })
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
}
