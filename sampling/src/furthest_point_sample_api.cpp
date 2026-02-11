#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "furthest_point_sample.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper, "furthest_point_sampling_wrapper");
}
