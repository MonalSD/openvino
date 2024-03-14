// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/iteration_chaining.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> configs = {{{ov::hint::inference_precision.name(), ov::element::f32}}};

const std::vector<ov::AnyMap> HeteroConfigs = {
    {{ov::hint::inference_precision.name(), ov::element::f32}, {ov::device::priorities(ov::test::utils::DEVICE_CPU)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVIterationChaining,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(configs)),
                         OVIterationChaining::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVIterationChaining,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(HeteroConfigs)),
                         OVIterationChaining::getTestCaseName);
}  // namespace
