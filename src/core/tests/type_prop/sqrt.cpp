// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sqrt.hpp"

#include "unary_ops.hpp"

using Type = ::testing::Types<ov::op::v0::Sqrt>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_sqrt, UnaryOperator, Type);
