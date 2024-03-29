// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/max_pool.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

struct MaxPoolParams {
    template <class Input_t, class Indices_t>
    MaxPoolParams(const Shape& input_shape,
                  const element::Type& input_type,
                  const std::vector<Input_t>& input_data,
                  const std::vector<Input_t>& expected_values,
                  const element::Type& indices_type,
                  const std::vector<Indices_t>& expected_indices,
                  const Strides& strides,
                  const Strides& dilations,
                  const Shape& pads_begin,
                  const Shape& pads_end,
                  const Shape& kernel,
                  const op::RoundingType& rounding_type,
                  const op::PadType pad_type = op::PadType::EXPLICIT,
                  const int64_t axis = 0)
        : m_input_shape(input_shape),
          m_input_type(input_type),
          m_indices_type(indices_type),
          m_input_data(CreateTensor(input_type, input_data)),
          m_expected_values(CreateTensor(input_type, expected_values)),
          m_expected_indices(CreateTensor(indices_type, expected_indices)),
          m_strides(strides),
          m_dilations(dilations),
          m_pads_begin(pads_begin),
          m_pads_end(pads_end),
          m_kernel(kernel),
          m_rounding_type(rounding_type),
          m_pad_type(pad_type),
          m_axis(axis) {}
    Shape m_input_shape;
    element::Type m_input_type;
    element::Type m_indices_type;
    ov::Tensor m_input_data;
    ov::Tensor m_expected_values;
    ov::Tensor m_expected_indices;
    Strides m_strides;
    Strides m_dilations;
    Shape m_pads_begin;
    Shape m_pads_end;
    Shape m_kernel;
    ov::op::RoundingType m_rounding_type;
    op::PadType m_pad_type;
    int64_t m_axis;
};

class ReferenceMaxPoolLayerTestV8 : public testing::TestWithParam<MaxPoolParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.m_input_data};
        refOutData = {params.m_expected_values, params.m_expected_indices};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<MaxPoolParams>& obj) {
        const auto p = obj.param;
        std::ostringstream result;
        result << p.m_input_shape.size() - 2 << "D/";
        result << "input_shape=" << p.m_input_shape << ";";
        result << "input_type=" << p.m_input_type << ";";
        result << "indices_type=" << p.m_indices_type << ";";
        result << "strides=" << p.m_strides << ";";
        result << "dilations=" << p.m_dilations << ";";
        result << "pads_begin=" << p.m_pads_begin << ";";
        result << "pads_end=" << p.m_pads_end << ";";
        result << "kernel=" << p.m_kernel << ";";
        result << "rounding_type=" << p.m_rounding_type << ";";
        result << "pad_type=" << p.m_pad_type << ";";
        result << "axis=" << p.m_axis;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const MaxPoolParams& params) {
        const auto in = std::make_shared<op::v0::Parameter>(params.m_input_type, params.m_input_shape);
        const auto max_pool = std::make_shared<op::v8::MaxPool>(in,
                                                                params.m_strides,
                                                                params.m_dilations,
                                                                params.m_pads_begin,
                                                                params.m_pads_end,
                                                                params.m_kernel,
                                                                params.m_rounding_type,
                                                                params.m_pad_type,
                                                                params.m_indices_type,
                                                                params.m_axis);
        return std::make_shared<ov::Model>(max_pool, ParameterVector{in});
    }
};

TEST_P(ReferenceMaxPoolLayerTestV8, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPool_With_Hardcoded_Refs,
    ReferenceMaxPoolLayerTestV8,
    ::testing::Values(
        MaxPoolParams(Shape{1, 1, 9},
                      element::i32,
                      std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<int32_t>{2, 3, 4, 5, 6, 7, 8, 9},
                      element::i64,
                      std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8},
                      Strides{1},
                      Strides{1},
                      Shape{},
                      Shape{},
                      Shape{2},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 9},
                      element::i32,
                      std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<int32_t>{2, 4, 6, 8},
                      element::i64,
                      std::vector<int64_t>{1, 3, 5, 7},
                      Strides{2},
                      Strides{1},
                      Shape{},
                      Shape{},
                      Shape{2},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 9},
                      element::i32,
                      std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<int32_t>{1, 3, 5, 7, 9},
                      element::i64,
                      std::vector<int64_t>{0, 2, 4, 6, 8},
                      Strides{2},
                      Strides{1},
                      Shape{},
                      Shape{},
                      Shape{2},
                      op::RoundingType::FLOOR,
                      op::PadType::SAME_LOWER),
        MaxPoolParams(Shape{1, 1, 9},
                      element::i32,
                      std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<int32_t>{2, 4, 6, 8, 9},
                      element::i64,
                      std::vector<int64_t>{1, 3, 5, 7, 8},
                      Strides{2},
                      Strides{1},
                      Shape{},
                      Shape{},
                      Shape{2},
                      op::RoundingType::FLOOR,
                      op::PadType::SAME_UPPER),
        MaxPoolParams(Shape{1, 1, 9},
                      element::i32,
                      std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<int32_t>{3, 5, 7, 9},
                      element::i32,
                      std::vector<int32_t>{2, 4, 6, 8},
                      Strides{2},
                      Strides{2},
                      Shape{},
                      Shape{},
                      Shape{2},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 2, 4},
                      element::f32,
                      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 0.0f, -3.14f, -2.71f, 5.0f},
                      std::vector<float>{3.0f, 4.0f, 0.0f, 5.0f},
                      element::i32,
                      std::vector<int32_t>{2, 3, 4, 7},
                      Strides{1},
                      Strides{1},
                      Shape{},
                      Shape{},
                      Shape{3},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 2, 4},
                      element::f32,
                      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 0.0f, -3.14f, -2.71f, 5.0f},
                      std::vector<float>{3.0f, 4.0f, 0.0f, 5.0f},
                      element::i32,
                      std::vector<int32_t>{2, 3, 0, 3},
                      Strides{1},
                      Strides{1},
                      Shape{},
                      Shape{},
                      Shape{3},
                      op::RoundingType::FLOOR,
                      op::PadType::EXPLICIT,
                      2),
        MaxPoolParams(Shape{1, 1, 9},
                      element::i32,
                      std::vector<int32_t>{1, 9, 3, 8, 5, 2, 6, 4, 7},
                      std::vector<int32_t>{1, 9, 6, 7},
                      element::i32,
                      std::vector<int32_t>{0, 1, 6, 8},
                      Strides{3},
                      Strides{1},
                      Shape{2},
                      Shape{2},
                      Shape{3},
                      op::RoundingType::FLOOR),
        /*************************************************/
        /***************** 2D test cases *****************/
        /*************************************************/
        MaxPoolParams(Shape{1, 1, 3, 3},
                      element::i32,
                      std::vector<int32_t>{3, 9, 10, 5, 7, 2, 18, 8, -2},
                      std::vector<int32_t>{9, 10, 18, 8},
                      element::i32,
                      std::vector<int32_t>{1, 2, 6, 7},
                      Strides{1, 1},
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 4, 4},  // simple 4x4 input test
                      element::i32,
                      std::vector<int32_t>{8, -9, 1, -16, -14, 15, -17, 19, -13, 3, 10, 17, 16, -11, -15, 20},
                      std::vector<int32_t>{15, 15, 19, 15, 15, 19, 16, 10, 20},
                      element::i32,
                      std::vector<int32_t>{5, 5, 7, 5, 5, 7, 12, 10, 15},
                      Strides{1, 1},
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 4, 4},
                      element::i32,
                      std::vector<int32_t>{8, -9, 1, -16, -14, 15, -17, 19, -13, 3, 10, 17, 16, -11, -15, 20},
                      std::vector<int32_t>{15, 19, 16, 20},
                      element::i32,
                      std::vector<int32_t>{5, 7, 12, 15},
                      Strides{2, 2},  // strides: 2x2
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 4, 4},
                      element::i32,
                      std::vector<int32_t>{8, -9, 1, -16, -14, 15, -17, 19, -13, 3, 10, 17, 16, -11, -15, 20},
                      std::vector<int32_t>{10, 17, 16, 20},
                      element::i32,
                      std::vector<int32_t>{10, 11, 12, 15},
                      Strides{1, 1},
                      Strides{2, 2},  // dilations: 2x2
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 4, 4},
                      element::i32,
                      std::vector<int32_t>{8, -9, 1, -16, -14, 15, -17, 19, -13, 3, 10, 17, 16, -11, -15, 20},
                      std::vector<int32_t>{15, 19, 16, 20},
                      element::i32,
                      std::vector<int32_t>{5, 7, 12, 15},
                      Strides{1, 1},
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{3, 3},
                      op::RoundingType::FLOOR),  // kernel: 3x3
        MaxPoolParams(Shape{1, 1, 5, 5},
                      element::i32,
                      std::vector<int32_t>{0,  -2, 24, 13, 7,  -5, -4, 4, 21, -18, 81, 20, -15,
                                           37, 23, 41, 18, 42, 8,  32, 9, 57, 58,  29, 3},
                      std::vector<int32_t>{0, 21, 81, 37},
                      element::i32,
                      std::vector<int32_t>{0, 8, 10, 13},
                      Strides{2, 3},  // strides: 2x3
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 5, 5},
                      element::i32,
                      std::vector<int32_t>{0,  -2, 24, 13, 7,  -5, -4, 4, 21, -18, 81, 20, -15,
                                           37, 23, 41, 18, 42, 8,  32, 9, 57, 58,  29, 3},
                      std::vector<int32_t>{0, 24, 57, 58},
                      element::i32,
                      std::vector<int32_t>{0, 2, 21, 22},
                      Strides{3, 2},  // strides: 3x2
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 5, 5},
                      element::i32,
                      std::vector<int32_t>{0,  -2, 24, 13, 7,  -5, -4, 4, 21, -18, 81, 20, -15,
                                           37, 23, 41, 18, 42, 8,  32, 9, 57, 58,  29, 3},
                      std::vector<int32_t>{81, 24, 81, 58},
                      element::i32,
                      std::vector<int32_t>{10, 2, 10, 22},
                      Strides{2, 2},  // strides: 2x2
                      Strides{2, 2},  // dilations: 2x2
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 5, 5},
                      element::i32,
                      std::vector<int32_t>{0,  -2, 24, 13, 7,  -5, -4, 4, 21, -18, 81, 20, -15,
                                           37, 23, 41, 18, 42, 8,  32, 9, 57, 58,  29, 3},
                      std::vector<int32_t>{0, 24, 21, 81, 42, 37, 57, 58, 32},
                      element::i32,
                      std::vector<int32_t>{0, 2, 8, 10, 17, 13, 21, 22, 19},
                      Strides{2, 2},  // strides: 2x2
                      Strides{1, 1},
                      Shape{1, 1},  // pads_begin: 1x1
                      Shape{1, 1},  // pads_end: 1x1
                      Shape{3, 3},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 5, 5},
                      element::i32,
                      std::vector<int32_t>{0,  -2, 24, 13, 7,  -5, -4, 4, 21, -18, 81, 20, -15,
                                           37, 23, 41, 18, 42, 8,  32, 9, 57, 58,  29, 3},
                      std::vector<int32_t>{81, 37, 81, 58, 58, 58},
                      element::i32,
                      std::vector<int32_t>{10, 13, 10, 22, 22, 22},
                      Strides{2, 2},  // strides: 2x2
                      Strides{1, 1},
                      Shape{},
                      Shape{2, 1},  // pads_end: 2x1
                      Shape{3, 3},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 2, 3, 3},
                      element::i64,
                      std::vector<int64_t>{0, -2, 24, 13, 7, -5, -4, 4, 21, -18, 81, 20, -15, 37, 23, 41, 18, 42},
                      std::vector<int64_t>{13, 24, 13, 21, 81, 81, 41, 42},
                      element::i64,
                      std::vector<int64_t>{3, 2, 3, 8, 1, 1, 6, 8},
                      Strides{1, 1},
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      op::RoundingType::FLOOR,
                      op::PadType::EXPLICIT,
                      2),  // axis: 2
        MaxPoolParams(Shape{1, 1, 2, 2},
                      element::i32,
                      std::vector<int32_t>{1, 2, 3, 4},
                      std::vector<int32_t>{1, 2, 3, 4},
                      element::i32,
                      std::vector<int32_t>{0, 1, 2, 3},
                      Strides{1, 1},
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{1, 1},
                      op::RoundingType::FLOOR),  // kernel: 1x1
        /*************************************************/
        /***************** 3D test cases *****************/
        /*************************************************/
        MaxPoolParams(Shape{1, 1, 3, 3, 3},
                      element::i32,
                      std::vector<int32_t>{0,  1,  2,  3,  4,  5, 6, 7, 8,   10,  20, 30, 40, -20,
                                           60, 70, 80, 50, 50, 1, 2, 3, -15, -10, 50, 30, 81},
                      std::vector<int32_t>{40, 60, 80, 80, 50, 60, 80, 81},
                      element::i32,
                      std::vector<int32_t>{12, 14, 16, 16, 18, 14, 16, 26},
                      Strides{1, 1, 1},
                      Strides{1, 1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2, 2},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 3, 3, 3},
                      element::i32,
                      std::vector<int32_t>{0,  1,  2,  3,  4,  5, 6, 7, 8,   10,  20, 30, 40, -20,
                                           60, 70, 80, 50, 50, 1, 2, 3, -15, -10, 50, 30, 81},
                      std::vector<int32_t>{-20, -20, -20, -20, -20, -20, -20, -20},
                      element::i32,
                      std::vector<int32_t>{13, 13, 13, 13, 13, 13, 13, 13},
                      Strides{2, 2, 2},
                      Strides{2, 2, 2},
                      Shape{1, 1, 1},
                      Shape{1, 1, 1},
                      Shape{2, 2, 2},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 3, 3, 3},
                      element::i32,
                      std::vector<int32_t>{0,  1,  2,  3,  4,  5, 6, 7, 8,   10,  20, 30, 40, -20,
                                           60, 70, 80, 50, 50, 1, 2, 3, -15, -10, 50, 30, 81},
                      std::vector<int32_t>{8, 80, 81},
                      element::i32,
                      std::vector<int32_t>{8, 16, 26},
                      Strides{1, 1, 1},
                      Strides{1, 1, 1},
                      Shape{},
                      Shape{},
                      Shape{1, 3, 3},
                      op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 3, 3, 3},
                      element::i32,
                      std::vector<int32_t>{0,  1,  2,  3,  4,  5, 6, 7, 8,   10,  20, 30, 40, -20,
                                           60, 70, 80, 50, 50, 1, 2, 3, -15, -10, 50, 30, 81},
                      std::vector<int32_t>{4, 5, 7, 8, 40, 60, 80, 80, 50, 2, 50, 81},
                      element::i32,
                      std::vector<int32_t>{4, 5, 7, 8, 3, 5, 7, 7, 0, 2, 6, 8},
                      Strides{1, 1, 1},
                      Strides{1, 1, 1},
                      Shape{},
                      Shape{},
                      Shape{1, 2, 2},
                      op::RoundingType::FLOOR,
                      op::PadType::EXPLICIT,
                      3)),

    ReferenceMaxPoolLayerTestV8::getTestCaseName);

class ReferenceMaxPoolLayerTestV14 : public testing::TestWithParam<MaxPoolParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.m_input_data};
        refOutData = {params.m_expected_values, params.m_expected_indices};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<MaxPoolParams>& obj) {
        const auto p = obj.param;
        std::ostringstream result;
        result << p.m_input_shape.size() - 2 << "D/";
        result << "input_shape=" << p.m_input_shape << ";";
        result << "input_type=" << p.m_input_type << ";";
        result << "indices_type=" << p.m_indices_type << ";";
        result << "strides=" << p.m_strides << ";";
        result << "dilations=" << p.m_dilations << ";";
        result << "pads_begin=" << p.m_pads_begin << ";";
        result << "pads_end=" << p.m_pads_end << ";";
        result << "kernel=" << p.m_kernel << ";";
        result << "rounding_type=" << p.m_rounding_type << ";";
        result << "pad_type=" << p.m_pad_type << ";";
        result << "axis=" << p.m_axis;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const MaxPoolParams& params) {
        const auto in = std::make_shared<op::v0::Parameter>(params.m_input_type, params.m_input_shape);
        const auto max_pool = std::make_shared<op::v14::MaxPool>(in,
                                                                 params.m_strides,
                                                                 params.m_dilations,
                                                                 params.m_pads_begin,
                                                                 params.m_pads_end,
                                                                 params.m_kernel,
                                                                 params.m_rounding_type,
                                                                 params.m_pad_type,
                                                                 params.m_indices_type,
                                                                 params.m_axis);
        return std::make_shared<ov::Model>(max_pool, ParameterVector{in});
    }
};

TEST_P(ReferenceMaxPoolLayerTestV14, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_MaxPool_With_Hardcoded_Refs,
    ReferenceMaxPoolLayerTestV14,
    ::testing::Values(
        MaxPoolParams(Shape{1, 1, 9},
                      element::i32,
                      std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<int32_t>{2, 3, 4, 5, 6, 7, 8, 9},
                      element::i64,
                      std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8},
                      Strides{1},
                      Strides{1},
                      Shape{},
                      Shape{},
                      Shape{2},
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 9},
                      element::i32,
                      std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<int32_t>{2, 4, 6, 8},
                      element::i64,
                      std::vector<int64_t>{1, 3, 5, 7},
                      Strides{2},
                      Strides{1},
                      Shape{},
                      Shape{},
                      Shape{2},
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 9},
                      element::i32,
                      std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<int32_t>{1, 3, 5, 7, 9},
                      element::i64,
                      std::vector<int64_t>{0, 2, 4, 6, 8},
                      Strides{2},
                      Strides{1},
                      Shape{},
                      Shape{},
                      Shape{2},
                      ov::op::RoundingType::FLOOR,
                      op::PadType::SAME_LOWER),
        MaxPoolParams(Shape{1, 1, 9},
                      element::i32,
                      std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<int32_t>{2, 4, 6, 8, 9},
                      element::i64,
                      std::vector<int64_t>{1, 3, 5, 7, 8},
                      Strides{2},
                      Strides{1},
                      Shape{},
                      Shape{},
                      Shape{2},
                      ov::op::RoundingType::FLOOR,
                      op::PadType::SAME_UPPER),
        MaxPoolParams(Shape{1, 1, 9},
                      element::i32,
                      std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<int32_t>{3, 5, 7, 9},
                      element::i32,
                      std::vector<int32_t>{2, 4, 6, 8},
                      Strides{2},
                      Strides{2},
                      Shape{},
                      Shape{},
                      Shape{2},
                      ov::op::RoundingType::CEIL),
        MaxPoolParams(Shape{1, 2, 4},
                      element::f32,
                      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 0.0f, -3.14f, -2.71f, 5.0f},
                      std::vector<float>{3.0f, 4.0f, 0.0f, 5.0f},
                      element::i32,
                      std::vector<int32_t>{2, 3, 4, 7},
                      Strides{1},
                      Strides{1},
                      Shape{},
                      Shape{},
                      Shape{3},
                      ov::op::RoundingType::CEIL),
        MaxPoolParams(Shape{1, 2, 4},
                      element::f32,
                      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 0.0f, -3.14f, -2.71f, 5.0f},
                      std::vector<float>{3.0f, 4.0f, 0.0f, 5.0f},
                      element::i32,
                      std::vector<int32_t>{2, 3, 0, 3},
                      Strides{1},
                      Strides{1},
                      Shape{},
                      Shape{},
                      Shape{3},
                      ov::op::RoundingType::CEIL_TORCH,
                      op::PadType::EXPLICIT,
                      2),
        MaxPoolParams(Shape{1, 1, 9},
                      element::i32,
                      std::vector<int32_t>{1, 9, 3, 8, 5, 2, 6, 4, 7},
                      std::vector<int32_t>{1, 9, 6, 7},
                      element::i32,
                      std::vector<int32_t>{0, 1, 6, 8},
                      Strides{3},
                      Strides{1},
                      Shape{2},
                      Shape{2},
                      Shape{3},
                      ov::op::RoundingType::CEIL_TORCH),
        /*************************************************/
        /***************** 2D test cases *****************/
        /*************************************************/
        MaxPoolParams(Shape{1, 1, 3, 3},
                      element::i32,
                      std::vector<int32_t>{3, 9, 10, 5, 7, 2, 18, 8, -2},
                      std::vector<int32_t>{9, 10, 18, 8},
                      element::i32,
                      std::vector<int32_t>{1, 2, 6, 7},
                      Strides{1, 1},
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 4, 4},  // simple 4x4 input test
                      element::i32,
                      std::vector<int32_t>{8, -9, 1, -16, -14, 15, -17, 19, -13, 3, 10, 17, 16, -11, -15, 20},
                      std::vector<int32_t>{15, 15, 19, 15, 15, 19, 16, 10, 20},
                      element::i32,
                      std::vector<int32_t>{5, 5, 7, 5, 5, 7, 12, 10, 15},
                      Strides{1, 1},
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 4, 4},
                      element::i32,
                      std::vector<int32_t>{8, -9, 1, -16, -14, 15, -17, 19, -13, 3, 10, 17, 16, -11, -15, 20},
                      std::vector<int32_t>{15, 19, 16, 20},
                      element::i32,
                      std::vector<int32_t>{5, 7, 12, 15},
                      Strides{2, 2},  // strides: 2x2
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 4, 4},
                      element::i32,
                      std::vector<int32_t>{8, -9, 1, -16, -14, 15, -17, 19, -13, 3, 10, 17, 16, -11, -15, 20},
                      std::vector<int32_t>{10, 17, 16, 20},
                      element::i32,
                      std::vector<int32_t>{10, 11, 12, 15},
                      Strides{1, 1},
                      Strides{2, 2},  // dilations: 2x2
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 4, 4},
                      element::i32,
                      std::vector<int32_t>{8, -9, 1, -16, -14, 15, -17, 19, -13, 3, 10, 17, 16, -11, -15, 20},
                      std::vector<int32_t>{15, 19, 16, 20},
                      element::i32,
                      std::vector<int32_t>{5, 7, 12, 15},
                      Strides{1, 1},
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{3, 3},  // kernel: 3x3
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 5, 5},
                      element::i32,
                      std::vector<int32_t>{0,  -2, 24, 13, 7,  -5, -4, 4, 21, -18, 81, 20, -15,
                                           37, 23, 41, 18, 42, 8,  32, 9, 57, 58,  29, 3},
                      std::vector<int32_t>{0, 21, 81, 37},
                      element::i32,
                      std::vector<int32_t>{0, 8, 10, 13},
                      Strides{2, 3},  // strides: 2x3
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 5, 5},
                      element::i32,
                      std::vector<int32_t>{0,  -2, 24, 13, 7,  -5, -4, 4, 21, -18, 81, 20, -15,
                                           37, 23, 41, 18, 42, 8,  32, 9, 57, 58,  29, 3},
                      std::vector<int32_t>{0, 24, 57, 58},
                      element::i32,
                      std::vector<int32_t>{0, 2, 21, 22},
                      Strides{3, 2},  // strides: 3x2
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 5, 5},
                      element::i32,
                      std::vector<int32_t>{0,  -2, 24, 13, 7,  -5, -4, 4, 21, -18, 81, 20, -15,
                                           37, 23, 41, 18, 42, 8,  32, 9, 57, 58,  29, 3},
                      std::vector<int32_t>{81, 24, 81, 58},
                      element::i32,
                      std::vector<int32_t>{10, 2, 10, 22},
                      Strides{2, 2},  // strides: 2x2
                      Strides{2, 2},  // dilations: 2x2
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 5, 5},
                      element::i32,
                      std::vector<int32_t>{0,  -2, 24, 13, 7,  -5, -4, 4, 21, -18, 81, 20, -15,
                                           37, 23, 41, 18, 42, 8,  32, 9, 57, 58,  29, 3},
                      std::vector<int32_t>{81, 37, 81, 58, 58, 58},
                      element::i32,
                      std::vector<int32_t>{10, 13, 10, 22, 22, 22},
                      Strides{2, 2},  // strides: 2x2
                      Strides{1, 1},
                      Shape{},
                      Shape{2, 1},  // pads_end: 2x1
                      Shape{3, 3},
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 5, 5},
                      element::i32,
                      std::vector<int32_t>{0,  -2, 24, 13, 7,  -5, -4, 4, 21, -18, 81, 20, -15,
                                           37, 23, 41, 18, 42, 8,  32, 9, 57, 58,  29, 3},
                      std::vector<int32_t>{0, 24, 21, 81, 42, 37, 57, 58, 32},
                      element::i32,
                      std::vector<int32_t>{0, 2, 8, 10, 17, 13, 21, 22, 19},
                      Strides{2, 2},  // strides: 2x2
                      Strides{1, 1},
                      Shape{1, 1},  // pads_begin: 1x1
                      Shape{1, 1},  // pads_end: 1x1
                      Shape{3, 3},
                      ov::op::RoundingType::CEIL),
        MaxPoolParams(Shape{1, 2, 3, 3},
                      element::i64,
                      std::vector<int64_t>{0, -2, 24, 13, 7, -5, -4, 4, 21, -18, 81, 20, -15, 37, 23, 41, 18, 42},
                      std::vector<int64_t>{13, 24, 13, 21, 81, 81, 41, 42},
                      element::i64,
                      std::vector<int64_t>{3, 2, 3, 8, 1, 1, 6, 8},
                      Strides{1, 1},
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2},
                      ov::op::RoundingType::CEIL_TORCH,
                      op::PadType::EXPLICIT,
                      2),  // axis: 2
        MaxPoolParams(Shape{1, 1, 2, 2},
                      element::i32,
                      std::vector<int32_t>{1, 2, 3, 4},
                      std::vector<int32_t>{1, 2, 3, 4},
                      element::i32,
                      std::vector<int32_t>{0, 1, 2, 3},
                      Strides{1, 1},
                      Strides{1, 1},
                      Shape{},
                      Shape{},
                      Shape{1, 1},
                      ov::op::RoundingType::CEIL_TORCH),  // kernel: 1x1
        /*************************************************/
        /***************** 3D test cases *****************/
        /*************************************************/
        MaxPoolParams(Shape{1, 1, 3, 3, 3},
                      element::i32,
                      std::vector<int32_t>{0,  1,  2,  3,  4,  5, 6, 7, 8,   10,  20, 30, 40, -20,
                                           60, 70, 80, 50, 50, 1, 2, 3, -15, -10, 50, 30, 81},
                      std::vector<int32_t>{40, 60, 80, 80, 50, 60, 80, 81},
                      element::i32,
                      std::vector<int32_t>{12, 14, 16, 16, 18, 14, 16, 26},
                      Strides{1, 1, 1},
                      Strides{1, 1, 1},
                      Shape{},
                      Shape{},
                      Shape{2, 2, 2},
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 3, 3, 3},
                      element::i32,
                      std::vector<int32_t>{0,  1,  2,  3,  4,  5, 6, 7, 8,   10,  20, 30, 40, -20,
                                           60, 70, 80, 50, 50, 1, 2, 3, -15, -10, 50, 30, 81},
                      std::vector<int32_t>{-20, -20, -20, -20, -20, -20, -20, -20},
                      element::i32,
                      std::vector<int32_t>{13, 13, 13, 13, 13, 13, 13, 13},
                      Strides{2, 2, 2},
                      Strides{2, 2, 2},
                      Shape{1, 1, 1},
                      Shape{1, 1, 1},
                      Shape{2, 2, 2},
                      ov::op::RoundingType::FLOOR),
        MaxPoolParams(Shape{1, 1, 3, 3, 3},
                      element::i32,
                      std::vector<int32_t>{0,  1,  2,  3,  4,  5, 6, 7, 8,   10,  20, 30, 40, -20,
                                           60, 70, 80, 50, 50, 1, 2, 3, -15, -10, 50, 30, 81},
                      std::vector<int32_t>{8, 80, 81},
                      element::i32,
                      std::vector<int32_t>{8, 16, 26},
                      Strides{1, 1, 1},
                      Strides{1, 1, 1},
                      Shape{},
                      Shape{},
                      Shape{1, 3, 3},
                      ov::op::RoundingType::CEIL),
        MaxPoolParams(Shape{1, 1, 3, 3, 3},
                      element::i32,
                      std::vector<int32_t>{0,  1,  2,  3,  4,  5, 6, 7, 8,   10,  20, 30, 40, -20,
                                           60, 70, 80, 50, 50, 1, 2, 3, -15, -10, 50, 30, 81},
                      std::vector<int32_t>{4, 5, 7, 8, 40, 60, 80, 80, 50, 2, 50, 81},
                      element::i32,
                      std::vector<int32_t>{4, 5, 7, 8, 3, 5, 7, 7, 0, 2, 6, 8},
                      Strides{1, 1, 1},
                      Strides{1, 1, 1},
                      Shape{},
                      Shape{},
                      Shape{1, 2, 2},
                      ov::op::RoundingType::CEIL_TORCH,
                      op::PadType::EXPLICIT,
                      3)),

    ReferenceMaxPoolLayerTestV14::getTestCaseName);