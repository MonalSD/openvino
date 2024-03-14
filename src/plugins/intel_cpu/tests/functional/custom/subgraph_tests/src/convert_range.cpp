// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/graph_util.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

/*
  This test runs the following subgraph:

              param1          param2
                |               |
              (INT)             |
                |               |
             Convert            |
                |               |
             (FP32)             |
                |               |
        (No reorder to LP)      |
                |               |
      const     |   const       |
           \    |    /          |
            \   |   /          /
              Range           /
                |            /
               /\           /
              /  \         /
    (No reorder)  \       /
           /       \     /
        Convert     MatMul
           |          |
         (INT)      Result
    const  |
      |    |
    (FP32) |
      \   /
     Gather()
        |
      MatMul
        |
      Result

  This test is needed to cover logic that allows to avoid computational error for subgraph: "[I32] -> Convert -> [F32]
  -> Range" due to precision lowering for floating point path inside "EnforceInferencePrecision" pass".
  TODO: Incorrect subgraph is generated by ONNX FE + ticket 117861.

  Also test logic to avoid enforcing F32 input of convert to lower precision in subgraph :" Range -> [F32] -> Convert ->
  [I32]" in ticket 129874
*/

using ConvertRangeSubgraphCPUTestParams = std::tuple<
    std::map<std::string, ov::element::Type>,
    std::vector<InputShape>,
    std::vector<ov::Shape>
>;

class ConvertRangeSubgraphCPUTest: public testing::WithParamInterface<ConvertRangeSubgraphCPUTestParams>,
                                 virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvertRangeSubgraphCPUTestParams> obj) {
        std::map<std::string, ov::element::Type> additionalConfig;
        std::vector<InputShape> inputShapes;
        std::vector<ov::Shape> targetShapes;
        std::tie(additionalConfig, inputShapes, targetShapes) = obj.param;
        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")";
        }
        result << "Prc=" << additionalConfig[ov::hint::inference_precision.name()];
        return result.str();
    }
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        std::vector<InputShape> inputShapes;
        std::vector<ov::Shape> targetShapes;
        std::map<std::string, ov::element::Type> additionalConfig;
        std::tie(additionalConfig, inputShapes, targetShapes) = this->GetParam();

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        init_input_shapes(inputShapes);
        ov::ParameterVector input_params;
        for (auto&& shape : inputDynamicShapes) {
            input_params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }
        auto zero = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        auto shapeof = std::make_shared<ov::op::v3::ShapeOf>(input_params[0], ov::element::i32);
        auto gather = std::make_shared<ov::op::v8::Gather>(shapeof, one, zero);
        auto convert = std::make_shared<ov::op::v0::Convert>(gather, ov::element::f32);
        auto start = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 0);
        auto step = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 1);
        auto range = std::make_shared<ov::op::v4::Range>(start, convert, step, ov::element::f32);
        range->set_friendly_name("float32_range");
        auto matmul = std::make_shared<ov::op::v0::MatMul>(range, input_params[1], false, true);
        auto result = std::make_shared<ov::op::v0::Result>(matmul);
        result->set_friendly_name("output");

        // second branch ensure convert after range doesn't use low-precision float as input
        auto convert2 = std::make_shared<ov::op::v0::Convert>(range, ov::element::i32);
        convert2->set_friendly_name("convert_to_i32");
        auto const0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 0.1f);
        auto data = std::make_shared<ov::op::v1::Broadcast>(const0, shapeof);
        auto gather2 = std::make_shared<ov::op::v1::Gather>(data, convert2, one);
        auto matmul2 = std::make_shared<ov::op::v0::MatMul>(gather2, input_params[1], false, true);
        auto result2 = std::make_shared<ov::op::v0::Result>(matmul2);
        ov::ResultVector output_results = {result, result2};

        function = std::make_shared<ov::Model>(output_results, input_params, "convert_range");
    };

    void checkResults() {
        bool found_range = false;
        bool found_convert = false;
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            if (n->get_friendly_name() == "float32_range") {
                ASSERT_EQ(n->get_input_element_type(1), ElementType::f32);
                found_range = true;
            }
            if (n->get_friendly_name() == "convert_to_i32") {
                ASSERT_EQ(n->get_input_element_type(0), ElementType::f32);
                found_convert = true;
            }
        }
        ASSERT_TRUE(found_range);
        ASSERT_TRUE(found_convert);
    }
};

namespace {
TEST_P(ConvertRangeSubgraphCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    checkResults();
}

const std::vector<std::map<std::string, ov::element::Type>> additionalConfig = {
    {{ov::hint::inference_precision.name(), ov::element::bf16}},
    {{ov::hint::inference_precision.name(), ov::element::f16}}
};

const std::vector<std::vector<InputShape>> input_shapes = {
    {
        {{1, -1}, {{1, 291}}},  // input 1
        {{1, -1}, {{1, 291}}},  // input 2
    }
};

const std::vector<std::vector<ov::Shape>> target_shapes = {
    {
        {1},
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertRangeSubgraphCPUTest,
                         ConvertRangeSubgraphCPUTest,
                         ::testing::Combine(::testing::Values(additionalConfig[0], additionalConfig[1]),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(target_shapes)),
                         ConvertRangeSubgraphCPUTest::getTestCaseName);

} // namespace
}  // namespace test
}  // namespace ov
