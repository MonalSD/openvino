// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

template <typename T>
OutputVector translate_direct_reduce_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Any", "All", "EuclideanNorm", "Max", "Mean", "Min", "Sum"});
    auto input = node.get_input(0);
    auto axis = node.get_input(1);
    auto keep_dims = node.get_attribute<bool>("keep_dims", false);
    auto reduce_op = make_shared<T>(input, axis, keep_dims);
    set_node_name(node.get_name(), reduce_op);
    return {reduce_op};
}

OutputVector translate_prod_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Prod"}, true);
    auto input = node.get_input(0);
    auto axis = node.get_input(1);
    auto keep_dims = node.get_attribute<bool>("keep_dims", false);
    auto complex_type_input = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    if (complex_type_input) {
        input = complex_type_input->input_value(0);
        auto output_shape = input->get_shape();
        output_shape.push_back(2);  // Adding auxillary dimension
        auto output_tensor = make_shared<Tensor>(element::f32, output_shape);
        output_tensor->set_attribute("ComplexTypeMark", complex_type_input);
        auto prod_op = make_shared<v1::ReduceProd>(input, axis, keep_dims);
        auto complex_result = make_shared<ComplexTypeMark>(prod_op, complex_type_input->get_complex_part_type());
        set_node_name(node.get_name(), prod_op);
        return {complex_result};
    } else {
        auto prod_op = make_shared<v1::ReduceProd>(input, axis, keep_dims);
        set_node_name(node.get_name(), prod_op);
        return {prod_op};
    }
}

template OutputVector translate_direct_reduce_op<v1::ReduceLogicalOr>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceLogicalAnd>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceMax>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceMean>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceMin>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceProd>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceSum>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v4::ReduceL2>(const NodeContext& node);
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
