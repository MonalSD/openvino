// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/select.hpp"

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_select_base_op(const NodeContext& node,
                                      const Output<Node>& condition,
                                      const Output<Node>& x,
                                      const Output<Node>& y) {
    auto complex_type_mark_x = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    auto complex_type_mark_y = as_type_ptr<ComplexTypeMark>(y.get_node_shared_ptr());

    if (complex_type_mark_x && complex_type_mark_y) {
        auto const_1 = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
        auto cond_shape = make_shared<v3::ShapeOf>(condition, element::i32);
        auto new_cond_shape = make_shared<v0::Concat>(OutputVector{cond_shape, const_1}, 0);
        auto prep_cond = make_shared<v1::Reshape>(condition, new_cond_shape, false)->output(0);
        // at this point all inputs are NumPy broadcastable

        auto complex_x = complex_type_mark_x->input_value(0);
        auto complex_y = complex_type_mark_y->input_value(0);

        auto select = make_shared<v1::Select>(prep_cond, complex_x, complex_y);
        set_node_name(node.get_name(), select);

        auto complex_result =
            make_shared<ComplexTypeMark>(select->output(0), complex_type_mark_x->get_complex_part_type());
        return {complex_result};
    }
    // at this point all inputs are NumPy broadcastable
    auto select = make_shared<v1::Select>(condition, x, y);
    set_node_name(node.get_name(), select);
    return {select};
}

OutputVector translate_select_v2_op(const NodeContext& node) {
    // according to the TensorFlow documentation. See in the code:
    // https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/lite/kernels/select.cc#L188-L211
    //
    // SelectV2 op selects values of 'x' if the corresponding value of 'condition'
    // is true or the value of 'y' if false. There are valid condition input sizes:
    // 1. Either the same shape (in which case the select is elementwise), or
    // 2. Broadcastable shapes between 'condition', 'x' and 'y'.
    default_op_checks(node, 3, {"SelectV2"});
    // no preparation for inputs are needed
    // inputs are already NumPy broadcastable
    return translate_select_base_op(node, node.get_input(0), node.get_input(1), node.get_input(2));
}

OutputVector translate_select_op(const NodeContext& node) {
    // according to the TensorFlow documentation. See in the code:
    // https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/lite/kernels/select.cc#L188-L211
    //
    // Select op selects values of 'x' if the corresponding value of 'condition' is
    // true or the value of 'y' if false. There are valid condition input sizes:
    // 1. Either the same shape (in which case the select is elementwise), or
    // 2. condition must be Rank 1 and match over the first dimension, or
    // 3. condition is scalar
    default_op_checks(node, 3, {"Select"}, true);
    auto condition = node.get_input(0);
    auto x = node.get_input(1);
    auto y = node.get_input(2);
    auto complex_type_mark_x = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    auto complex_type_mark_y = as_type_ptr<ComplexTypeMark>(y.get_node_shared_ptr());
    auto x_rank = compute_subgraph_scalar_rank(x, element::i32);

    if (complex_type_mark_x || complex_type_mark_y) {
        x = complex_type_mark_x->input_value(0);
        y = complex_type_mark_y->input_value(0);
        auto cond_shape = make_shared<v3::ShapeOf>(x, element::i32);
        auto const_1 = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
        auto new_cond_shape = make_shared<v0::Concat>(OutputVector{const_1, cond_shape}, 0);

        auto prep_cond = make_shared<v1::Reshape>(condition, new_cond_shape, false)->output(0);
        auto const_0 = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
        prep_cond = make_shared<v0::Squeeze>(prep_cond, const_0);

        auto select_real = translate_select_base_op(node, prep_cond, x, y);
        auto select_imag = translate_select_base_op(node, prep_cond, x, y);

        auto concat_result = make_shared<v0::Concat>(OutputVector{select_real, select_imag}, -1);

        set_node_name(node.get_name(), concat_result);

        auto complex_result =
            make_shared<ComplexTypeMark>(concat_result->output(0), complex_type_mark_x->get_complex_part_type());
        return {complex_result};
    }

    auto cond_shape = make_shared<v3::ShapeOf>(x, element::i32);
    auto const_1 = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    auto new_cond_shape = make_shared<v0::Concat>(OutputVector{const_1, cond_shape}, 0);

    auto prep_cond = make_shared<v1::Reshape>(condition, new_cond_shape, false)->output(0);
    auto const_0 = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    prep_cond = make_shared<v0::Squeeze>(prep_cond, const_0);

    return translate_select_base_op(node, prep_cond, x, y);
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
