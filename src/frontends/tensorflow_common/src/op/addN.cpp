// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/concat.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_add_n_op(const NodeContext& node) {
    default_op_checks(node, 1, {"AddN", "ADD_N"},true);
    int num_size = static_cast<int>(node.get_input_size());
    auto inputs = node.get_input(0);
    auto complex_type_mark_inputs = as_type_ptr<ComplexTypeMark>(inputs.get_node_shared_ptr());
    if (complex_type_mark_inputs) {
        inputs = complex_type_mark_inputs->input_value(0);
        auto gather_index_real = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto gather_index_imag = make_shared<v0::Constant>(element::i32, Shape{}, 1);

        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);

        auto inputs_real = make_shared<v8::Gather>(inputs, gather_index_real, minus_one)->output(0);
        auto inputs_imag = make_shared<v8::Gather>(inputs, gather_index_imag, minus_one)->output(0);

        Output<Node> result_real = inputs_real;
        Output<Node> result_imag = inputs_imag;
        for (int ind = 1; ind < num_size; ++ind) {
            auto complex_type_mark_input = as_type_ptr<ComplexTypeMark>(node.get_input(ind).get_node_shared_ptr());
            auto input_real = make_shared<v8::Gather>(complex_type_mark_input->input_value(0), gather_index_real, minus_one)->output(0);
            auto input_imag = make_shared<v8::Gather>(complex_type_mark_input->input_value(0), gather_index_imag, minus_one)->output(0);
            result_real = make_shared<v1::Add>(result_real, input_real);
            result_imag = make_shared<v1::Add>(result_imag, input_imag);
        }
        auto concat_result = make_shared<v0::Concat>(OutputVector{result_real, result_imag}, -1);
        set_node_name(node.get_name(), concat_result);
        auto complex_result = make_shared<ComplexTypeMark>(concat_result->output(0), complex_type_mark_inputs);
        return {complex_result};

    }
    
    Output<Node> result = node.get_input(0);
    for (int ind = 1; ind < num_size; ++ind) {
        result = make_shared<v1::Add>(result, node.get_input(ind));
    }
    set_node_name(node.get_name(), result.get_node_shared_ptr());
    return {result};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
