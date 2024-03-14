// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_scatter_nd_op(const NodeContext& node) {
    default_op_checks(node, 3, {"ScatterNd", "SCATTER_ND"}, true);
    auto input_indices = node.get_input(0);
    auto updates = node.get_input(1);
    auto shape = node.get_input(2);
    auto complex_type_mark_updates = as_type_ptr<ComplexTypeMark>(updates.get_node_shared_ptr());

    if (complex_type_mark_updates) {
        updates = complex_type_mark_updates->input_value(0);
    }
    // Add two auxiliary dimensions to the shape tensor
    Shape shape_dims = shape->get_shape();
    shape_dims.insert(shape_dims.begin(), 1);
    shape_dims.insert(shape_dims.begin(), 1);
    auto updated_shape = make_shared<v3::Constant>(shape_dims, shape->get_data_type());

    auto input_data = create_same_type_const<int32_t>(updates, vector<int32_t>{0}, Shape{1});
    auto broadcast = make_shared<v3::Broadcast>(input_data, updated_shape);

    auto scatter_nd = make_shared<v3::ScatterNDUpdate>(broadcast, input_indices, updates);
    set_node_name(node.get_name(), scatter_nd);

    if (complex_type_inputs) {
        auto complex_scatter_nd =
            make_shared<ComplexTypeMark>(scatter_nd, complex_type_mark_updates->get_complex_part_type());
        return {complex_scatter_nd};
    }

    return {scatter_nd};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
