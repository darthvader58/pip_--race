from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUTPUT_PATH = Path(__file__).with_name("pitwit_demo.onnx")


def main() -> None:
    weights = np.zeros((16, 2), dtype=np.float32)
    bias = np.asarray([-0.8, -2.0], dtype=np.float32)

    # Pit-risk logit.
    weights[1, 0] = -0.15  # throttle
    weights[2, 0] = 0.30  # brake
    weights[3, 0] = 0.035  # tire age
    weights[11, 0] = 0.80  # cheap stop status
    weights[14, 0] = 0.90  # degradation index
    weights[15, 0] = -1.80  # distance to pit entry

    # Tire-degradation logit.
    weights[3, 1] = 0.09
    weights[6, 1] = 0.40
    weights[8, 1] = -0.30
    weights[14, 1] = 1.30

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "weights"], ["weighted"]),
            helper.make_node("Add", ["weighted", "bias"], ["logits"]),
        ],
        "pitwit_demo_model",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 16])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 2])],
        [
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="pip-race-demo",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, OUTPUT_PATH)


if __name__ == "__main__":
    main()
