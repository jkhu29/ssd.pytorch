import os
import sys

import torch
import onnx
import onnxsim

from ssd import build_ssd
from data import VOC_CLASSES


class TransModel(object):
    def __init__(self, weight_path=None, device="cuda") -> None:
        num_classes = len(VOC_CLASSES) + 1  # +1 for background
        self.model = build_ssd("test", 300, num_classes)
        self.model = self.model.to(device)
        print("loading weight file from : {}".format(weight_path))
        ckpt = torch.load(weight_path)
        self.model.load_state_dict(ckpt)
        print("loading weight file is done")
        self.model.eval()
        self.dummy_input = torch.randn((1, 3, 300, 300)).requires_grad_(False).to(device)
        print("\n=========BEGIN TO TRANS TORCH-MODEL=========\n")

    def _to_script(self):
        traced_script_module = torch.jit.trace(self.model, self.dummy_input)
        traced_script_module.save("yolo_script.pth")
        print("\n========END! TRANS TORCH TO SCRIPT!========\n")

    def _to_onnx(
        self, 
        version: int = 11, 
        verbose: bool = False, 
        check: bool = True, 
        simplify: bool = True,
        output_path: str = "ssd.onnx"
    ):
        input_names = ["input"]
        output_names = ["output"]
        torch.onnx.export(
            self.model,
            (self.dummy_input),
            output_path,
            input_names=input_names,
            output_names=output_names,
            verbose=verbose,
            opset_version=version,
            # export_params=True,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )

        if check and os.path.exists(output_path):
            print("\n=========         CHECKING!        =========\n")
            onnx_model = onnx.load(output_path)

            if simplify:
                import onnxoptimizer
                optimizers_list = [
                    'eliminate_deadend',
                    'eliminate_nop_dropout',
                    'eliminate_nop_cast',
                    'eliminate_nop_monotone_argmax', 'eliminate_nop_pad',
                    'extract_constant_to_initializer', 'eliminate_unused_initializer',
                    'eliminate_nop_transpose',
                    'eliminate_nop_flatten', 'eliminate_identity',
                    'fuse_add_bias_into_conv',
                    'fuse_consecutive_concats',
                    'fuse_consecutive_log_softmax',
                    'fuse_consecutive_reduce_unsqueeze', 'fuse_consecutive_squeezes',
                    'fuse_consecutive_transposes', 'fuse_matmul_add_bias_into_gemm',
                    'fuse_pad_into_conv', 'fuse_transpose_into_gemm', 'eliminate_duplicate_initializer',
                    'fuse_bn_into_conv',
                ]
                onnx_model, _ = onnxsim.simplify(onnx_model)
                onnx_model = onnxoptimizer.optimize(onnx_model, optimizers_list, fixed_point=True)
                onnx.checker.check_model(onnx_model)
                onnx.save(onnx_model, output_path)

            print(onnx.helper.printable_graph(onnx_model.graph))

        print("\n=========END! TRANS TORCH TO ONNX!=========\n")
    
    def _to_paddle_lite(self): 
        from x2paddle.convert import onnx2paddle

        onnx2paddle(
            "ssd.onnx", "./paddle_lite",
            convert_to_lite=True,
            lite_valid_places="arm",
            lite_model_type="naive_buffer",
        )


if __name__ == "__main__":
    trans = TransModel(weight_path="./pretrain.pth")
    trans._to_paddle_lite()
