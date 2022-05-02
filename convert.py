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
        self.model = build_ssd("convert", 300, num_classes)
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
        input_names = ["input_0"]
        output_names = ["loc", "conf"]
        torch.onnx.export(
            self.model,
            (self.dummy_input),
            output_path,
            input_names=input_names,
            output_names=output_names,
            verbose=verbose,
            opset_version=version,
            export_params=True,
            do_constant_folding=True,
        )

        if check and os.path.exists(output_path):
            print("\n=========         CHECKING!        =========\n")
            onnx_model = onnx.load(output_path)
            if simplify:
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, "Simplified ONNX model could not be validated"
                onnx.save(onnx_model, output_path)
            print(onnx.helper.printable_graph(onnx_model.graph))

        print("\n=========END! TRANS TORCH TO ONNX!=========\n")


if __name__ == "__main__":
    trans = TransModel(weight_path="./pretrain.pth")
    trans._to_onnx()
