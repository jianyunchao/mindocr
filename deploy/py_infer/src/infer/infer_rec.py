import os
from collections import defaultdict
from typing import Union, List, Tuple

import math
import numpy as np

from .infer_base import InferBase
from ..data_process import gear_utils, cv_utils, build_preprocess, build_postprocess
from ..core import Model, ShapeType
from ..utils import safe_div


class TextRecognizer(InferBase):
    def __init__(self, args):
        super(TextRecognizer, self).__init__(args)

        self._hw_list = []
        self._bs_list = []

        self.model = defaultdict()
        self.shape_type = None

    def __get_shape_for_single_model(self, filename):
        model = Model(backend=self.args.backend, model_path=filename, device_id=self.args.device_id)
        shape_type, shape_info = model.get_shape_info()

        self.shape_type = shape_type
        if shape_type == ShapeType.DYNAMIC_BATCHSIZE:
            raise ValueError(
                f"Input shape don't support dynamic batch_size for single recognition model, "
                f"but got dynamic batch_size={shape_info[0]} for {filename}.")

        if shape_type == ShapeType.STATIC_SHAPE:
            n, _, h, w = shape_info
            self._hw_list = [(h, w)]
            self._bs_list = [n]
        elif shape_type == ShapeType.DYNAMIC_IMAGESIZE:
            n, _, hw_list = shape_info
            self._hw_list = list(hw_list)
            self._bs_list.append(n)
        else:  # dynamic shape
            n, _, h, w = shape_info
            self._hw_list = [(h, w)]
            self._bs_list = [n]

        self.model[n] = model

        return shape_info

    def __get_resized_hw(self, image_list):
        if self.shape_type != ShapeType.DYNAMIC_SHAPE:
            resized_hw_list = [gear_utils.get_matched_gear_hw(cv_utils.get_hw_of_img(image), self._hw_list)
                               for image in image_list]
            max_h, max_w = max(resized_hw_list, key=lambda x: x[0] * x[1])
        else:
            model_h, model_w = self._hw_list[0]
            hw_list = [cv_utils.get_hw_of_img(image) for image in image_list]
            max_h = model_h if model_h > 0 else math.ceil(safe_div(max([h for h, _ in hw_list]), 32)) * 32
            max_w = model_w if model_w > 0 else math.ceil(safe_div(max([w for _, w in hw_list]), 32)) * 32

        return max_h, max_w

    def init(self, warmup=False):
        model_path = self.args.rec_model_path

        if os.path.isfile(model_path):
            self.__get_shape_for_single_model(model_path)

        if os.path.isdir(model_path):
            chw_list = []
            for path in os.listdir(model_path):
                shape = self.__get_shape_for_single_model(os.path.join(model_path, path))
                if self.shape_type in (ShapeType.STATIC_SHAPE, ShapeType.DYNAMIC_SHAPE):
                    raise ValueError(
                        f"rec_model_dir must be a file when use static or dynamic shape for recognition model, "
                        f"but got rec_model_dir={model_path} is a dir.")
                chw_list.append(str((shape[2:])))

            if len(set(chw_list)) != 1 or len(set(self._bs_list)) != len(self._bs_list):
                raise ValueError(
                    f"Input shape must have same image_size and different batch_size when use the combination of "
                    f"dynamic batch_size and image_size for recognition model. "
                    f"Please check every model file in {model_path}.")

        self._bs_list.sort()

        self.preprocess_ops = build_preprocess(self.args.rec_config_path)
        params = {"character_dict_path": self.args.character_dict_path}
        self.postprocess_ops = build_postprocess(self.args.rec_config_path, **params)

        if warmup:
            for model in self.model.values():
                model.warmup()

    def __call__(self, image: Union[np.ndarray, List[np.ndarray]]) -> Union[str, List[str]]:
        inputs = [image] if isinstance(image, np.ndarray) else image
        split_inputs_bs, split_outputs = self.preprocess(inputs)
        split_outputs = [self.model_infer(output["image"]) for output in split_outputs]
        outputs = []
        for batch, split_output in zip(split_inputs_bs, split_outputs):
            output = self.postprocess(split_output, batch)
            outputs.extend(output)

        return outputs[0] if isinstance(image, np.ndarray) else outputs

    def preprocess(self, image: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
        num_image = len(image)
        batch_list = gear_utils.get_matched_gear_bs(num_image, self._bs_list) if self._bs_list[0] > 0 else [num_image]
        start_index = 0
        split_inputs_bs = []
        split_outputs = []
        for batch in batch_list:
            upper_bound = min(start_index + batch, num_image)
            split_input = image[start_index:upper_bound]
            split_output = self.preprocess_ops(split_input, image_shape=self.__get_resized_hw(split_input))
            split_output = gear_utils.padding_to_batch(split_output, batch)
            split_inputs_bs.append(upper_bound - start_index)
            split_outputs.append(split_output)
            start_index += batch

        return split_inputs_bs, split_outputs

    def model_infer(self, input: np.ndarray):
        bs, *_ = input.shape
        n = bs if bs in self._bs_list else -1
        return self.model[n].infer([input])

    def postprocess(self, input, batch=None) -> List[str]:
        if batch is not None:
            input = input[:batch, ...] if isinstance(input, np.ndarray) else [x[:batch, ...] for x in input]
        return self.postprocess_ops(input)
