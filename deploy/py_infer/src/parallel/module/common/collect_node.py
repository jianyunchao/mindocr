import os
from collections import defaultdict
from ctypes import c_uint64
from multiprocessing import Manager

import numpy as np

from ...datatype import StopData, ProcessData, ProfilingData
from ...framework import ModuleBase
from ....data_process.utils import cv_utils
from ....infer import TaskType
from ....utils import safe_list_writer, log, Visualization, VisMode

RESULTS_SAVE_FILENAME = {
    TaskType.DET: 'det_results.txt',
    TaskType.CLS: 'cls_results.txt',
    TaskType.REC: 'rec_results.txt',
    TaskType.DET_REC: 'pipeline_results.txt',
    TaskType.DET_CLS_REC: 'pipeline_results.txt'
}


class CollectNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super().__init__(args, msg_queue)
        self.image_sub_remaining = defaultdict(int)
        self.image_pipeline_res = defaultdict(list)
        self.infer_size = 0
        self.image_total = Manager().Value(c_uint64, 0)
        self.task_type = args.task_type
        self.res_save_dir = args.res_save_dir

        self.save_filename = RESULTS_SAVE_FILENAME[self.task_type]

    def init_self_args(self):
        super().init_self_args()

    def stop_handle(self, input_data):
        self.image_total.value = input_data.image_total

    def single_image_save(self, image_name, image):
        if self.args.save_crop_res_dir:
            filename = os.path.join(self.args.crop_save_dir, os.path.splitext(image_name)[0])
            vis_tool = Visualization(VisMode.crop)
            box_list = [np.array(x["points"]).reshape(-1, 2) for x in self.image_pipeline_res[image_name]]
            crop_list = vis_tool(image, box_list)
            for i, crop in enumerate(crop_list):
                cv_utils.img_write(filename + '_crop_' + str(i) + '.jpg', crop)

        if self.args.save_vis_pipeline_save_dir:
            filename = os.path.join(self.args.vis_pipeline_save_dir, os.path.splitext(image_name)[0])
            vis_tool = Visualization(VisMode.bbox_text)
            box_list = [np.array(x["points"]).reshape(-1, 2) for x in self.image_pipeline_res[image_name]]
            text_list = [x["transcription"] for x in self.image_pipeline_res[image_name]]
            box_text = vis_tool(image, box_list, text_list, font_path=self.args.vis_font_path)
            cv_utils.img_write(filename + '.jpg', box_text)

        if self.args.save_vis_det_save_dir:
            filename = os.path.join(self.args.vis_det_save_dir, os.path.splitext(image_name)[0])
            vis_tool = Visualization(VisMode.bbox)
            box_list = [np.array(x).reshape(-1, 2) for x in self.image_pipeline_res[image_name]]
            box_line = vis_tool(image, box_list)
            cv_utils.img_write(filename + '.jpg', box_line)

        log.info(f"{image_name} is finished.")

    def final_text_save(self):
        save_filename = os.path.join(self.res_save_dir, self.save_filename)
        safe_list_writer(self.image_pipeline_res, save_filename)
        log.info(f'save infer result to {save_filename} successfully')

    def result_handle(self, input_data):
        if self.task_type in (TaskType.DET_REC, TaskType.DET_CLS_REC):
            for result in input_data.infer_result:
                self.image_pipeline_res[input_data.image_name].append(
                    {"transcription": result[-1], "points": result[:-1]})
            if not input_data.infer_result:
                self.image_pipeline_res[input_data.image_name] = []
        elif self.task_type == TaskType.DET:
            self.image_pipeline_res[input_data.image_name].extend(input_data.infer_result)
        elif self.task_type in (TaskType.REC, TaskType.CLS):
            self.image_pipeline_res[input_data.image_name] = input_data.infer_result
        else:
            raise NotImplementedError(f"Task type do not support.")

        if input_data.image_id in self.image_sub_remaining:
            self.image_sub_remaining[input_data.image_id] -= input_data.sub_image_size
            if not self.image_sub_remaining[input_data.image_id]:
                self.image_sub_remaining.pop(input_data.image_id)
                self.infer_size += 1
                self.single_image_save(input_data.image_name, input_data.frame)
        else:
            remaining = input_data.sub_image_total - input_data.sub_image_size
            if remaining:
                self.image_sub_remaining[input_data.image_id] = remaining
            else:
                self.infer_size += 1
                self.single_image_save(input_data.image_name, input_data.frame)

    def process(self, input_data):
        if isinstance(input_data, ProcessData):
            self.result_handle(input_data)
        elif isinstance(input_data, StopData):
            self.stop_handle(input_data)
        else:
            raise ValueError('unknown input data')

        if self.image_total.value and self.infer_size == self.image_total.value:
            self.final_text_save()
            self.send_to_next_module('stop')

    def stop(self):
        profiling_data = ProfilingData(module_name=self.module_name, instance_id=self.instance_id,
                                       device_id=self.device_id, process_cost_time=self.process_cost.value,
                                       send_cost_time=self.send_cost.value, image_total=self.image_total.value)
        self.msg_queue.put(profiling_data, block=False)
        self.is_stop = True
