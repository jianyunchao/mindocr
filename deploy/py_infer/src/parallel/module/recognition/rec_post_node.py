from ...framework import ModuleBase
from ....infer import TextRecognizer, TaskType


class RecPostNode(ModuleBase):
    def __init__(self, args, msg_queue):
        super(RecPostNode, self).__init__(args, msg_queue)
        self.text_recognizer = None
        self.task_type = self.args.task_type

    def init_self_args(self):
        self.text_recognizer = TextRecognizer(self.args)
        self.text_recognizer.init()
        self.text_recognizer.free_model()

        super().init_self_args()

    def process(self, input_data):
        if input_data.skip:
            self.send_to_next_module(input_data)
            return

        output = input_data.output
        batch = input_data.sub_image_size

        texts = self.text_recognizer.postprocess(output, batch)

        if self.task_type == TaskType.REC:
            input_data.infer_result = texts[0]
        else:
            for result, text in zip(input_data.infer_result, texts):
                result.append(text)

        self.send_to_next_module(input_data)
