[English](../../en/inference/inference_tutorial_en.md) | 中文

## 推理 - 使用教程

### 1. 简介

MindOCR的推理支持Ascend310/Ascend310P设备，采用[MindSpore Lite](https://www.mindspore.cn/lite)和[ACL](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_000004.html)两种推理后端，
集成了文本检测、角度分类和文字识别模块，实现了端到端的OCR推理过程，并采用流水并行化方式优化推理性能。


### 2. 运行环境


请参考[运行环境准备](./environment_cn.md)，配置MindOCR的推理运行环境，注意结合模型的支持情况来选择ACL/Lite环境。

### 3. 模型转换

MindOCR除了支持自身训练端导出模型的推理外，还支持第三方模型的推理，列表见[MindOCR模型支持列表](./models_list_cn.md)和[第三方模型支持列表](./models_list_thirdparty_cn.md)。

请参考[模型转换教程](./convert_tutorial_cn.md)，将其转换为MindOCR推理支持的模型格式。

### 4. 推理

进入到MindOCR推理侧目录下：`cd deploy/py_infer`.

#### 4.1 命令示例

- 检测+分类+识别

  ```shell
  python infer.py \
  	--input_images_dir=/path/to/images \
  	--backend=lite \
  	--det_model_path=/path/to/mindir/dbnet_resnet50.mindir \
  	--det_model_name=en_ms_det_dbnet_resnet50 \
  	--cls_model_path=/path/to/mindir/cls_mv3.mindir \
  	--cls_model_path=ch_pp_mobile_cls_v2.0 \
  	--rec_model_path=/path/to/mindir/crnn_resnet34.mindir \
  	--rec_model_name=en_ms_rec_crnn_resnet34 \
  	--res_save_dir=det_cls_rec
  ```

  结果保存在det_cls_rec/pipeline_results.txt，格式如下：

  ```
  img_478.jpg	[{"transcription": "spa", "points": [[1114, 35], [1200, 0], [1234, 52], [1148, 97]]}, {...}]
  ```

- 检测+识别

  不传入方向分类相关的参数，就会跳过方向分类流程，只执行检测+识别

  ```shell
  python infer.py \
  	--input_images_dir=/path/to/images \
  	--backend=lite \
  	--det_model_path=/path/to/mindir/dbnet_resnet50.mindir \
  	--det_model_name=en_ms_det_dbnet_resnet50 \
  	--rec_model_path=/path/to/mindir/crnn_resnet34.mindir \
  	--rec_model_name=en_ms_rec_crnn_resnet34 \
  	--res_save_dir=det_rec
  ```

  结果保存在det_rec/pipeline_results.txt，格式如下：

  ```
  img_478.jpg	[{"transcription": "spa", "points": [[1114, 35], [1200, 0], [1234, 52], [1148, 97]]}, {...}]
  ```

- 检测

  可以单独运行文本检测

  ```shell
  python infer.py \
  	--input_images_dir=/path/to/images \
  	--backend=lite \
  	--det_model_path=/path/to/mindir/dbnet_resnet50.mindir \
  	--det_model_name=en_ms_det_dbnet_resnet50 \
  	--res_save_dir=det
  ```

  结果保存在det/det_results.txt，格式如下：

  ```
  img_478.jpg    [[[1114, 35], [1200, 0], [1234, 52], [1148, 97]], [...]]]
  ```

- 分类

  可以单独运行文本方向分类

  ```shell
  python infer.py \
  	--input_images_dir=/path/to/images \
  	--backend=lite \
  	--cls_model_path=/path/to/mindir/cls_mv3.mindir \
  	--cls_model_path=ch_pp_mobile_cls_v2.0 \
  	--res_save_dir=cls
  ```

  结果保存在cls/cls_results.txt，格式如下：

  ```
  word_867.png   ["180", 0.5176]
  word_1679.png  ["180", 0.6226]
  word_1189.png  ["0", 0.9360]
  ```

- 识别

  可以单独运行文字识别

  ```shell
  python infer.py \
  	--input_images_dir=/path/to/images \
  	--backend=lite \
  	--rec_model_path=/path/to/mindir/crnn_resnet34.mindir \
  	--rec_model_name=en_ms_rec_crnn_resnet34 \
  	--res_save_dir=rec
  ```

  结果保存在rec/rec_results.txt，格式如下：

  ```
  word_421.png   "under"
  word_1657.png  "candy"
  word_1814.png  "cathay"
  ```

#### 4.2 详细推理参数解释

- 基本设置


| 参数名称          | 类型 | 默认值   | 含义                         |
|:-----------------|:----|:-------|:-----------------------------|
| input_images_dir | str | 无      | 单张图像或者图片文件夹           |
| device           | str | Ascend | 推理设备名称，支持：Ascend       |
| device_id        | int | 0      | 推理设备id                     |
| backend          | str | acl    | 推理后端，支持：acl, lite       |
| parallel_num     | int | 1      | 推理流水线中每个节点并行数        |
| precision_mode   | str | fp32   | 推理的精度模式，支持：fp16, fp32 |

- 结果保存

| 参数名称               | 类型  | 默认值             | 含义                      |
|:----------------------|:-----|:------------------|:-------------------------|
| res_save_dir          | str  | inference_results | 推理结果的保存路径           |
| vis_det_save_dir      | str  | 无                | 绘制检测框的图片保存路径      |
| vis_pipeline_save_dir | str  | 无                | 绘制检测框和文本的图片保存路径 |
| vis_font_path         | str  | 无                | 绘制文字时的字体路径         |
| crop_save_dir         | str  | 无                | 文本检测后裁剪图片的保存路径   |
| show_log              | bool | False             | 是否打印日志                |
| save_log_dir          | str  | 无                | 日志保存文件夹              |

- 文本检测

| 参数名称         | 类型 | 默认值 | 含义                   |
|:----------------|:----|:------|:----------------------|
| det_model_path  | str | 无    | 文本检测模型的文件路径     |
| det_model_name  | str | 无    | 文本检测模型的名称        |
| det_config_path | str | 无    | 文本检测模型的配置文件路径 |

- 文本方向分类

| 参数名称         | 类型 | 默认值 | 含义                       |
|:----------------|:----|:------|:-------------------------|
| cls_model_path  | str | 无    | 文本方向分类模型的文件路径     |
| cls_model_name  | str | 无    | 文本方向分类模型的名称        |
| cls_config_path | str | 无    | 文本方向分类模型的配置文件路径 |

- 文本识别

| 参数名称             | 类型 | 默认值 | 含义                                             |
|:--------------------|:----|:------|:------------------------------------------------|
| rec_model_path      | str | 无    | 文本检测模型的文件路径                               |
| rec_model_name      | str | 无    | 文本检测模型的名称                                  |
| rec_config_path     | str | 无    | 文本检测模型的配置文件路径                           |
| character_dict_path | str | 无    | 文本识别模型对应的词典文件路径，默认值只支持数字和英文小写 |

说明：

1. 对于已适配的模型，`*_model_path`、`*_model_name`和`*_config_path`是对应绑定起来的，可参考[MindOCR模型支持列表](./models_list_cn.md)和[第三方模型支持列表](./models_list_thirdparty_cn.md)，
   其中`*_model_name`和`*_config_path`都是用来确定预/后处理参数的，在使用时选择其一即可；

2. 如果需要适配自己的模型，则*_config_path传入自定义的yaml文件即可，格式请参考MindOCR模型的[configs](../../../configs)或第三方模型的[configs](../../../deploy/py_infer/src/configs)。
