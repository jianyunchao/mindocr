[English](../../en/datasets/mlt2017.md) | 中文

# 数据集下载

MLT (Multi-Lingual) 2017 [文章](https://ieeexplore.ieee.org/abstract/document/8270168)

[下载地址](https://rrc.cvc.uab.es/?ch=8&com=downloads): 在下载之前，您需要先注册一个账号。

<details>
  <summary>从何处下载 MLT 2017</summary>

MLT 2017 数据集包含两个任务. 任务 1 是文本检测 (多语言文本)。 任务2是文本识别。

### 文本检测

有11个与任务1相关的文件需要下载（[下载地址](https://rrc.cvc.uab.es/?ch=8&com=downloads)）， 它们分别是：

```
ch8_training_images_x.zip(x from 1 to 8)
ch8_validation_images.zip
ch8_training_localization_transcription_gt_v2.zip
ch8_validation_localization_transcription_gt_v2.zip
```

测试集不需要下载。

### 文本识别

有6个与任务2相关的文件需要下载（[下载地址](https://rrc.cvc.uab.es/?ch=8&com=downloads)）， 它们分别是：
```
 ch8_training_word_images_gt_part_x.zip (x from 1 to 3)
 ch8_validation_word_images_gt.zip
 ch8_training_word_gt_v2.zip
 ch8_validation_word_gt_v2.zip
 ```
</details>


在下载完成后, 将文件放于 `[path-to-data-dir]` 文件夹内，如下所示:
```
path-to-data-dir/
  mlt2017/
    # text detection
    ch8_training_images_1.zip
    ch8_training_images_2.zip
    ch8_training_images_3.zip
    ch8_training_images_4.zip
    ch8_training_images_5.zip
    ch8_training_images_6.zip
    ch8_training_images_7.zip
    ch8_training_images_8.zip
    ch8_training_localization_transcription_gt_v2.zip
    ch8_validation_images.zip
    ch8_validation_localization_transcription_gt_v2.zip
    # word recognition
    ch8_training_word_images_gt_part_1.zip
    ch8_training_word_images_gt_part_2.zip
    ch8_training_word_images_gt_part_3.zip
    ch8_training_word_gt_v2.zip
    ch8_validation_word_images_gt.zip
    ch8_validation_word_gt_v2.zip
    
    
```

[返回](../../../tools/dataset_converters/README_CN.md)