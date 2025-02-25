English | [中文](../../cn/inference/environment_cn.md)

## Inference - Environment Installationn

MindOCR supports inference for Ascend310/Ascend310P device.

Please make sure that the Ascend AI processor software package is correctly installed on your system. If it is not
installed, please refer to the section
[Installing Ascend AI processor software package](https://www.mindspore.cn/install/en#installing-ascend-ai-processor-software-package)
to install it.

The MindOCR backend supports two types of inference:
[ACL](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_000004.html)
and [MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/master/index.html). Please choose the installation
environment based on the usage scenario, with the following differences:

|                    | ACL                                                                                                                     | Lite                                                                                                             |
|:-------------------|:------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|
| file extension     | .om                                                                                                                     | .mindir                                                                                                          |
| conversion tool    | [ATC](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/atctool/atctool_000001.html) | [converter_lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/converter_tool.html)            |
| application scenes | inference for third-party model, it needs to be converted to an OM file                                                 | 1. inference for MindSpore model<br>2. inference for third-party model<br>it needs to be converted to an OM file |

### 1. ACL inference

For the ACL inference of MindOCR, it currently relies on the Python API interface by
[MindX](https://www.hiascend.com/software/Mindx-sdk), which currently only supports Python 3.9.

| package | version |
|:--------|:--------|
| Python  | 3.9     |
| MindX   | 3.0.0   |

On the basis of the Python 3.9 environment, download the mxVision SDK installation package for
[MindX](https://www.hiascend.com/zh/software/mindx-sdk/commercial) and refer to the
[tutorial](https://www.hiascend.com/document/detail/zh/mind-sdk/300/quickstart/visionquickstart/visionquickstart_0003.html)
for installation. The main steps are as follows:

```shell
# add executable permissions
chmod +x Ascend-mindxsdk-mxvision_{version}_linux-{arch}.run
# execute the installation command
# if prompted to specify the path to CANN, add parameters such as: --cann-path=/usr/local/Ascend/latest
./Ascend-mindxsdk-mxvision_{version}_linux-{arch}.run --install
# set environment variable
source mxVision/set_env.sh
```

After installation, test whether mindx can be imported normally：`python -c "import mindx"`

If prompted that mindx cannot be found, go to the mxVision/Python directory and install the corresponding Whl package:

```
cd mxVision/python
pip install *.whl
```

### 2. MindSpore Lite inference

For the MindSpore Lite inference of MindOCR, It requires the version 2.0.0-rc1 or higher of the
[MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/master/index.html) **cloud-side** inference toolkit.

[Download](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html) the Ascend version of the cloud-side
inference toolkit tar.gz file, as well as the Python interface Wheel package.

The download address provides the Python package for version 3.7. If you need other versions, please refer to the
[compilation tutorial]().

Just decompress the inference toolkit, and set environment variables:

```shell
export LITE_HOME=/your_path_to/mindspore-lite
export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
```

The wheel package of the python interface is installed using pip:

```shell
pip install mindspore_lite-{version}-{python_version}-linux_{arch}.whl
```
