#include "crnn_post_process.h"
#include "collect_process/collect_process.h"
#include "MxBase/MxBase.h"
#include "utils.h"

using namespace AscendBaseModule;

CrnnPostProcess::CrnnPostProcess() {
    withoutInputQueue_ = false;
    isStop_ = false;
}

CrnnPostProcess::~CrnnPostProcess() = default;

Status CrnnPostProcess::Init(CommandParser &options, ModuleInitArgs &initArgs) {
    LogInfo << "Begin to init instance " << initArgs.instanceId;

    AssignInitArgs(initArgs);
    Status ret = ParseConfig(options);
    if (ret != Status::OK) {
        LogError << "crnn_post_process[" << instanceId_ << "]: Fail to parse config params.";
        return ret;
    }
    crnnPost_.ClassNameInit(recDictionary_);

    LogInfo << "crnn_post_process [" << instanceId_ << "] Init success.";
    return Status::OK;
}

Status CrnnPostProcess::DeInit() {
    LogInfo << "crnn_post_process [" << instanceId_ << "]: Deinit success.";
    return Status::OK;
}

Status CrnnPostProcess::ParseConfig(CommandParser &options) {
    recDictionary_ = options.GetStringOption("--character_dict_path");
    Status ret = Utils::CheckPath(recDictionary_, "character label file");
    if (ret != Status::OK) {
        LogError << "Character label file: " << recDictionary_ << " is not exist of can not read.";
        return Status::COMM_INVALID_PARAM;
    }
    LogDebug << "dictPath: " << recDictionary_;

    resultPath_ = options.GetStringOption("--res_save_dir");
    if (resultPath_[resultPath_.size() - 1] != '/') {
        resultPath_ += "/";
    }
    return Status::OK;
}

Status CrnnPostProcess::PostProcessMindXCrnn(uint32_t framesSize, std::vector<MxBase::Tensor> &inferOutput,
                                             std::vector<std::string> &textsInfos) {
    auto *objectInfo = (int64_t *) inferOutput[0].GetData();
    auto objectNum = (size_t) inferOutput[0].GetShape()[1];
    crnnPost_.CalcMindXOutputIndex(objectInfo, framesSize, objectNum, textsInfos);
    return Status::OK;
}

Status CrnnPostProcess::PostProcessLiteCrnn(uint32_t framesSize, std::vector<mindspore::MSTensor> &inferOutput,
                                            std::vector<std::string> &textsInfos) {
    auto *objectInfo = (int32_t *) inferOutput[0].Data().get();
    auto objectNum = (size_t) inferOutput[0].Shape()[1];
    crnnPost_.CalcLiteOutputIndex(objectInfo, framesSize, objectNum, textsInfos);
    return Status::OK;
}

Status CrnnPostProcess::Process(std::shared_ptr<void> commonData) {
    auto startTime = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CommonData> data = std::static_pointer_cast<CommonData>(commonData);
    std::vector<std::string> recResVec;
    if (!data->eof) {
        Status ret;
        if (data->backendType == BackendType::ACL) {
            ret = PostProcessMindXCrnn(data->frameSize, data->outputMindXTensorVec, data->inferRes);
        } else if (data->backendType == BackendType::LITE) {
            ret = PostProcessLiteCrnn(data->frameSize, data->outputLiteTensorVec, data->inferRes);
        } else {
            ret = Status::UNSUPPORTED_INFER_ENGINE;
        }
        if (ret != Status::OK) {
            return ret;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    profile::recPostProcessTime_ += costTime;
    profile::e2eProcessTime_ += costTime;
    SendToNextModule(MT_CollectProcess, data, data->channelId);
    return Status::OK;
}
