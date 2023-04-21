#ifndef MODULES_CLS_POST_PROCESS_H
#define MODULES_CLS_POST_PROCESS_H

#include <unordered_set>

#include "module_manager/module_manager.h"
#include "config_parser/config_parser.h"
#include "profile.h"
#include "data_type/data_type.h"
#include "MxBase/MxBase.h"
#include "Log/Log.h"

class ClsPostProcess : public AscendBaseModule::ModuleBase {
public:
    ClsPostProcess();

    ~ClsPostProcess() override;

    Status Init(CommandParser &options, AscendBaseModule::ModuleInitArgs &initArgs) override;

    Status DeInit() override;

protected:
    Status Process(std::shared_ptr<void> inputData) override;

private:
    Status PostProcessMindXCls(uint32_t framesSize, std::vector<MxBase::Tensor> &inferOutput,
                               std::vector<cv::Mat> &imgMatVec, std::vector<std::string> &inferRes);

    Status PostProcessLiteCls(uint32_t framesSize, std::vector<mindspore::MSTensor> &inferOutput,
                              std::vector<cv::Mat> &imgMatVec, std::vector<std::string> &inferRes);

    void GenerateInferResAndRotate(uint32_t framesSize, const std::vector<cv::Mat> &imgMatVec,
                                   const std::vector<int64_t> &shape,
                                   const float *tensorData, std::vector<std::string> &inferRes);

    std::string nextModule_;
    TaskType taskType_;
};

MODULE_REGIST(ClsPostProcess)

#endif
