#ifndef MODULES_CLS_INFER_PROCESS_H
#define MODULES_CLS_INFER_PROCESS_H

#include "module_manager/module_manager.h"
#include "module_manager/module_factory.h"
#include "config_parser/config_parser.h"
#include "data_type/data_type.h"
#include "profile.h"
#include "MxBase/MxBase.h"
#include "Log/Log.h"
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/types.h"
#include "utils.h"
#include "cls_post_process/cls_post_process.h"

using namespace AscendBaseModule;

class ClsInferProcess : public ModuleBase {
public:
    ClsInferProcess();

    ~ClsInferProcess() override;

    Status Init(CommandParser &options, ModuleInitArgs &initArgs) override;

    Status DeInit() override;

protected:
    Status Process(std::shared_ptr<void> inputData) override;

private:
    int clsHeight_;
    int clsWidth_;
    int32_t deviceId_ = 0;
    std::unique_ptr<MxBase::Model> clsNetMindX_;
    mindspore::Model *clsNetLite_;
    std::vector<uint32_t> batchSizeList_;
    BackendType backend_;

    Status ParseConfig(CommandParser &options, ModuleInitArgs &initArgs);

    Status MindXModelInfer(const std::shared_ptr<CommonData> &data);

    Status LiteModelInfer(const std::shared_ptr<CommonData> &data);
};

MODULE_REGIST(ClsInferProcess)

#endif
