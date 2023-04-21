#ifndef MODULES_CLS_PREPROCESS_H
#define MODULES_CLS_PREPROCESS_H

#include <securec.h>
#include "Log/Log.h"
#include "module_manager/module_manager.h"
#include "config_parser/config_parser.h"
#include "utils.h"
#include "data_type/data_type.h"
#include "profile.h"
#include "command_parser/command_parser.h"


class ClsPreProcess : public AscendBaseModule::ModuleBase {
public:
    ClsPreProcess();

    ~ClsPreProcess() override;

    Status Init(CommandParser &options, AscendBaseModule::ModuleInitArgs &initArgs) override;

    Status DeInit() override;

protected:
    Status Process(std::shared_ptr<void> inputData) override;

private:
    int clsHeight_ = 48;
    int clsWidth_ = 192;
    int32_t deviceId_ = 0;
    std::vector<uint64_t> batchSizeList_;

    Status ParseConfig(CommandParser &configParser);

    std::vector<uint32_t> GetClsBatchSize(uint32_t frameSize);

    uint8_t *PreprocessCls(std::vector<cv::Mat> &frames, uint32_t batchSize);

    TaskType taskType_;
};

MODULE_REGIST(ClsPreProcess)

#endif
