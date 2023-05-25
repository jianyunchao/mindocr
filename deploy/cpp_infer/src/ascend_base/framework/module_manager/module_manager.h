#ifndef ASCEND_BASE_MODULE_MANAGER_H
#define ASCEND_BASE_MODULE_MANAGER_H

#include "acl/acl.h"
#include "Log/Log.h"
#include "module_manager/module_base.h"
#include "module_manager/module_factory.h"
#include "command_parser/command_parser.h"
#include "utils.h"

namespace AscendBaseModule {
    const std::string PIPELINE_DEFAULT = "DefaultPipeline";

    struct ModuleDesc {
        std::string moduleName;
        int moduleCount; // -1 using the defaultCount
    };

    struct ModuleConnectDesc {
        std::string moduleSend;
        std::string moduleRecv;
        ModuleConnectType connectType;
    };

// information for one type of module
    struct ModulesInformation {
        std::vector<std::shared_ptr<ModuleBase>> moduleVec;
        std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueueVec;
    };

    using ModulesInfo = ModulesInformation;

    class ModuleManager {
    public:
        ModuleManager();

        ~ModuleManager();

        Status Init(CommandParser &options, std::string &aclConfigPath);

        Status DeInit();

        Status
        RegisterModules(const std::string &pipelineName, ModuleDesc *moduleDesc, int moduleTypeCount, int defaultCount);

        Status
        RegisterModuleConnects(const std::string &pipelineName, ModuleConnectDesc *connectDescArr,
                               int moduleConnectCount);

        Status RegisterInputVec(const std::string &pipelineName, const std::string &moduleName,
                                std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueVec);

        Status RegisterOutputModule(const std::string &pipelineName, const std::string &moduleSend,
                                    const std::string &moduleRecv,
                                    ModuleConnectType connectType,
                                    const std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> &outputQueVec);

        Status RunPipeline();

    private:
        Status InitModuleInstance(const std::shared_ptr<ModuleBase> &moduleInstance, int instanceId,
                                  const std::string &pipelineName, const std::string &moduleName);

        Status InitPipelineModule();

        Status DeInitPipelineModule();

        static void StopModule(const std::shared_ptr<ModuleBase> &moduleInstance);

    private:
        int32_t deviceId_ = 0;
        std::map<std::string, std::map<std::string, ModulesInfo>> pipelineMap_ = {};
        ConfigParser configParser_ = {};
        CommandParser options_ = {};
    };
}

#endif
