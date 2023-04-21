#ifndef MODULES_SIGNAL_H
#define MODULES_SIGNAL_H

#include <atomic>
#include <mutex>

class profile {
public:
    profile(const profile &) = delete;

    profile operator = (const profile &) = delete;

    ~profile() = default;

    static profile &GetInstance();

    int GetThreadNum() const;

    void SetThreadNum(int num);

    std::atomic_int &GetStoppedThreadNum();

    static bool signalReceived_;
    static double detPreProcessTime_;
    static double detInferTime_;
    static double detPostProcessTime_;

    static double clsPreProcessTime_;
    static double clsInferTime_;
    static double clsPostProcessTime_;

    static double recPreProcessTime_;
    static double recInferTime_;
    static double recPostProcessTime_;
    static double e2eProcessTime_;

    static double detInferProcessTime_;
    static double recInferProcessTime_;
    static double clsInferProcessTime_;

private:
    std::atomic_int threadNum_ {0 };
    std::atomic_int stoppedThreadNum_ {0 };

    profile() {}
};

#endif
