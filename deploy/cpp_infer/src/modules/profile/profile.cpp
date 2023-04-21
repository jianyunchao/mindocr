#include "profile.h"

static std::mutex g_mtx = {};
bool profile::signalReceived_ = false;
double profile::detInferTime_ = 0;
double profile::recInferTime_ = 0;
double profile::e2eProcessTime_ = 0;

double profile::detPreProcessTime_ = 0;
double profile::detPostProcessTime_ = 0;

double profile::clsPreProcessTime_ = 0;
double profile::clsInferTime_ = 0;
double profile::clsPostProcessTime_ = 0;

double profile::recPreProcessTime_ = 0;
double profile::recPostProcessTime_ = 0;

double profile::detInferProcessTime_ = 0;
double profile::recInferProcessTime_ = 0;
double profile::clsInferProcessTime_ = 0;

profile &profile::GetInstance() {
    std::unique_lock<std::mutex> lock(g_mtx);
    static profile singleton;
    return singleton;
}

std::atomic_int &profile::GetStoppedThreadNum() {
    return stoppedThreadNum_;
}

int profile::GetThreadNum() const {
    return threadNum_;
}

void profile::SetThreadNum(int num) {
    threadNum_ = num;
}
