#include "utility/Initialization.hpp"

// project
#include "build/version.hpp"
#include "utility/Resources.hpp"

// libraries
#include "XLink/XLink.h"
#include "backward.hpp"
#include "spdlog/cfg/env.h"
#include "spdlog/cfg/helpers.h"
#include "spdlog/details/os.h"
#include "spdlog/spdlog.h"
extern "C" {
#include "XLink/XLinkLog.h"
}

// For easier access to dai namespaced symbols
namespace dai {

// Anonymous namespace to hide 'Preloader' symbol and variable as its not needed to be visible to other compilation units
namespace {

// Doing early static initialization hits this stage faster than some libraries initialize their global static members

// Preloader uses static global object constructor (works only for shared libraries)
// to execute some code upon final executable launch  or library import
// Preloader
// struct Preloader {
//     Preloader(){
//         initialize();
//     }
// } preloader;

}  // namespace

// Backward library stacktrace handling
static backward::SignalHandling* pSignalHandler;

bool initialize(std::string additionalInfo, bool installSignalHandler) {
    // atomic bool for checking whether depthai was already initialized
    static std::atomic<bool> initialized{false};
    if(initialized.exchange(true)) return true;

    // install backward if specified
    auto envSignalHandler = spdlog::details::os::getenv("DEPTHAI_INSTALL_SIGNAL_HANDLER");
    if(installSignalHandler && envSignalHandler != "0") {
        pSignalHandler = new backward::SignalHandling;
    }

    // Set global logging level from ENV variable 'DEPTHAI_LEVEL'
    // Taken from spdlog, to replace with DEPTHAI_LEVEL instead of SPDLOG_LEVEL
    // spdlog::cfg::load_env_levels();
    auto envLevel = spdlog::details::os::getenv("DEPTHAI_LEVEL");
    if(!envLevel.empty()) {
        spdlog::cfg::helpers::load_levels(envLevel);
    } else {
        // Otherwise set default level to WARN
        spdlog::set_level(spdlog::level::warn);
    }

    // Print core commit and build datetime
    if(!additionalInfo.empty()) {
        spdlog::debug("{}", additionalInfo);
    }
    spdlog::debug(
        "Library information - version: {}, commit: {} from {}, build: {}", build::VERSION, build::COMMIT, build::COMMIT_DATETIME, build::BUILD_DATETIME);

    // Executed at library load time

    // Preload Resources (getting instance causes some internal lazy loading to start)
    Resources::getInstance();

    // Static global handler
    static XLinkGlobalHandler_t xlinkGlobalHandler = {};
    xlinkGlobalHandler.protocol = X_LINK_USB_VSC;
    auto status = XLinkInitialize(&xlinkGlobalHandler);
    if(X_LINK_SUCCESS != status) {
        throw std::runtime_error("Couldn't initialize XLink");
    }
    // Suppress XLink related errors
    mvLogDefaultLevelSet(MVLOG_LAST);

    spdlog::debug("Initialize - finished");

    return true;
}

}  // namespace dai
