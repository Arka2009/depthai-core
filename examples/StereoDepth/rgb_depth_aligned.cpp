#include <cstdio>
#include <iostream>

#include "utility.hpp"

// Includes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"

// Optional. If set (true), the ColorCamera is downscaled from 1080p to 720p.
// Otherwise (false), the aligned depth is automatically upscaled to 1080p
static std::atomic<bool> downscaleColor{true};
static constexpr int fps = 30;
// The disparity is computed at this resolution, then upscaled to RGB resolution
static constexpr auto monoRes = dai::MonoCameraProperties::SensorResolution::THE_720_P;

static float rgbWeight = 0.4f;
static float depthWeight = 0.6f;

static void updateBlendWeights(int percentRgb, void* ctx) {
    rgbWeight = float(percentRgb) / 100.f;
    depthWeight = 1.f - rgbWeight;
}

int main() {
    using namespace std;

    // Create pipeline
    dai::Pipeline pipeline;
    std::vector<std::string> queueNames;
    std::unordered_map<std::string, unsigned> frameCntMap;
    std::string dirName = "/home/amaity/Desktop/TestImages";

    // Define sources and outputs
    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    auto left = pipeline.create<dai::node::MonoCamera>();
    auto right = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    auto xJpegRgb = pipeline.create<dai::node::XLinkOut>();
    auto xJpeqDis = pipeline.create<dai::node::XLinkOut>();

    xJpegRgb->setStreamName("rgb");
    queueNames.push_back("rgb");
    xJpeqDis->setStreamName("depth");
    queueNames.push_back("depth");

    // Properties
    camRgb->setBoardSocket(dai::CameraBoardSocket::RGB);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setFps(fps);
    if(downscaleColor) camRgb->setIspScale(2, 3);
    // For now, RGB needs fixed focus to properly align with depth.
    // This value was used during calibration
    camRgb->initialControl.setManualFocus(135);

    left->setResolution(monoRes);
    left->setBoardSocket(dai::CameraBoardSocket::LEFT);
    left->setFps(fps);
    right->setResolution(monoRes);
    right->setBoardSocket(dai::CameraBoardSocket::RIGHT);
    right->setFps(fps);

    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    // LR-check is required for depth alignment
    stereo->setLeftRightCheck(true);
    stereo->setDepthAlign(dai::CameraBoardSocket::RGB);

    // Linking
    left->out.link(stereo->left);
    right->out.link(stereo->right);
    camRgb->isp.link(xJpegRgb->input);
    stereo->disparity.link(xJpeqDis->input);

    // Connect to device and start pipeline
    dai::Device device(pipeline);

    // Sets queues size and behavior
    for(const auto& name : queueNames) {
        device.getOutputQueue(name, 4, false);
        frameCntMap[name] = 0;
    }

    std::unordered_map<std::string, cv::Mat> frame;

    auto rgbWindowName = "rgb";
    auto depthWindowName = "depth";
    auto blendedWindowName = "rgb-depth";
    cv::namedWindow(rgbWindowName);
    cv::namedWindow(depthWindowName);
    cv::namedWindow(blendedWindowName);
    int defaultValue = (int)(rgbWeight * 100);
    cv::createTrackbar("RGB Weight %", blendedWindowName, &defaultValue, 100, updateBlendWeights);

    while(true) {
        std::unordered_map<std::string, std::shared_ptr<dai::ImgFrame>> latestPacket;

        auto queueEvents = device.getQueueEvents(queueNames);
        for(const auto& name : queueEvents) {
            auto packets = device.getOutputQueue(name)->tryGetAll<dai::ImgFrame>();
            auto count = packets.size();
            if(count > 0) {
                latestPacket[name] = packets[count - 1];
            }
        }

        for(const auto& name : queueNames) {
            if(latestPacket.find(name) != latestPacket.end()) {
                if (name == "depth") {
                    auto maxDisparity = stereo->initialConfig.getMaxDisparity();
                    frame[name] = latestPacket[name]->getFrame();
                    frame[name].convertTo(frame[name], CV_8UC1, 255. / maxDisparity);
                } else 
                    frame[name] = latestPacket[name]->getCvFrame();

                std::stringstream videoStr;

                videoStr << dirName << "/" << name << "_" << frameCntMap[name]++ << ".jpeg";
                cv::imwrite(videoStr.str(),frame[name]);

            }
        }
    }
    return 0;
}
