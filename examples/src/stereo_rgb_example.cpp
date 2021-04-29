

#include <cstdio>
#include <iostream>

#include "utility.hpp"

// Inludes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"

int main() {
    using namespace std;
    using namespace std::chrono;
    // TODO - split this example into two separate examples
    bool withDepth = true;

    dai::Pipeline p;

    auto monoLeft = p.create<dai::node::MonoCamera>();
    auto cam = p.create<dai::node::ColorCamera>();


    // auto monoRight = p.create<dai::node::MonoCamera>();
    auto xoutLeft = p.create<dai::node::XLinkOut>();
    // auto xoutRight = p.create<dai::node::XLinkOut>();
    auto stereo = withDepth ? p.create<dai::node::StereoDepth>() : nullptr;
    auto xoutDisp = p.create<dai::node::XLinkOut>();
    auto xoutDepth = p.create<dai::node::XLinkOut>();
    auto xoutRectifL = p.create<dai::node::XLinkOut>();
    auto xoutRectifR = p.create<dai::node::XLinkOut>();

    // XLinkOut
    xoutLeft->setStreamName("left");
    // xoutRight->setStreamName("right");
    if(withDepth) {
        xoutDisp->setStreamName("disparity");
        xoutDepth->setStreamName("depth");
        xoutRectifL->setStreamName("rectified_left");
        xoutRectifR->setStreamName("rectified_right");
    }

    // MonoCamera
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);
    monoLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
    monoLeft->setImageOrientation(dai::CameraImageOrientation::ROTATE_180_DEG);

    cam->setBoardSocket(dai::CameraBoardSocket::RGB);
    cam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_4_K);
    cam->setIspScale(1, 3);
    cam->initialControl.setManualFocus(135);
    cam->setImageOrientation(dai::CameraImageOrientation::ROTATE_180_DEG);

    // monoLeft->setFps(5.0);
    // monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::1080);
    // monoRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);
    // monoRight->setFps(5.0);

    bool outputDepth = true;
    bool outputRectified = true;
    bool lrcheck = true;
    bool extended = false;
    bool subpixel = true;

    int maxDisp = 96;
    if(extended) maxDisp *= 2;
    if(subpixel) maxDisp *= 32;  // 5 bits fractional disparity

    if(withDepth) {
        // StereoDepth
        stereo->setConfidenceThreshold(200);
        stereo->setRectifyEdgeFillColor(0);  // black, to better see the cutout
        // stereo->loadCalibrationFile("../../../../depthai/resources/depthai.calib");
        // stereo->setInputResolution(1280, 720);
        // TODO: median filtering is disabled on device with (lrcheck || extended || subpixel)
        // stereo->setMedianFilter(dai::StereoDepthProperties::MedianFilter::MEDIAN_OFF);
        stereo->setLeftRightCheck(lrcheck);
        stereo->setExtendedDisparity(extended);
        stereo->setSubpixel(subpixel);

        // Link plugins CAM -> STEREO -> XLINK
        monoLeft->out.link(stereo->left);
        cam->isp.link(stereo->right);

        stereo->syncedLeft.link(xoutLeft->input);
        // stereo->syncedRight.link(xoutRight->input);
        if(outputRectified) {
            stereo->rectifiedLeft.link(xoutRectifL->input);
            stereo->rectifiedRight.link(xoutRectifR->input);
        }
        stereo->disparity.link(xoutDisp->input);
        stereo->depth.link(xoutDepth->input);

    } else {
        // Link plugins CAM -> XLINK
        monoLeft->out.link(xoutLeft->input);
        // cam->isp.link(xoutRight->input);
    }

    // CONNECT TO DEVICE
    dai::Device d(p);
    d.startPipeline();

    auto leftQueue = d.getOutputQueue("left", 8, false);
    // auto rightQueue = d.getOutputQueue("right", 8, false);
    auto dispQueue = withDepth ? d.getOutputQueue("disparity", 8, false) : nullptr;
    auto depthQueue = withDepth ? d.getOutputQueue("depth", 8, false) : nullptr;
    auto rectifLeftQueue = withDepth ? d.getOutputQueue("rectified_left", 8, false) : nullptr;
    auto rectifRightQueue = withDepth ? d.getOutputQueue("rectified_right", 8, false) : nullptr;
    
    cout << "Queses ready " <<  std::endl;
    while(1) {
        auto t1 = steady_clock::now();
        cout << "leftQueue ready " <<  std::endl;

        auto left = leftQueue->get<dai::ImgFrame>();

        auto t2 = steady_clock::now();
        cv::imshow("left", cv::Mat(left->getHeight(), left->getWidth(), CV_8UC1, left->getData().data()));
        auto t3 = steady_clock::now();
        cout << "rightQueue ready " <<  std::endl;

        // auto right = rightQueue->get<dai::ImgFrame>();
    // cout << "rightQueue ready " << right->getWidth() << " " << right->getHeight() << " " << right->getData().size() << std::endl;

        
        auto t4 = steady_clock::now();
        // cv::imshow("right", right->getCvFrame());
        // cv::imshow("right", cv::Mat(right->getHeight(), right->getWidth(), CV_8UC1, right->getData().data()));

        auto t5 = steady_clock::now();

        if(withDepth) {
            // Note: in some configurations (if depth is enabled), disparity may output garbage data
            auto disparity = dispQueue->get<dai::ImgFrame>();
            cout << "dispQueue ready " << disparity->getHeight() << disparity->getWidth() << " " << disparity->getData().size() << std::endl;

            cv::Mat disp(disparity->getHeight(), disparity->getWidth(), subpixel ? CV_16UC1 : CV_8UC1, disparity->getData().data());
            cout << "leftQueue ready " <<  std::endl;

            disp.convertTo(disp, CV_8UC1, 255.0 / maxDisp);  // Extend disparity range
            cout << "leftQueue ready " <<  std::endl;

            cv::imshow("disparity", disp);
            cout << "leftQueue ready " <<  std::endl;

            cv::Mat disp_color;
            cv::applyColorMap(disp, disp_color, cv::COLORMAP_JET);
            cv::imshow("disparity_color", disp_color);

            if(outputDepth) {
                cout << "depth ready " <<  std::endl;

                auto depth = depthQueue->get<dai::ImgFrame>();
                cv::imshow("depth", cv::Mat(depth->getHeight(), depth->getWidth(), CV_16UC1, depth->getData().data()));
            }

            // if(outputRectified) {
            //     auto rectifL = rectifLeftQueue->get<dai::ImgFrame>();
            //     cv::Mat rectifiedLeftFrame = rectifL->getFrame();
            //     cv::flip(rectifiedLeftFrame, rectifiedLeftFrame, 1);
            //     cv::imshow("rectified_left", rectifiedLeftFrame);

            //     auto rectifR = rectifRightQueue->get<dai::ImgFrame>();
            //     cv::Mat rectifiedRightFrame = rectifR->getFrame();

            //     cv::flip(rectifiedRightFrame, rectifiedRightFrame, 1);
            //     cv::imshow("rectified_right", rectifiedRightFrame);
            // }
        }

        int ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        int ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        int ms3 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
        int ms4 = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();
        int loop = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t1).count();

        std::cout << ms1 << " " << ms2 << " " << ms3 << " " << ms4 << " loop: " << loop << std::endl;
        int key = cv::waitKey(1);
        if(key == 'q') {
            return 0;
        }
    }
}