#pragma once

#include <QAbstractVideoFilter>
#include <opencv2/core.hpp>

class AbstractVideoFilterRunnable : public QVideoFilterRunnable {
public:
    bool isFrameValid(QVideoFrame* frame) const noexcept;

protected:
    void videoFrameInGrayScaleAndColor(QVideoFrame* frame, cv::Mat& grayscale, cv::Mat& frameMat);
    void grayscaleToVideoFrame(QVideoFrame* frame, const cv::Mat& grayscale, cv::Mat& frameMat) const;
};
