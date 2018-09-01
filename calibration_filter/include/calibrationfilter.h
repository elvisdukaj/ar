// Copyright (c) 2017 Elvis Dukaj
// 
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
// 
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "abstractopencvrunnablefilter.h"
#include <opencv2/core.hpp>
#include <vector>

class CalibrationFilter : public QAbstractVideoFilter {
    Q_OBJECT

    Q_PROPERTY(QSize chessBoardSize READ chessBoardSize WRITE chessBoardSize)

    Q_PROPERTY(bool chessBoardFound READ isChessBoardFound WRITE setChessBoardFound NOTIFY chessBoardFound)
    Q_PROPERTY(bool calibrated READ isCalibrated WRITE setCalibrated NOTIFY calibrationFinished)

    Q_PROPERTY(bool showNegative READ hasToShowNegative WRITE setShowNegative)
    Q_PROPERTY(bool showUnsistorted READ showUnsistorted WRITE showUnsistorted)

    Q_PROPERTY(int maxFrames READ maxFrames )
    Q_PROPERTY(int goodFrames READ goodFrames)

public:
    QVideoFilterRunnable* createFilterRunnable() override;

    QSize chessBoardSize() const noexcept { return m_chessboardSize; }
    void chessBoardSize(const QSize& size) { m_chessboardSize = size; }

    bool isChessBoardFound() const { return m_chessboardFound; }
    void setChessBoardFound(bool f) { m_chessboardFound = f; }

    int isCalibrated() const { return m_calibrated; }
    void setCalibrated(bool cal = true) { m_calibrated = cal; }

    bool hasToShowNegative() const noexcept { return m_showNegative; }
    void setShowNegative(bool s) noexcept { m_showNegative = s; }

    bool showUnsistorted() const noexcept { return m_showUnsistorted; }
    void showUnsistorted(bool show) noexcept { m_showUnsistorted = show; }

    int maxFrames() const noexcept { return m_maxFrames; }
    void maxFrames(int n) noexcept { m_maxFrames = n; }

    int goodFrames() const noexcept { return m_goodFrames; }
    void addGoodFrame() { ++m_goodFrames; }


signals:
    void chessBoardFound();
    void calibrationFinished();

private:
    friend class ThresholdFilterRunnable;

private:
    QSize m_chessboardSize = QSize{8, 6};
    bool m_calibrated = false;
    bool m_cornerFound = false;
    bool m_chessboardFound = false;
    bool m_showNegative = false;
    bool m_showUnsistorted = true;
    int m_maxFrames = 25;
    int m_goodFrames = 0;
};

class CameraCalibrator {
public:
    CameraCalibrator(const cv::Size boardSize = cv::Size{8, 6})
        : m_boardSize{boardSize} {}

    std::vector<cv::Point2f> findChessboard(const cv::Mat& mat);
    double calibrate(cv::Size& imageSize);
    void remap(const cv::Mat& image, cv::Mat& outImage);
    const cv::Size boardSize() const noexcept { return m_boardSize; }

    const cv::Mat cameraMatrix() const noexcept { return m_cameraMatrix; }
    const cv::Mat distortion() const noexcept { return m_distCoeffs; }

private:
    void addPoints(const std::vector<cv::Point2f>& imageCorners);

private:
    cv::Size m_boardSize;

    std::vector<std::vector<cv::Point3f>> m_worldObjectPoints;
    std::vector<std::vector<cv::Point2f>> m_imagePoints;

    cv::Mat m_foundChessBoard;

    cv::Mat m_cameraMatrix;
    cv::Mat m_distCoeffs;
    std::vector<cv::Mat> m_rotationVecs;
    std::vector<cv::Mat> m_translationtVecs;
    cv::Mat m_mapX, m_mapY;
    bool m_mustInitUndistort;
};

class CalibrationFilterRunnable : public AbstractVideoFilterRunnable {
public:
    CalibrationFilterRunnable(CalibrationFilter* filter);
    QVideoFrame run(QVideoFrame* input, const QVideoSurfaceFormat &surfaceFormat, RunFlags flags) override;    

private:
    void showFlipped(QVideoFrame* frame);
    void showNegative(QVideoFrame* frame);
    void showUndistorted(QVideoFrame* frame);
    void acquireFrame(QVideoFrame* frame);
    void convertToCvMat(QVideoFrame* frame, cv::Mat& mat);

private:
    CalibrationFilter* m_filter;
    cv::Mat m_lastFrameWithChessBoard;
    CameraCalibrator m_calibrator;
};
