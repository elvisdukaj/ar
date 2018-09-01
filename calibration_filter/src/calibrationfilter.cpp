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

#include "calibrationfilter.h"
#include <opencv2/core/types_c.h>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <QDebug>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <fstream>
using namespace std;

// Print camera parameters to the output file
static void saveCameraParams(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, double totalAvgErr )
{
    cv::FileStorage fs( "cameraCalibration.xml", cv::FileStorage::WRITE );

    fs << "CameraMatrix" << cameraMatrix;
    fs << "DistortionCoefficients" << distCoeffs;
    fs << "AvgReprojection_Error" << totalAvgErr;
}

QVideoFilterRunnable* CalibrationFilter::createFilterRunnable()
{
    return new CalibrationFilterRunnable(this);
}

CalibrationFilterRunnable::CalibrationFilterRunnable(CalibrationFilter* filter)
    : m_filter{filter}
{
}

QVideoFrame CalibrationFilterRunnable::run(QVideoFrame* frame, const QVideoSurfaceFormat&, QVideoFilterRunnable::RunFlags)
{
    if (!isFrameValid(frame))
    {
        qDebug() << "Frame is NOT valid";
        return QVideoFrame{};
    }

    if (!frame->map(QAbstractVideoBuffer::ReadWrite))
    {
        qDebug() << "Unable to map the videoframe in memory" << endl;
        return *frame;
    }

    try
    {
        if (m_filter->hasToShowNegative())
            showNegative(frame);

        else if (!m_filter->isCalibrated())
            acquireFrame(frame);

        else if(m_filter->isCalibrated() && m_filter->showUnsistorted())
            showUndistorted(frame);
        else
            showFlipped(frame);


    }
    catch(const std::exception& exc)
    {
        qDebug() << exc.what();
    }

    frame->unmap();
    return *frame;
}

void CalibrationFilterRunnable::showFlipped(QVideoFrame* frame)
{
    cv::Mat frameMat;
    convertToCvMat(frame, frameMat);
    cv::flip(frameMat, frameMat, 1);
}

void CalibrationFilterRunnable::showNegative(QVideoFrame* frame)
{
    cv::Mat frameMat;
    convertToCvMat(frame, frameMat);
    cv::flip(frameMat, frameMat, 1);
    m_lastFrameWithChessBoard.copyTo(
                frameMat(cv::Rect{
                             20, 20,
                             m_lastFrameWithChessBoard.cols,
                             m_lastFrameWithChessBoard.rows
                         }
                         )
                );
}

void CalibrationFilterRunnable::showUndistorted(QVideoFrame* frame)
{
    cv::Mat frameMat;
    convertToCvMat(frame, frameMat);
    cv::flip(frameMat, frameMat, 1);
    m_calibrator.remap(frameMat, frameMat );
}

void CalibrationFilterRunnable::acquireFrame(QVideoFrame* frame)
{
    cv::Mat frameMat, grayscale;
    videoFrameInGrayScaleAndColor(frame, grayscale, frameMat);

    auto corners = m_calibrator.findChessboard(grayscale);

    if (!corners.empty())
    {
        cv::drawChessboardCorners(
                    frameMat,
                    m_calibrator.boardSize(),
                    corners,
                    true
                    );

        emit m_filter->chessBoardFound();
        cv::bitwise_not(frameMat, m_lastFrameWithChessBoard);
        cv::resize(m_lastFrameWithChessBoard, m_lastFrameWithChessBoard,
                   cv::Size{0, 0}, 0.25, 0.25
                   );

        m_filter->addGoodFrame();

        if (m_filter->goodFrames() == m_filter->maxFrames())
        {
            auto error = m_calibrator.calibrate(frameMat.size());

            saveCameraParams(m_calibrator.cameraMatrix(), m_calibrator.distortion(), error);

            qDebug() << "calibration done! Error: " << error;

            m_filter->setCalibrated();
            emit m_filter->calibrationFinished();
        }
    }
}

void CalibrationFilterRunnable::convertToCvMat(QVideoFrame *frame, cv::Mat& frameMat)
{
    auto width = frame->width();
    auto height = frame->height();
    auto data = frame->bits();

    switch (frame->pixelFormat()) {
    case QVideoFrame::Format_RGB32:
        frameMat = cv::Mat{height, width, CV_8UC4, data};
        return;

    case QVideoFrame::Format_RGB24:
        frameMat = cv::Mat{height, width, CV_8UC3, data};
        return;

    case QVideoFrame::Format_YUV420P:
        frameMat = cv::Mat{height, width, CV_8UC1, data};
        return;

    default:
        throw std::runtime_error{"Unknown video frame type"};
    }
}

vector<cv::Point2f> CameraCalibrator::findChessboard(const cv::Mat& grayscale)
{
    vector<cv::Point2f> imageCorners;
    if (cv::findChessboardCorners(grayscale, m_boardSize, imageCorners, cv::CALIB_CB_FAST_CHECK))
    {
        cv::cornerSubPix(
                    grayscale, imageCorners, m_boardSize,
                    cv::Size(-1, -1),
                    cv::TermCriteria{
                        CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,
                        300, 0.01}
                    );

        if (imageCorners.size() == m_boardSize.area())
            addPoints(imageCorners);
        else
            imageCorners.clear();
    }

    return imageCorners;
}

void CameraCalibrator::addPoints(const std::vector<cv::Point2f> &imageCorners)
{
    vector<cv::Point3f> objectCorners;

    for (auto y = 0; y < m_boardSize.height; ++y)
        for (auto x = 0; x < m_boardSize.width; ++x)
            objectCorners.push_back(cv::Point3f(y, x, 0.0f));

    m_worldObjectPoints.push_back(objectCorners);
    m_imagePoints.push_back(imageCorners);
}


using namespace cv;
double CameraCalibrator::calibrate(cv::Size& imageSize)
{
    qDebug() << "Calibrating...";
    m_mustInitUndistort = true;

    // start calibration
    auto res = cv::calibrateCamera(
                m_worldObjectPoints,
                m_imagePoints,
                imageSize,
                m_cameraMatrix,
                m_distCoeffs,
                m_rotationVecs,
                m_translationtVecs
                );

    cout << "Camera matrix: " << m_cameraMatrix << '\n'
         << "Camera distortion: " << m_distCoeffs << '\n'
         << endl;

    return res;
}

void CameraCalibrator::CameraCalibrator::remap(const cv::Mat& image, cv::Mat& outImage)
{
    if (m_mustInitUndistort)
    {
        cv::initUndistortRectifyMap(
                    m_cameraMatrix,     // computed camera matrix
                    m_distCoeffs,       // computed distortion matrix
                    cv::Mat(),          // optional rectification (none)
                    cv::Mat(),          // camera matrix to generate undistorted
                    image.size(),       // size of undistorted
                    CV_32FC1,           // type of output map
                    m_mapX, m_mapY      // the x and y mapping functions
                    );
        m_mustInitUndistort = false;
    }

    // Apply mapping functions
    cv::remap(image, outImage, m_mapX, m_mapY, cv::INTER_LINEAR);
}
