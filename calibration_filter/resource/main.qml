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

import QtQuick 2.7
import QtQuick.Controls 2.0
import QtQuick.Layouts 1.3
import QtMultimedia 5.5

import com.elvis.ar 1.0

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    title: qsTr("Calibration Example")

    Camera {
        id: camera
        deviceId: QtMultimedia.availableCameras[1].deviceId
    }

    CalibrationFilter {
        id: calibrationFilter

        onChessBoardFound: {
            timer.start()
            calibrationFilter.showNegative = true;
            progressiveText.text = goodFrames + " / " + maxFrames;
        }

        onCalibrationFinished: {
            showUndistorted.enabled = true
            progressiveText.enabled = false
        }

        showUnsistorted: showUndistorted.checked
    }

    VideoOutput {
        id: videoOutput
        source: camera
        anchors.fill: parent

        focus : visible // to receive focus and capture key events when visible
        filters: [calibrationFilter]
    }

    Timer {
        id: timer
        interval: 2000
        running: false
        repeat: false

        onTriggered: {
            calibrationFilter.showNegative = false;
        }
    }

    Switch {
        id: showUndistorted

        anchors.bottom: parent.bottom
        anchors.right: parent.right

        text: qsTr("Show Undistorted")
        checked: false
        checkable: true
        enabled: false
    }

    Label {
        id: progressiveText

        anchors.bottom: parent.bottom
        anchors.left: parent.left

        text: ""
    }
}
