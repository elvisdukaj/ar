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
//        deviceId: QtMultimedia.availableCameras[1].deviceId
//        deviceId: QtMultimedia.defaultCamera.deviceId
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
