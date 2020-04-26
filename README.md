## A PyTorch Module for Performing Vectorized Optical Flow

In OpenVINO >= 2020.1, the model obtains the camera config from the plugin. Using Motion Tracking on single object or several objects, the flow model is integrated with ONNX. 

## Calibration of the Model

Parameters:

- Camera View Config for each Person
- Calibration of 3D space over 2D plane, using Homography and Camera Attributes
- Measurement of Distance of Person from Camera
- A MaxMin Selection of A Single Non-Oscillating point

