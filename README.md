## A PyTorch Module for Performing Vectorized Optical Flow

In OpenVINO >= 2020.1, the model obtains the camera config from the plugin. Using Motion Tracking on single object or several objects, the flow model is integrated with ONNX. 

## Calibration of the Model

Parameters:

- Camera View Config for each Person
- Calibration of 3D space over 2D plane, using Homography and Camera Attributes
- Measurement of Distance of Person from Camera
- A MaxMin Selection of A Single Non-Oscillating point

## Commands executed

python convert.py "models/optical_flow.onnx"

/usr/bin/python3.6 /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py --input_model models/optical_flow.onnx --log_level=DEBUG --input_shape "[1,1,240,320]" --output_dir models/