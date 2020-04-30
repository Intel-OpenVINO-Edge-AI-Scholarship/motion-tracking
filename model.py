import numpy as np
from torch import nn
import torch
from network import Complex, Flow, World, Pose
import json
from network.utils import convert_view_to_df
from torch.autograd import Variable
import cv2

def create_pose_model(image_width=320, image_height=240, factor=0.001):
    pose = Pose(image_width=image_width, image_height=image_height, factor=factor)
    return pose

def create_world_model(pose):
    world_model = World(pose)
    return world_model

def create_flow_model(world):
    flow_model = Flow(world)
    return flow_model

def save_onnx_complex(flow, filename="motion_tracking.onnx"):

    print("Saving onnx file for motion tracking with optical flow model")

    heatmap = np.ones((240,320))
    image = torch.randn(240,320,3)
    view = json.load(open("plugin_config/camera.json", "r"))

    complex_model = Complex(flow)
    d = complex_model(heatmap, image, view)

    torch.onnx.export(complex_model, (heatmap, image, view), filename, 
        verbose=True, input_names=['heatmap', 'image', 'view'], output_names=['data'])

def save_onnx_flow(world, filename="optical_flow.onnx"):

    print("Saving onnx file for motion tracking with optical flow model")

    image = np.random.random((240,320,3))

    detection = torch.from_numpy(np.vstack([(1,1,1,10,20,100,300)]*100).reshape(1,1,100,7)) # 1x1x100x7

    def load_depth_map_in_m(img, width=320, height=240, factor=0.001):
        img = cv2.cvtColor((np.abs(img)/np.abs(img).max()*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (width,height))
        return torch.from_numpy(img * factor).view(1,1,height,width)

    depth_map = load_depth_map_in_m(image)

    depth_map[depth_map == 0.0] = 1000.0

    flow_model = Flow(world)
    d = flow_model(depth_map, detection)

    torch.onnx.export(flow_model, (depth_map, detection), filename, 
        verbose=True, input_names=['depth_map', 'detection'], output_names=['data'])