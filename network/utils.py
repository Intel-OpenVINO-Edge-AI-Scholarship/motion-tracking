import torch
import pandas as pd
import numpy as np

def normalize(v):
    return v/torch.norm(v, dim=2).unsqueeze(2)

def normalize_norm(v):
    return v/torch.norm(v)

def normalize_norm_np(v):
    return v/np.linalg.norm(v)

def flatten_points(points):
    return points.view(-1, 4)

def reshape_points(height, width, points):
    other_dim = points.shape[1]
    return points.view(height,width,other_dim)

def transform_points(transform, points):
    assert points.shape[2] == 4
    height = points.shape[0]
    width = points.shape[1]
    points = flatten_points(points)
    return reshape_points(height,width,(torch.matmul(transform, points.t())).t())

def crossProduct(vect_A, vect_B, cross_P):
    cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1]
    cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2]
    cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0]
    return cross_P

def convert_view_to_df(view, columns=['shutter_open.camera.x', 'shutter_open.camera.y',
       'shutter_open.camera.z', 'shutter_open.lookat.x',
       'shutter_open.lookat.y', 'shutter_open.lookat.z',
       'shutter_open.timestamp', 'shutter_close.camera.x',
       'shutter_close.camera.y', 'shutter_close.camera.z',
       'shutter_close.lookat.x', 'shutter_close.lookat.y',
       'shutter_close.lookat.z', 'shutter_close.timestamp']):
    
    return pd.DataFrame(view.values.reshape(1,-1), columns=columns)

def build_shutter_open_view(view):
    return {
        "camera": {
            "x": view['shutter_open.camera.x'],
            "y": view['shutter_open.camera.y'],
            "z": view['shutter_open.camera.z']
        },
        "lookat": {
            "x": view['shutter_open.lookat.x'],
            "y": view['shutter_open.lookat.y'],
            "z": view['shutter_open.lookat.z']
        },
        "timestamp": view['shutter_close.timestamp']
    }

def build_shutter_close_view(view):
    return {
        "camera": {
            "x": view['shutter_close.camera.x'],
            "y": view['shutter_close.camera.y'],
            "z": view['shutter_close.camera.z']
        },
        "lookat": {
            "x": view['shutter_close.lookat.x'],
            "y": view['shutter_close.lookat.y'],
            "z": view['shutter_close.lookat.z']
        },
        "timestamp": view['shutter_close.timestamp']
    }
