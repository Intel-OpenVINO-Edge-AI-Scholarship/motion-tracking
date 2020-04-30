import sys
import model
    
pose = model.create_pose_model()
world = model.create_world_model(pose)
model.save_onnx_flow(world, sys.argv[1])
