import torch
from torch import nn
import tensorflow as tf
import argparse
from keras import backend as K
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb_filename", default=None, 
        help='conversion protocol buffer filename')
    parser.add_argument("--module_path", default=None, 
        help='path of module')
    parser.add_argument("--model_path", default=None, 
        help='path of the model')

    return parser.parse_args()

def create_model(args):
    import sys
    sys.path.append(args.module_path)
    import arcface
    return arcface.obtain_model(args.model_path)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=None):

    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    graph = session.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, 
        output_names, freeze_var_names)

        return frozen_graph

if __name__ == "__main__":

    args = parse_args()
    model = create_model(args)

    frozen_graph = freeze_session(K.get_session(), 
                output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, args.pb_filename, as_text=False)
