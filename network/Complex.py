import torch
from torch import nn
import numpy as np
from .utils import *
import cv2

class Complex(nn.Module):

    def __init__(self, flow):
        super(Complex, self).__init__()
        self.flow = flow
	
    def forward(self, counter_series, image, view):

        view = convert_view_to_df(view)
    
        rgb, optical_flow_derivatives = self.flow.compute_flow(image, view)

        lab = cv2.cvtColor((rgb/rgb.max()*255).astype(np.uint8), cv2.COLOR_RGB2LAB)

        def apply_complex(a):
            return np.complex(a[0], a[1])
        def apply_complex_conjugate(a):
            return np.complex(a[0], -a[1])

        opt_der = optical_flow_derivatives.detach().numpy().astype(np.float64)
        opt_der /= np.expand_dims(counter_series,2)

        # flow derivative
        du_dv_complex_value = np.apply_along_axis(apply_complex, 2, opt_der)
        uv_complex = np.apply_along_axis(apply_complex, 2, lab[:,:,:2].detach().numpy().astype(np.float64))
        uv_complex_conjugate = np.apply_along_axis(apply_complex_conjugate, 2, lab[:,:,:2].detach().numpy().astype(np.float64))

        fl_ci = du_dv_complex_value / uv_complex_conjugate
        amplitude = fl_ci.imag
        real_part = fl_ci.real
        phase = np.angle(uv_complex)
        temporal = np.angle(fl_ci)

        return torch.from_numpy(amplitude, dtype=torch.float64), torch.from_numpy(real_part, dtype=torch.float64), \
    torch.from_numpy(phase,dtype=torch.float64), torch.from_numpy(temporal, dtype=torch.float64), \
    torch.from_numpy(optical_flow_derivatives, dtype=torch.float64), torch.from_numpy(rgb, dtype=torch.float64)