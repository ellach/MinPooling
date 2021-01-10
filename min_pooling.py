import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MinPooling2d(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(MinPooling2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, image):
        output_holder = []
        for k, img in enumerate(image[0]):
            offsets_image = F.unfold(img.unsqueeze(0).unsqueeze(0), kernel_size=(self.kernel_size, self.kernel_size),
                                     stride=(2, 2)).reshape((1, -1, self.kernel_size, self.kernel_size))
            shape = int(np.sqrt(offsets_image.shape[1]))
            off = offsets_image.reshape((shape, shape, self.kernel_size, self.kernel_size))
            min_values = torch.tensor(
                [[torch.min(item) for j, item in enumerate(row)] for i, row in enumerate(off)])
            output_holder.append(min_values)
        output = torch.zeros((image.shape[0], image.shape[1], len(output_holder[0]), len(output_holder[0])))
        for i, item in enumerate(output_holder):
            output[0][i] = item
        return output
