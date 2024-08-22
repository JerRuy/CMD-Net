import torch.nn as nn
import math

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.normal_(0, math.sqrt(2. / n))

        elif isinstance(m, nn.ConvTranspose2d):
            #print('2')
            n = 4 * m.in_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.normal_(0, math.sqrt(2. / n))
           



