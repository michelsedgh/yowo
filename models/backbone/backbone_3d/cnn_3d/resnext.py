import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
from functools import partial

__all__ = ['resnext50', 'resnext101', 'resnet152']


model_urls = {
    "resnext50": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/resnext-50-kinetics.pth",
    "resnext101": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/resnext-101-kinetics.pth",
    "resnext152": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/resnext-152-kinetics.pth"
}



def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()

    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    zero_pads = zero_pads.to(out.data.device)
    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 cardinality=32,
                 in_channels=3):
        self.inplanes = 64
        self.in_channels = in_channels
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c2 = self.maxpool(c1)

        c2 = self.layer1(c2)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        if c5.size(2) > 1:
            c5 = torch.mean(c5, dim=2, keepdim=True)
        
        return c5.squeeze(2)


def load_weight(model, arch):
    print('Loading pretrained weight ...')
    url = model_urls[arch]
    # check
    if url is None:
        print('No pretrained weight for 3D CNN: {}'.format(arch.upper()))
        return model
        
    print('Loading 3D backbone pretrained weight: {}'.format(arch.upper()))
    # checkpoint state dict
    checkpoint = load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
    checkpoint_state_dict = checkpoint.pop('state_dict')

    # model state dict
    model_state_dict = model.state_dict()
    # reformat checkpoint_state_dict:
    new_state_dict = {}
    for k in checkpoint_state_dict.keys():
        v = checkpoint_state_dict[k]
        new_state_dict[k[7:]] = v

    # check
    for k in list(new_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(new_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                # Special handling for conv1.weight when using motion modules (4/5/6 channels)
                if k == 'conv1.weight' and shape_model[1] > 3 and shape_checkpoint[1] == 3:
                    # Copy RGB weights to first 3 channels, init extra channels with small values
                    old_weight = new_state_dict[k]  # [64, 3, 7, 7, 7]
                    new_weight = torch.zeros(shape_model)  # [64, 4/5/6, 7, 7, 7]
                    new_weight[:, :3] = old_weight
                    # Initialize extra channels with small random values (scaled from RGB mean)
                    for ch in range(3, shape_model[1]):
                        new_weight[:, ch:ch+1] = 0.1 * old_weight.mean(dim=1, keepdim=True)
                    new_state_dict[k] = new_weight
                    print(f'  ✅ Expanded conv1.weight from 3→{shape_model[1]} channels for motion module')
                else:
                    new_state_dict.pop(k)
                    # print(k)
        else:
            new_state_dict.pop(k)
            # print(k)

    model.load_state_dict(new_state_dict)
        
    return model


def resnext50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnext50')

    return model


def resnext101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnext101')

    return model


def resnext152(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnext152')

    return model


# build 3D resnet
def build_resnext_3d(model_name='resnext101', pretrained=True, use_motion_diff=False,
                     use_optical_flow=False, use_motion_enhanced=False):
    # Determine input channels based on motion module
    # motion_diff: 4ch (RGB + diff)
    # optical_flow: 5ch (RGB + flow_x + flow_y)
    # motion_enhanced: 6ch (RGB + diff + flow_x + flow_y)
    if use_motion_enhanced:
        in_channels = 6
    elif use_optical_flow:
        in_channels = 5
    elif use_motion_diff:
        in_channels = 4
    else:
        in_channels = 3
    
    if model_name == 'resnext50':
        model = resnext50(pretrained=pretrained, in_channels=in_channels)
        feats = 2048

    elif model_name == 'resnext101':
        model = resnext101(pretrained=pretrained, in_channels=in_channels)
        feats = 2048

    elif model_name == 'resnext152':
        model = resnext152(pretrained=pretrained, in_channels=in_channels)
        feats = 2048

    return model, feats


if __name__ == '__main__':
    import time
    from thop import profile

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model, feats = build_resnext_3d(model_name='resnext50', pretrained=False)
    model = model.to(device)

    x = torch.randn(1, 3, 32, 256, 256).to(device)
    # star time
    t0 = time.time()
    # inference
    outs = model(x)
    for y in outs:
        print(y.shape)
    # end time
    print('Inference time: {}'.format(time.time() - t0))

    # FLOPs & Params
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))
