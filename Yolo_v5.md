## The Main parts of YOLO version 5

## Focus module
- Input image pixel을 가로, 세로 각각에 대해서 홀수, 짝수로 4개로 분할하여, 이것을 4개의 채널로 변형.
- From [ C, H, W]
- To [ 4*C , H / 2, W / 2 ]
```
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, 1)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
```

## Spatial Pyramid Pooling
- Half the input channel(e.g. if input channenls are 1024, hidden channels are 512 = 1024 / 2)
- [ 5 x 5 ], [ 9 x 9 ], [ 13 x 13 ]의 넓이를 가지는 MaxPooling 2d를 하여, Input feature map과 동일한 크기의 feature map을 만든다.
- 동일한 크기의 feature map을 만들기 위해서 patch size //2의 padding이 필요.
- 원래 feature map( [ 1 x 1] maxpooling 2d나 마찬가지다)까지 포함하여, 총 4개의 feature map을 channel-wise concat한다. (512 channels * 4)
```
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
```        

## Cross Stage Partial Networks
### Bottleneck CSP
- path1 : input channel을 output channel의 1/2로 [1, 1] conv, bn, act 후, Bottleneck을 지난다. 그리고 다시 [1, 1] conv.
- path2: input channel을 output channel의 1/2로 [1, 1]로 단순 conv 한다.
- path1, path2를 channel-wise concat후 bn, act후, output channel로 [1, 1] conv, bn, act한다. 
```
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(c2, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
```

### Bottleneck
- path1 : input channel을 output channel의 1/2로 [1, 1] conv, bn, act 후, output channel로 [3, 3] conv, bc, act.
- path2 : input channel 그대로 shortcut
- path1, path2를 element-wise addition
```
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
```

### Conv
```
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(Conv, self).__init__()
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # padding
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
```        
### Anchors
```
  - [116,90, 156,198, 373,326]  # P5/32
  - [30,61, 62,45, 59,119]  # P4/16
  - [10,13, 16,30, 33,23]  # P3/8
```

### Backbone
- Large
```
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
  ]
```

### Yolo v5 head
- 1/32 Large head
  - 1024 channels from Backbone SPP
  - 3 times Bottleneck CSP (1024 channels)
  - Conv [1, 1] 512 channels
   - (L-1) nearest upsample to 1/16 Mid head to (M-1)
  - (L-2) Concat with 512 channels from 1/16 Mid head from (M-4) (1024 channels)
  - 3 times Bottlneck CSP (1024 channels)
  - conv2d to 85 outputs (80 classes + 4 box points + 1 objectness)
- 1/16 Mid head
  - 512 channels from 1/16 BCSP
  - (M-1) Concat with 512 channels from 1/32 Large head(L-1) (1024 channels)
  - 3 times Bottleneck CSP ( 512 channels)
  - Conv [1, 1] 256 channels
   - (M-2) nearest upsample to 1/8 Small head (S-1)
  - (M-3) Concat with 256 channels from 1/8 Mid head (S-2)  (256 channels)
  - 3 times Bottleneck CSP ( 512 channels)
   - (M-4) Conv [512, 3, 2] to 1/32 Large head
  - conv2d to 85 outputs (80 classes + 4 box points + 1 objectness)
- 1/8 Small head
  - 256 channels from 1/8 BCSP
  - (S-1) Concat with 256 channels from 1/16 Large head (512 channels)
  - 3 times Bottleneck CSP ( 256 channels)
   - (S-2) Conv [256, 3, 2] to 1/16 Mid head
  - conv2d to 85 outputs (80 classes + 4 box points + 1 objectness) 
