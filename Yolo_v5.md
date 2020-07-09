# Anatomy of YOLO v5
## Repository URL
https://github.com/ultralytics/yolov5/

## COCO Performance
<img src="https://user-images.githubusercontent.com/26833433/85340570-30360a80-b49b-11ea-87cf-bdf33d53ae15.png">
<img src="https://user-images.githubusercontent.com/30591790/86998538-91741280-c1eb-11ea-86a7-98d5239df6b4.PNG">


| Model | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>50</sub> | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5s](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)    | 36.6     | 36.6     | 55.8     | **2.1ms** | **476** || 7.5M   | 13.2B
| [YOLOv5m](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)    | 43.4     | 43.4     | 62.4     | 3.0ms     | 333     || 21.8M  | 39.4B
| [YOLOv5l](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)    | 46.6     | 46.7     | 65.4     | 3.9ms     | 256     || 47.8M  | 88.1B
| [YOLOv5x](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)    | **48.4** | **48.4** | **66.9** | 6.1ms     | 164     || 89.0M  | 166.4B
| [YOLOv3-SPP](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)  | 45.6     | 45.5     | 65.2     | 4.5ms     | 222     || 63.0M  | 118.0B

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

### Yolo v5 head description
<img src="https://user-images.githubusercontent.com/30591790/87012652-84175200-c204-11ea-9ad4-d0b471d0ebda.jpeg">

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

```
head:
  [[-1, 3, BottleneckCSP, [1024, False]],  # 9

   [-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1]],  # 18 (P3/8-small)

   [-2, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1]],  # 22 (P4/16-medium)

   [-2, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1]],  # 26 (P5/32-large)

   [[], 1, Detect, [nc, anchors]],  # Detect(P5, P4, P3)
  ]
```  

### GIOU
- box loss : 1.0 - GIOU
- hyper parameters
```
       'giou': 0.05,  # giou loss gain
       'cls': 0.58,  # cls loss gain
       'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
```
```
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
```

## Dataset Processing and Training

### Training
- Learning rate decay
 - base LR 0.01
 - cosine decay to 300 epochs
- weight decay : 5e-4
- SGD momentum : 0.937

### Image to Float
```
imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
```

### Speed Up
 - Rectangular Training : 비슷한 aspect ratio를 가진 image들을 같은 batch에 묶어서 공통 영역만을 receptive field에 넣어서 이미지 면적을 줄인다.
```
        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride
```

### Mosaic Augmentation
  - sample을 4개씩 모아서 1/4 영역씩 담아서 Training
  - a : 잘라서 붙이게 될 target 영역
  - b : 잘라올 원본 image source 영역
```
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b
  ```

### Random Affine Augmentation
  - Rotation and Scale(R, 회전 및 확대, 축소), Translation(T, 수평이동), Shear(S, 기울임)
  - hyper parameters :
```
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)
```

```
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[1] + border[1]  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[0] + border[0]  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
```

### HSV Augmentation
  - color augmentation
  - hyper parameters :
```
       'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
```
```
def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
```    


 
 
