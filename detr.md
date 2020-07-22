# Anatomy of Detection Transformer(DETR)

## COCO
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.616
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.206
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.609
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.334
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.629
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.814
Training time 12 days, 16:01:22
```

## Abstract

- DETR has three main parts, which are transformer, backbone and matcher.
- First of all, a DETR matcher is for matching between groundtruth and predictions with Hungarian Algorithm, also known as the Munkres or Kuhn-Munkres algorithm.
- Secondly, a DETR has a transformer which contains Multi-head-attention as main modules.
- Finaly, DETR use resnet-50(or resnet-101) network as a image feature extractor.
- Additionaly, to compute losses, DETR use Hungarian-matched permutation and GIOU for labels loss and boxes loss.
- DETR also some embeddings and encodings for query and postion handling in a Transformer.

## Backbone feature extractor
- Input : scale augmented from shortest 480(to 800) to longest 1333. 3 x H x W
- Output : 2048 x H / 32 x W / 32
- for example : feature resultion by layers = 640-320-160-80-40-20
- final image feature resulution is 20 x 20 (for 640 x 640 image)
```
"The way DETR achieves this is by improving AP_L (+7.8), however note that the
model is still lagging behind in APs (-5.5). DETR-DC5 with the same number
of parameters and similar FLOP count has higher AP, but is still significantly
behind in APs too." (from the paper section "4.1 Comparison with Faster R-CNN")
```

## Hungarian Matcher

```
        # models/matcher.py
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
# Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
```
- All costs are computed pairwise beteen predictions and groundtruth targets.
- e.g. N is the number of predctions. M is the number of targets.
- Then All costs matrix C is a [N, M] pairwise matrix.
- The Hungarian Algotithm computes the optimal cost and permutation over the pairwise matrix C.

### GIOU
```
# util/box_ops.py
# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
```   

## Transformer
- d_model(=hidden_dims) : 512
- nhead : 8
- num_encoder_layers : 6
- num_decoder_layers : 6
- dim_feedforward : 2048
<img src="https://user-images.githubusercontent.com/30591790/87923609-b9e7f080-cab8-11ea-932f-09187f314d7f.PNG">

```
"the decoder receives queries (initially set to zero),
output positional encoding (object queries), and encoder memory, and produces
the final set of predicted class labels and bounding boxes through multiple multihead self-attention and decoder-encoder attention."
(from the paper section "A.3 Detailed architecture")

```
```
# models/transformer.py
    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
``` 
### Transformer Encoder
 - key, query : combined the image feature sources with spatial postition embedding
 - value : source (=image features)
```
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        ...
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
```        
### Transformer Decoder
#### Multi-head self-attention
- key, query : combined the targets with query postition embedding
- value : target (= last layer output.Initially set to zero)
#### Multi-head attention
- key : memory(=encoder output) with spatial postition embedding
- query : combined the targets(= MHSA outputs) with query postition embedding
- value : memory(=encoder output)
#### versus The (original) Transformer
<img src="https://user-images.githubusercontent.com/30591790/87923658-c9673980-cab8-11ea-9e20-10b7fc1269dc.PNG">

- Output Embedding is a (masked) target sequence.
  - versus the zero initial target and the object queries embedding in the DETR.
- Output Embedding is not an input to the second MHA.
  - versus in the DETR, it is an input to the MHA(the second attention layer) as well as to the MHSA(the first attention layer).
- Memory(the output of the encoder and the input V and K) has no spatial positional encoding.
  - versus in the DETR, the key(K) is combined with the spatial positional encoding.

## Transformer Input/Output Representation and Handling
### Spatial Positional Encoding/Embedding
- Along the 2-d axis X and Y each, allocate sine values for even indices and cosine values for odd indices.
```
e.g. [X, Y]
[[sin0, sin0], [cos0.25, sin0], [sin0.5, sin0], [sin0.75, sin0], [sin1.0, sin0],
[sin0, cos0.25], [cos0.25, cos0.25], [sin0.5, cos0.25], [sin0.75, cos0.25], [sin1.0, cos0.25],
[sin0, sin0.5], [cos0.25, sin0.5], [sin0.5, sin0.5], [sin0.75, sin0.5], [sin1.0, sin0.5],
[sin0, cos0.75], [cos0.25, cos0.75], [sin0.5, cos0.75], [sin0.75, cos0.75], [sin1.0, cos0.75],
[sin0, sin1.0], [cos0.25, sin1.0], [sin0.5, sin1.0], [sin0.75, sin1.0], [sin1.0, sin1.0]]
```
```
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
...
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
```        
- pos in
```
# models/detr.py
class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
...
    def forward(self, samples: NestedTensor):
...    
        features, pos = self.backbone(samples)
...

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
```
```
# models/transformer.py
class TransformerEncoderLayer(nn.Module):
...
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
...
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
```
- pos_embed in
```
# models/transformer.py
class Transformer(nn.Module):
...
    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
```

### Query Embedding
- = Object Queries
- = Output Positional Encoding
- query_embed in 
```
# models/detr.py
class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
    ...
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
    ...
    def forward(self, samples: NestedTensor):
    ...
    hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
    ...    
```
```
# models/transformer.py
class Transformer(nn.Module):
...
    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
```
- query_pos in

```
# models/transformer.py
class TransformerDecoder(nn.Module):
....
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
```

## Input Data Prepocessing and Augmentation
```
# datasets/coco.py
def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
```

## Comparison to the other modern object detectors
- no Deconvolution
  - deconv in the Retinanet, Yolo v5, CenterNet, etc.
- no feature pyramid
  - FPN in the Retinanet, Yolo v5.
- no SPP
  - SPP in Yolo v5
- Resnet 50 basic
  - Darknet 53 in Yolo v3
  - CSP in Yolo v5
  - Resnet 101, Resnext in the Retinanet
  - Hourglass-104 in Centernet
  ```
   Darknet53 is thicker than Resnet-50 in channel width, and is normally considered as more powerful and efficient than Resnet-101.
   Cross Stage Partial Networks(CSP) is 20~30% more efficient than Resnet-50
  ```
- no anchors
  - 9 anchors in Yolo v3-5
  - 45 anchors in Retinanet (9 anchors per layers * 5 layers)
- no key points
  - center point heatmaps in CenterNet
- no affine or mosaic augmentation
  - vs. Yolo v5 which uses both affine and mosaic augmentation.
- Do the object queries replace the SPP and the anchors, and MHSA and MHA replace FPN?
  - DETR has no prior like the SPP and the anchors, but it has embedding encodings.
  - DETR has no scale-dependent features and affine and translation augmentation, but has the attention modules.
- 86 GFLOPS / 28 FPS
  - 65 FPS in Yolo V4
  - 88 GFOPS  / 50 FPS (in serial) or 256 FPS (in batch) in Yolo V5
