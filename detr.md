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
## Transformer

## Transformer Input/Output Representation and Handling

### Positional Encoding/Embedding

### Query Embedding

## GIOU

## Input Data Prepocessing
### Scale Augmentation

## Comparison to the single step object detectors
- no feature pyramid
- no SPP
- Resnet 50 basic
- no key points

