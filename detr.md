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

## Hungarian Matcher

## Transformer

## Transformer Input/Output Representation and Handling

### Positional Encoding/Embedding

### Query Embedding

## Backbone

## GIOU

## Comparison to the single step object detectors
