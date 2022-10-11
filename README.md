Bridging the Gap from Asymmetry Tricks to Decorrelation Principles in Non-contrastive Self-supervised Learning
--------------------------------------------------------------------------------------------------------------


### Pretrained Model 
<table>
  <tr>
    <th>epochs</th>
    <th>batch size</th>
    <th>acc1</th>
    <th>acc5</th>
    <th colspan="4">download</th>
  </tr>
  <tr>
    <td>100</td>
    <td>2048</td>
    <td>69.0%</td>
    <td>88.8%</td>
    <td><a href="https://drive.google.com/file/d/11Q2mW1WPPDLsUZ82jLdgpXhGWfAF33O-/view?usp=sharing">ResNet-50</a></td>
    <td><a href="https://drive.google.com/file/d/1FDI3Vj7PjiXhAdtlfXBU6T_jGyNUabPG/view?usp=sharing">full checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/13GqaD9QU9-JvrLS_PQDh6k8FEMsEoarR/view?usp=sharing">train logs</a></td>
    <td><a href="https://drive.google.com/file/d/1DTPP035AMdnqr2624Gif3825XiIDtJbc/view?usp=sharing">val logs</a></td>
  </tr>
</table>

### Pre-Training
```
python main.py /path/to/imagenet/
```

### Evaluation: Linear Classification

```
python evaluate.py /path/to/imagenet/ /path/to/checkpoint/resnet50.pth --lr-classifier 0.3
```
### Citation
'''
'''
## Acknowledgement
Our code is inherited from [Barlow Twins](https://github.com/facebookresearch/barlowtwins). We thank the authors of the open source project.