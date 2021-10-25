---
date: 2021-07-11 11:35
title: "Project Fashion mmfashion"
categories: DevCourse2 DL MathJax FashionGAN_Proj
tags: DevCourse2 DL MathJax FashionGAN_Proj
## 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Get started Recommender
## Environment
- conda
    - conda
        ```
        conda create -n fashion python=3.8 
        ```
    - install pytorch
        ```
        conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
        ```
    - install mmcv
        ```
        pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu100/torch1.9.0/index.html
        ```

## Demo code
### Error
attribute predict
- FileNotFoundError: file "/DL/ProjFashion/mmfashion-master/configs/attribute_predict/global_predictor_vgg_attr.py" does not exist
    - `test_attr_predictor.py`
        - line39:
            - from: `default='configs/attribute_predict/global_predictor_vgg_attr.py'`
            - to: `default='configs/attribute_predict_coarse/global_predictor_vgg_attr.py'`

fashion compatibility and recommendation
- OSError: checkpoint/resnet18.pth is not a checkpoint file
    
    ```py
    import torch
    import torchvision

    model = torchvision.models.resnet18(pretrained=True, progress=True)
    torch.save(model.state_dict(), 'checkpoint/resnet18.pth')
    ```
- OSError: checkpoint/FashionRecommend/TypeAware/disjoint/l2_embed/epoch_16.pth is not a checkpoint file
    - `mkdir -p checkpoint/FashionRecommend/TypeAware/disjoint/l2_embed/`
    - and download weight from [here](https://github.com/open-mmlab/mmfashion/blob/master/docs/MODEL_ZOO.md)
        - This one below 👇
        - ResNet-18 	Disjoint 	fully-connected layer 	Triplet loss, Type-specific loss, Similarity loss, VSE loss 	50.4 	0.80
- `NotImplementedError: There were no tensor arguments to this function (e.g., you passed an empty list of Tensors), but no fallback function is registered for schema aten::_cat.  This usually means that this function requires a non-empty list of Tensors, or that you (the operator writer) forgot to register a fallback function.  Available functions are [CPU, CUDA, QuantizedCPU, BackendSelect, Named, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, UNKNOWN_TENSOR_TYPE_ID, AutogradMLC, AutogradHPU, AutogradNestedTensor, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, Tracer, Autocast, Batched, VmapMode].`
    - this means no data
    - they provide this code but they deleted `set2`, so use `set1` or `set3`
    ```
    python demo/test_fashion_recommender.py \
    --input_dir demo/imgs/fashion_compatibility/set2
    ```
    ```
    python demo/test_fashion_recommender.py \
    --input_dir demo/imgs/fashion_compatibility/set3
    ```

- `FileNotFoundError: [Errno 2] No such file or directory: 'data/Polyvore/polyvore_item_metadata.json'`
    - download polyvore data
    - `mkdir -p data/Polyvore/`
    - for predict, only json file [here](https://www.kaggle.com/dnepozitek/polyvore-outfits)
- `FileNotFoundError: [Errno 2] No such file or directory: 'data/Polyvore/disjoint/test.json'`
    - download polyvore data
    - `mkdir -p data/Polyvore/disjoint`
- `FileNotFoundError: [Errno 2] No such file or directory: 'data/Polyvore/disjoint/typespaces.p'`
    - duplicated
- `FileNotFoundError: [Errno 2] No such file or directory: 'data/Polyvore/disjoint/compatibility_test.txt'`
    - duplicated

- Done.

### Compatibility Score
only shows:  
`Compatibility score: 0.292`


Issue:
`You can offer a set of possible choices. And pick up the highest scored one.`
<https://github.com/open-mmlab/mmfashion/issues/64>




## BERT4REC
<https://aihub.or.kr/sites/default/files/Sample_data/%EA%B5%AC%EC%B6%95%ED%99%9C%EC%9A%A9%EA%B0%80%EC%9D%B4%EB%93%9C%EB%B6%81_2020-01/014.K_Fashion_%EC%9D%B4%EB%AF%B8%EC%A7%80_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EA%B5%AC%EC%B6%95_%EA%B0%80%EC%9D%B4%EB%93%9C%EB%9D%BC%EC%9D%B8.pdf>  
<http://dsba.korea.ac.kr/seminar/?mod=document&uid=48>  
<https://arxiv.org/abs/1904.06690>  


# Get started Segmentation
[README.md](https://github.com/open-mmlab/mmfashion/tree/master/configs/fashion_parsing_segmentation)  





# Appendix
## Reference
> <https://github.com/open-mmlab/mmfashion>  
> <https://github.com/open-mmlab/mmcv>  

