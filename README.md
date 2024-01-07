# **RDIR**: Recurrent Detect-Infer-Repeat

Official repository for RDIR: Capturing Temporally-Invariant Representations of Multiple Objects in Videos accepted at the Winter Conference on Applications of Computer Vision (WACV) 2024 (Workshops).

Code will be released soon.

| [Paper](https://openaccess.thecvf.com/content/WACV2024W/Pretrain/papers/Zielinski_RDIR_Capturing_Temporally-Invariant_Representations_of_Multiple_Objects_in_Videos_WACVW_2024_paper.pdf) | [Supplementary](https://openaccess.thecvf.com/content/WACV2024W/Pretrain/papers/Zielinski_RDIR_Capturing_Temporally-Invariant_Representations_of_Multiple_Objects_in_Videos_WACVW_2024_paper.pdf) | [Bibtex](#citation) |
| :---: | :---: | :---: |

## Abstract

> Learning temporally coherent representations of multiple objects in videos is crucial for understanding their complex dynamics and interactions over time. In this paper, we present a deep generative neural network, which can learn such representations by leveraging pretraining. Our model builds upon a scale-invariant structured autoencoder, extending it with a convolutional recurrent module to refine the learned representations through time and enable information sharing among multiple cells in multi-scale grids. This novel approach provides a framework for learning perobject representations from a pretrained object detection model, offering the ability to infer predefined types of objects, without the need for supervision. Through a series of experiments on benchmark datasets and real-life video footage, we demonstrate the spatial and temporal coherence of the learned representations, showcasing their applicability in downstream tasks such as object tracking. We analyze the method’s robustness by conducting an ablation study, and we compare it to other methods, highlighting the importance of the quality of objects’ representations.

# Citation
```
@InProceedings{Zielinski_2024_WACV,
    author    = {Zieli\'nski, Piotr and Kajdanowicz, Tomasz},
    title     = {RDIR: Capturing Temporally-Invariant Representations of Multiple Objects in Videos},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2024},
    pages     = {597-606}
}
```
