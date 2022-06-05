# Dense-Correspondence-Demo
This repo provides a demo to get dense correspondences from images for beginners, depending on the libraries of [Detectron2 and DensePose](https://github.com/facebookresearch/detectron2/blob/bb96d0b01d0605761ca182d0e3fac6ead8d8df6e/projects/DensePose/doc/GETTING_STARTED.md), thanks. 



## Visualizations for CSE

Input Image            |  CSE
:-------------------------:|:-------------------------:
![](images/image.jpg) |  ![](images/image_cse.png)
![](images/man.jpg) |  ![](images/man_cse.png)
![](images/sheep.jpg) |  ![](images/sheep_cse.png)

Note: These images are derived from [DensPose LVIS](https://www.lvisdataset.org/dataset).

## Installation

This code works well on MacOS and Linux, after following settings:
 
1. install [torch](https://pytorch.org) >= 1.9.0

2. install Detectron2 [(See Details)](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

3. install DensePose:

```bash
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
```

Note: You may uninstall cv2 before installing Detectron2, to avoid some erros.


## More

You can get more examples from [Detectron2](https://github.com/facebookresearch/detectron2).
