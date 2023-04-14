# ScratchFormer

# Remote Sensing Change Detection With Transformers Trained from Scratch
This repo contains the official **PyTorch** code for Remote Sensing Change Detection With Transformers Trained from Scratch [[Arxiv]](https://arxiv.org/). 

**Code will be released soon. Stay tuned!**

Highlights
-----------------
- **Train From Scratch** Our proposed solution for remote sensing change detection (CD) is called ScratchFormer, which utilizes a transformers-based Siamese architecture. Notably, ScratchFormer does not depend on pretrained weights or the need to train on another CD dataset.
change detection (CD).
- **Shuffled Sparse Attention:** The proposed ScratchFormer model incorporates a novel operation called shuffled sparse attention (SSA), which aims to improve the model's ability to focus on sparse informative regions that are important for the remote sensing change detection (CD) task.
- **Change-Enhanced Feature Fusion:** In addition, we present a change-enhanced feature fusion module (CEFF) that utilizes per-channel re-calibration to improve the relevant features for semantic changes, while reducing the impact of noisy features.

Visualization results of ScratchFormer
-----------------

<table>
  <tr>
    <td><img src="demo/comparison_on_DSIFN.jpg"></td>
  </tr>
  <tr>
    <td><img src="demo/comparison_on_Levir.jpg"></td>
  </tr>
</table>


### Contact

If you have any question, please feel free to contact the authors. Mustansar Fiaz: [mustansar.fiaz@mbzuai.ac.ae](mailto:mustansar.fiaz@mbzuai.ac.ae) or Mubashir Noman: [mubashir.noman@mbzuai.ac.ae](mailto:mubashir.noman@mbzuai.ac.ae).
