# ODES: Domain Adaptation with Expert Guidance for Online Medical Image Segmentation 

Code for **ODES**, accepted in MICCAI 2025 (Medical Image Computing and Computer-Assisted Intervention), Daejeon, South Korea. (under construction)

ODES is a novel framework for **online domain adaptation** in medical image segmentation that combines active learning, image pruning, and diversity-guided annotation to adapt pre-trained models to distribution shifts in real-time.

## ✨ Key Contributions

- **First AL-based online UDA framework** for medical image segmentation.
- Novel **image pruning** strategy based on batch norm divergence to select informative images.
- A **diversity-aware acquisition** function using spatial and feature-wise distances to enhance annotation utility.
- Online, source-free, and storage-free adaptation: data is processed **once per batch**.



### Dependencies

- Python 3.8+
- PyTorch >= 1.10
- OpenCV, NumPy, SciPy, scikit-learn


### Run Demo

```bash
python demo_ODES.py --cfg cfgs/example_config.yaml
```

Edit your YAML config to match your dataset path and parameters.



## 🧪 Citation

If you find this work useful, please cite:

```
@inproceedings{islam2025odes,
  title={ODES: Domain Adaptation with Expert Guidance for Online Medical Image Segmentation},
  author={Md Shazid Islam and Sayak Nag and Arindam Dutta and Sk Miraj Ahmed and Fahim Faisal Niloy and Shreyangshu Bera and Amit K. Roy-Chowdhury},
  booktitle={MICCAI},
  year={2025}
}
```


## 📬 Contact

For questions or collaborations:  
[misla048@ucr.edu](mailto:misla048@ucr.edu)

License: This project is licensed under the MIT License.
