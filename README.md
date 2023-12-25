# Resume Classification without OCR

A CNN Model built for classfying whether a given image is resume or not.

## Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install opendatasets.

```bash
pip install opendatasets
```

## Dataset

```python
import opendatasets as op
op.download("https://www.kaggle.com/datasets/pdavpoojan/the-rvlcdip-dataset-test")
```

The RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. There are 320,000 training images, 40,000 validation images, and 40,000 test images. The images are sized so their largest dimension does not exceed 1000 pixels

## References:
[Source Code](https://github.com/kaledhoshme123/Documents-Classification-Using-CNN/tree/main)

Adam W. Harley, A. U. (2015). Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval. Toronto, Ontario: Ryerson University.

The study, which was my main reference, includes the same process of dividing the document into four sections, but with a difference in how to collect the common features. PCA & Conca was used in the study, while it was used in the GlobalAveragePooling1D code.

Another difference is the study used multiple convolutional neural structures for each part extracted from the images of the document, while in my notebook, one convolutional neural network was used, and the network input was considered to be 4 parts representing one complete image.


# Results:

![ss1](https://github.com/harshalplus1/resume-classification-withoutOCR/assets/98384591/5233555a-8b89-46d7-a8f2-fd3703fae102)

![ss2](https://github.com/harshalplus1/resume-classification-withoutOCR/assets/98384591/a419523f-ada6-4a8f-9d43-645a8b7c2574)

![ss3](https://github.com/harshalplus1/resume-classification-withoutOCR/assets/98384591/6e3598ab-cbc4-4c51-8d2c-0a550fb4ffc7)
## Modules 

pandas, numpy, tensorflow, string, nltk, pathlib, os, cv2, matplotlib
