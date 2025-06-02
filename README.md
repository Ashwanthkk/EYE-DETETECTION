# Facial Landmark Detection with CNN 🧠👁️

This repository provides a Convolutional Neural Network (CNN) implementation in PyTorch to detect **7 facial landmarks** (14 coordinates) from input face images. It includes a case study that investigates and explains a dataset labeling issue using mathematical plotting.

---

## 🚀 Overview

* CNN model (`EyeCnn`) that takes RGB face images and predicts (x, y) coordinates for 7 facial keypoints.
* Designed to be modular and easy to understand.
* Includes an investigation into unexpected model predictions through 2D plotting and math analysis.

---

## 💡 Key Insight

During testing, the model predicted the right eye location incorrectly:

```
Expected: ≈95,80 
Predicted: ≈0.7, 2.0
```

### 🔍 Root Cause Investigation

1. Model output was reshaped into a `7x2` matrix:

   ```
   [
    [ 92.6419,   151.0434  ],
    [120.44958,   97.5884  ],
    [146.139,     93.4562  ],
    [ 91.81932,  127.27707 ],
    [152.65062,  153.18262 ],
    [  0.7328432,  2.0465028],  # Suspected right eye
    [  1.1650194,  2.7576346]
   ]
   ```
2. The coordinates were plotted using Matplotlib.
3. An anomaly was discovered in the 6th row (assumed to be the right eye), which was far from the expected eye region.
4. After inspecting the dataset, we confirmed the issue stemmed from incorrect label values in the original dataset.

---

## 📊 Dataset and Weights


* The dataset is available from [dataset link](https://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html).

* Trained weights is not provided in this repo you can train them using train.py.

Once finished, you can easily test the model using predict.py

---

## 🚚 Dependencies

* Python 3.7+
* PyTorch
* torchvision
* NumPy
* Matplotlib
* OpenCV
* PIL
* Albumentations

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📊 Project Structure

```
.
├── model.py          # CNN model architecture
├── utils.py          # Helper functions (e.g., visualization)
├── README.md         # This file
```

---

## 🧠 Lessons Learned

* Visual debugging via graph plotting can quickly reveal hidden issues in model inference.
* Incorrect dataset annotations can produce faulty results even with a good model.
* Combining deep learning with math provides strong diagnostic capability.

---

## 📽️ LinkedIn Demonstration

This project has also been shared on LinkedIn to demonstrate how mathematical reasoning and plotting were used to debug facial landmark outputs. The demonstration includes:

* Inference matrix plot.
* Annotated graph to locate erroneous coordinates.
* Explanation of right eye landmark failure.

> Transparency and learning from mistakes helps build trust and skill in AI workflows.

---

## 💬 Contact

Connect with me on [LinkedIn](https://www.linkedin.com/in/ashwanth-kk-267151341/)


---

> If you like this project, don't forget to ⭐ the repo!

