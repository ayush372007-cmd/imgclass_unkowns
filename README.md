# The Model That Doesn't Know: Open-Set Image Classification on CIFAR-10

CodeChef AIML recruitment, Task 1.

A small CNN learns 6 CIFAR-10 classes. It is then shown 4 classes it never saw during training, and it has to say "I don't know" to them instead of confidently guessing. I compare two ways of detecting unfamiliar inputs (max softmax probability and the energy score), measure how much they help, and analyse where and why they fail.

## Results at a glance

| Setting | AUROC ↑ | FPR@95 ↓ | Acc. on accepted known images | Open-set accuracy (all 10,000 test images) |
|---|---|---|---|---|
| No rejection (plain classifier) | n/a | 1.000 (every unknown gets a label) | 0.807 | 0.484 |
| Max softmax probability (MSP) | 0.624 | 0.901 | 0.834 | 0.515 |
| **Energy score** | **0.666** | **0.856** | **0.829** | **0.530** |

- Known-class test accuracy (6 classes, 6,000 unseen images): **0.807**. Final training accuracy: 0.803.
- Energy beats MSP on every detection metric.
- Rejection makes the model more trustworthy on what it keeps: accuracy on accepted known images rises from 0.807 to about 0.83, and open-set accuracy rises from 0.484 to 0.530.
- Detection is still weak. Even with energy, 85.6% of unknown images are accepted when 95% of known images are kept. The [failure analysis](#failure-analysis) explains why.

![Score distributions](plots/score_hist.png)

## Setup

**Google Colab (recommended, this is how the results were produced)**
1. Open `task1_open_set.ipynb` in Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all. The first cell installs `datasets` and downloads CIFAR-10 from Hugging Face. Training takes a few minutes on a T4.

**Local**
```bash
git clone <this repo>
cd <this repo>
pip install -r requirements.txt
jupyter notebook task1_open_set.ipynb
```
A GPU is optional but makes training much faster. The notebook prints `using cuda` or `using cpu`.

## Repository layout

```
task1_open_set.ipynb   # the full pipeline: data, training, uncertainty, analysis
requirements.txt
plots/
  score_hist.png             # known vs unknown score distributions (MSP and energy)
  confusion.png              # confusion matrix on the 6 known classes
  fail_confident_wrong.png   # known images classified wrongly with high confidence
  fail_unknown_accepted.png  # unknown images the detector accepted
  fail_known_rejected.png    # known images the detector wrongly rejected
```

## Implemented features

Mapped to the task's expected functionality:

| # | Requirement | What I did |
|---|---|---|
| 01 | Classification problem with ≥5 known categories | CIFAR-10, 6 known classes: plane, car, bird, cat, deer, dog |
| 02 | Dataset and preprocessing pipeline | See [Dataset and preprocessing](#dataset-and-preprocessing) |
| 03 | Train a model | 3-layer CNN trained from scratch, 15 epochs, Adam |
| 04 | Evaluate on unseen known-class images | 6,000 CIFAR-10 test images of the known classes: 0.807 accuracy |
| 05 | Introduce excluded categories | 4,000 test images of frog, horse, ship and truck, never seen in training |
| 06 | Detect uncertain or unfamiliar inputs | Max softmax probability and energy score, each with a rejection threshold |
| 07 | Compare with and without uncertainty handling | [Results table](#results-at-a-glance): open-set accuracy 0.484 → 0.530 |
| 08 | Analyse incorrect predictions and detector failures | Per-class acceptance rates, confusion matrix, example images of each failure type |

Bonus items covered: an out-of-distribution detection technique (energy score), a comparison of two uncertainty methods (MSP vs energy), and a visualisation of confidence distributions for known and unknown classes.

## Dataset and preprocessing

- **Dataset:** CIFAR-10 (60,000 colour images, 32×32, 10 classes), loaded from Hugging Face (`uoft-cs/cifar10`). I used Hugging Face because the torchvision download was too slow in Colab. The images and labels are the same.
- **Known/unknown split:**
  - Known (trained on): plane, car, bird, cat, deer, dog (original labels 0–5, unchanged as 0–5).
  - Unknown (held out entirely): frog, horse, ship, truck (original labels 6–9, mapped to −1 and never used for training).
- **Split sizes:** 30,000 training images (known only), 6,000 known test images, 4,000 unknown test images.
- **Training transforms:** random crop of 32×32 after 4-pixel padding, random horizontal flip, conversion to float in [0, 1], then normalisation with mean 0.5 and std 0.5 per channel (giving values in [−1, 1]).
- **Test transforms:** the same conversion and normalisation, with no augmentation.
- **Batch size:** 64.

## Technical approach

### Classifier
A small CNN in the style of the PyTorch 60-minute blitz, made wider and deeper:

```
conv 3→32 (3×3) → ReLU → maxpool    32×32 → 16×16
conv 32→64      → ReLU → maxpool    16×16 → 8×8
conv 64→128     → ReLU → maxpool    8×8   → 4×4
flatten (2048) → fc 256 → ReLU → dropout 0.3 → fc 6 (logits)
```
About 0.62M parameters. Trained with cross-entropy loss and Adam (learning rate 1e-3) for 15 epochs.

### Uncertainty scores
Both scores are computed from the logits of the trained classifier, so no retraining is needed. Higher means "looks like a known class".

- **Max softmax probability (MSP)** (Hendrycks & Gimpel, 2017): `max_k softmax(z)_k`. This is the baseline.
- **Energy score** (Liu et al., NeurIPS 2020): `logsumexp_k(z_k)`, which is the negative of the free energy. Softmax normalises the logits away, so a "confident" 0.9 can come from small or large logits. The energy score keeps their overall magnitude, and in practice unknown inputs tend to produce smaller logits overall.

### Rejection rule
An image is accepted (given a class label) if its score is at least a threshold, and rejected as "unknown" otherwise. The threshold is the 5th percentile of the known test images' scores, so 95% of known images are accepted.

### Metrics
- **AUROC:** the probability that a random known image scores higher than a random unknown one. 0.5 is chance and 1.0 is perfect separation. It does not depend on the threshold.
- **FPR@95:** the fraction of unknown images still accepted when 95% of known images are accepted. Lower is better.
- **Accuracy on accepted known images:** classification accuracy on only the known images that pass the threshold.
- **Open-set accuracy:** over all 10,000 test images. A known image counts as correct if it is accepted and correctly classified; an unknown image counts as correct if it is rejected.

## Failure analysis

> TODO: fill in from cell 6 and cell 7 output.

### Which unknowns slip through

| Unknown class | Accepted (MSP) | Accepted (energy) | Most often predicted as |
|---|---|---|---|
| frog | TODO | TODO | TODO |
| horse | TODO | TODO | TODO |
| ship | TODO | TODO | TODO |
| truck | TODO | TODO | TODO |

TODO: one or two sentences on the pattern, e.g. whether the unknowns that pass are the ones that look like a known class.

![Unknowns that slipped through](plots/fail_unknown_accepted.png)

### Mistakes on known classes

![Known-class confusion matrix](plots/confusion.png)

TODO: the most confused pairs from the confusion matrix.

Mean MSP: correct known TODO, wrong known TODO, unknown TODO.

TODO: one sentence on whether the model is less confident when it is wrong, and whether that gap is large enough to be useful.

![Confident mistakes](plots/fail_confident_wrong.png)

### Known images wrongly rejected

![Known images rejected as unknown](plots/fail_known_rejected.png)

TODO: what these images have in common (unusual pose, cropped, low contrast, background-dominated, etc.).

### Why detection is weak here
- **This is a near-OOD split.** The unknown classes are visually close to known ones at 32×32: truck vs car, horse vs deer, ship vs plane (sky and water backgrounds look alike). Near-OOD is much harder than far-OOD, such as telling CIFAR images from digits.
- **The model was never trained to be uncertain.** Cross-entropy on 6 classes only teaches it to separate those 6. Nothing pushes its outputs towards low confidence on other content, so an unknown image that shares features with a known class gets a confident known-class prediction.
- **The classifier itself is modest.** Vaze et al. (ICLR 2022, "Open-Set Recognition: A Good Closed-Set Classifier Is All You Need?") show that open-set detection improves strongly as closed-set accuracy improves. An 80%-accurate small CNN is expected to be a weak detector.

## Assumptions

- "Unfamiliar inputs" means images from CIFAR-10 classes excluded from training, not corrupted images or other datasets.
- Keeping 95% of known images is an acceptable operating point. This is the standard choice behind FPR@95.
- A rejected known image counts as an error in open-set accuracy, the same as a misclassified one.

## Limitations

- **The threshold is picked on the test set.** It is the 5th percentile of the known test scores. A proper setup would pick it on a held-out validation split of the training classes. The threshold-free AUROC is unaffected, but FPR@95 and the open-set accuracies are slightly optimistic.
- **Detection is weak:** FPR@95 is 0.856, so most unknowns are still accepted. The rejection mainly removes the least confident images, which helps known-class accuracy more than it catches unknowns.
- **Small CNN trained from scratch, single run, no fixed random seed.** Numbers will vary a little between runs, and I did not report variance across seeds.
- **Post-hoc scores only.** Both methods reuse a normally trained classifier. Methods that change training (such as outlier exposure) were not tried.
- **One known/unknown split.** Results depend on which classes are held out; a different split would give different numbers.
- **Calibration was not evaluated** (no reliability diagram or ECE).
- **The training accuracy is measured with dropout and augmentation switched on**, which is why it is slightly below the test accuracy.

## What I would do next

1. Pick the threshold on a validation split carved from the training data.
2. Use a stronger classifier, such as a pretrained ResNet-18 fine-tuned on the 6 classes, which should raise both accuracy and AUROC.
3. Try feature-space detectors (Mahalanobis distance, k-nearest-neighbour distance) and outlier exposure.
4. Evaluate calibration with a reliability diagram and ECE, and try temperature scaling.
5. Test robustness on corrupted images (noise, blur) and add a small Gradio demo.

## References

- Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks", ICLR 2017.
- Liu, Wang, Owens & Li, "Energy-based Out-of-distribution Detection", NeurIPS 2020.
- Vaze, Han, Vedaldi & Zisserman, "Open-Set Recognition: A Good Closed-Set Classifier Is All You Need?", ICLR 2022.
- PyTorch 60-minute blitz, "Training a Classifier" tutorial.
