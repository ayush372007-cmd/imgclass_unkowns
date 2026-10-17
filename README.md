# The Model That Doesn't Know

My solution to CodeChef AIML Task 1. I trained an image classifier on 6 classes and then tested whether it can tell when an image belongs to none of them.

## Methodology

**Data.** I used CIFAR-10 (32×32 colour images, 10 classes). I loaded it from Hugging Face (`uoft-cs/cifar10`) because the torchvision download was too slow in Colab.

I split the classes into two groups:
- Known (used for training): plane, car, bird, cat, deer, dog.
- Unknown (never shown during training): frog, horse, ship, truck.

This gave me 30,000 training images, 6,000 test images of known classes and 4,000 test images of unknown classes. I gave the unknown images the label -1 just to mark them. They are never used in training.

**Preprocessing.** Every image is converted to a tensor and scaled to [0, 1], then normalised with mean 0.5 and std 0.5 so the values are in [-1, 1]. For training images only, I also added a random crop (pad by 4 pixels, then crop back to 32×32) and a random horizontal flip. This is data augmentation, which helps the model not overfit.

**Model.** I started from the CNN in the PyTorch 60-minute blitz tutorial and made it bigger:
- 3 convolution layers (32, 64 and 128 filters, 3×3), each followed by ReLU and 2×2 max pooling
- a fully connected layer with 256 units, ReLU and dropout of 0.3
- an output layer with 6 outputs, one per known class

I trained it with cross-entropy loss and the Adam optimiser (learning rate 0.001) for 15 epochs, with batch size 64, on a Colab T4 GPU.

**Detecting unknown images.** After training, I take the model's 6 output scores (logits) for each test image and turn them into one confidence score. I tried two ways:
1. **Max softmax probability (MSP):** apply softmax and take the highest probability.
2. **Energy score:** take logsumexp of the logits. Unlike softmax, this keeps the overall size of the logits, not just how they compare to each other.

For each method, I set a threshold at the 5th percentile of the known test images' scores, so 95% of known images are accepted. Any image below the threshold is rejected as "unknown".

**Metrics.**
- **AUROC:** the chance that a random known image gets a higher score than a random unknown image. 0.5 means guessing and 1.0 is perfect.
- **FPR@95:** the share of unknown images that still get accepted when 95% of known images are accepted. Lower is better.
- **Open-set accuracy:** accuracy over all 10,000 test images. A known image counts as correct if it is accepted and classified correctly. An unknown image counts as correct if it is rejected.

## Observations

**Classifier.** Final training accuracy was 0.803 and accuracy on the 6,000 known test images was 0.807. The training accuracy is slightly lower because it is measured with dropout and augmentation turned on.

**With and without rejection.**

| | AUROC | FPR@95 | Accuracy on accepted known images | Open-set accuracy |
|---|---|---|---|---|
| No rejection | - | - | 0.807 | 0.484 |
| MSP | 0.624 | 0.901 | 0.834 | 0.515 |
| Energy | 0.666 | 0.856 | 0.829 | 0.530 |

What I noticed:
- Without rejection, every unknown image gets labelled as one of the 6 known classes, so all 4,000 of them are wrong. That is why open-set accuracy is only 0.484.
- Both methods help. Energy worked better than MSP: higher AUROC (0.666 vs 0.624) and fewer unknowns accepted (85.6% vs 90.1%). It also gave the best open-set accuracy, 0.530.
- Rejecting the lowest-scoring 5% of known images raised accuracy on the images I kept from 0.807 to about 0.83. So the images the model is least confident about are more often the ones it gets wrong.
- Detection is still weak. Even with energy, about 86% of unknown images get through.
- In the score histograms (`score_hist.png`), the known and unknown distributions overlap a lot, which matches the low AUROC.

**Which unknowns get through and why.**

| Unknown class | Accepted (MSP) | Accepted (energy) | Most often predicted as |
|---|---|---|---|
| frog | 0.86 | 0.84 | cat 36%, deer 32%, bird 24% |
| horse | 0.84 | 0.85 | deer 44%, dog 24% |
| ship | 0.95 | 0.86 | plane 67%, car 17% |
| truck | 0.95 | 0.87 | car 71%, plane 17% |

- The model labels each unknown as the known class that looks most like it. Trucks become cars, ships become planes, and horses become deer or dogs. Vehicles go to vehicles and animals go to animals.
- With MSP, ships and trucks are the hardest to catch, because 95% of them are accepted. Each of them has one very similar known class, so the model is confident.
- Energy helped mainly with ships and trucks (from 0.95 to 0.86 and 0.87). For frogs and horses it made almost no difference.
- Frogs and horses get spread across several animal classes. The model seems to know it is looking at an animal but not which one.

**Average confidence (MSP).** Correct known images: 0.866. Wrong known images: 0.602. Unknown images: 0.729.

- The model is less confident when it makes a mistake on a known class, so confidence does tell me something about its own errors.
- But it is more confident on unknown images than on its own mistakes. So the threshold is better at removing wrong predictions on known classes than at catching new classes, which is why accuracy on the accepted images went up while most unknowns still got through.

**Mistakes on the known classes** (`confusion.png`, 1,000 test images per class).

| Class | plane | car | bird | cat | deer | dog |
|---|---|---|---|---|---|---|
| Accuracy | 0.888 | 0.972 | 0.712 | 0.642 | 0.865 | 0.765 |

- Cat and dog get mixed up the most. 176 cats were predicted as dog and 121 dogs as cat. Cat is the worst class at 0.642.
- 121 birds were predicted as deer. I think this is because both are often shown against grass or trees.
- Car is almost always right (0.972). Vehicles are easy to separate from animals, and most mistakes happen between similar animals. This is the same pattern I saw with the unknown classes.

**Example images** (`fail_unknown_accepted.png`, `fail_known_rejected.png`).
- The 8 unknowns my detector was most sure about are all trucks predicted as car with probability 1.00. Most are red fire trucks or pickups with wheels, windows and a shape like a car. The model is completely confident on them, so no threshold could reject them without also rejecting real cars.
- The known images that got rejected are mostly blurry or low-contrast, have a small or partly hidden object, or are mostly background (a red field, the sky, grass). A couple are unusual examples of their class, like a yellow plane photographed close up. It makes sense that the model is unsure about these.

I think detection is weak because the unknown classes look like known ones at 32×32. A truck looks like a car, a horse looks like a deer, and a ship on water looks like a plane in the sky. The model was only trained to pick between 6 classes, so nothing taught it to be unsure about something new. The table above supports this: the unknowns are almost always predicted as their closest-looking known class.

## Limitations

- **I picked the threshold on the test set.** A better way is to keep part of the training data as a validation set and choose the threshold there. AUROC doesn't depend on the threshold, so it is not affected, but FPR@95 and the open-set accuracies may be slightly optimistic.
- **Detection is weak.** With energy, 85.6% of unknown images are still accepted.
- **The unknown classes are very similar to the known ones.** This makes the task harder than detecting completely different images, and my results only cover this one split.
- **Small model, trained from scratch, one run.** I didn't set a random seed or repeat training, so the numbers would change a little if I ran it again.
- **Both methods only look at the trained model's outputs.** I didn't try methods that change how the model is trained to make it better at spotting unknowns.
- **I didn't measure calibration**, meaning whether a confidence of 0.8 really is right 80% of the time.

