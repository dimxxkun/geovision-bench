# Model card

## Purpose and scope

LandCoverCNN is a small supervised baseline for assigning one of ten EuroSAT land-cover
labels to a 64 × 64 RGB image. It is a new portfolio experiment by Sanusi Abdulfatai
Oyewunmi, created with coding-agent assistance. It is not a deployed environmental
monitoring system, a pretrained foundation model, or evidence of prior employment.

## Data and split

[EuroSAT RGB v2](https://zenodo.org/records/7711810) contains 27,000 Sentinel-2-derived
RGB patches. The original dataset is described by Helber, Bischke, Dengel and Borth in
*EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover
Classification*. The [official repository](https://github.com/phelber/EuroSAT) supplies
the dataset links and licensing information. Dataset attribution and applicable
Sentinel source terms remain separate from this repository's code license.

The experiment uses 18,900 training, 4,050 validation, and 4,050 test images. Splits are
stratified by class with seed 2026. Exact file duplicates are grouped before assignment;
the downloaded archive contained no repeated file SHA-256 hashes. This audit does not
detect re-encoded duplicates or spatial neighbours.

**This is not a spatial holdout.** RGB filenames do not provide sufficient geographic
metadata for a verified region-disjoint split. Nearby patches may share landscape,
acquisition, or atmospheric characteristics. Test accuracy must not be interpreted as
accuracy on a new country, sensor, acquisition season, or operational AOI.

## Training and model selection

- A from-scratch three-stage strided convolutional network with depthwise refinement,
  batch normalization, SiLU and global average pooling.
- Fixed scaling to [-1, 1]; no dataset statistics estimated from validation or test.
- Training-only horizontal/vertical flips and transposition. No synthetic data is used
  in the published EuroSAT training run.
- AdamW, cosine learning-rate decay, gradient clipping, deterministic PyTorch algorithms,
  and explicit seeds. The lowest validation NLL chooses the saved checkpoint.
- No pretrained weights. No architecture search, test-driven tuning, or cross-validation.
- Scalar temperature selected from 201 logarithmically spaced candidates in [0.1, 10]
  plus 1.0, minimizing validation NLL. Test labels never fit calibration.
- The colour-statistics comparison is ridge regression on train-standardized RGB mean,
  standard deviation, and 2 × 2 quadrant means. Its ridge coefficient is fixed at 1.

Exact configurations and outputs live in `results/eurosat-cpu/`. Re-running on another
PyTorch build or processor can produce different floating-point results despite fixed
seeds. The checkpoint is for inference, not exact training resumption; optimizer/RNG
resume state is not advertised.

## Evaluation

Accuracy, macro F1, per-class precision/recall, a confusion matrix, NLL, multiclass Brier
score, and 15-bin equal-width expected calibration error are reported. ECE is a summary
of this sample and binning scheme, not a guarantee of calibrated probabilities under
distribution shift. Empty reliability bins are retained with null statistics.

`confident_errors.json` identifies the 24 most confidently wrong test predictions.
It records original relative paths, expected labels and predicted probabilities, so
the errors can be inspected after downloading the dataset. Dataset images and model
weights are not checked into Git.

## Known limitations

- Single seed, single image split, CPU-only local validation; no geographic transfer study.
- RGB discards the multispectral information available in the original 13-band product.
- One label per patch hides mixed land use and does not produce segmentation masks.
- No cloud mask, out-of-distribution rejection, uncertainty interval, or abstention policy.
- Not validated for safety-critical, defence, regulatory, cadastral or property decisions.
- The test split is used once for the declared CNN evaluation and fixed baseline report;
  further model development should use validation data and a fresh final holdout.

## Sensible next experiments

Recover acquisition/scene metadata to establish spatially disjoint evaluation; report
multiple seeds and bootstrap intervals; compare with a pretrained backbone under an
equal compute budget; evaluate multispectral inputs and geographic domain shift.
These are proposed experiments, not completed results.
