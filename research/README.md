# CFAR-YOLOv11 — Resource-Optimised Dark-Vessel Detection in SAR Imagery

Implementation of the proposal *"Resource Optimised Detection of Dark Vessels and prediction of vessel
navigation direction using polarimetric SAR data based on Robust CFAR algorithm and Lightweight Deep Learning
Models"*, re-based from YOLOv8 onto **YOLOv11** (the C2f block modified in the proposal is superseded by C3k2
in YOLOv11; our Dense Context-Aware block modifies C3k2 accordingly).

## 1. Motivation

Ship detection in SAR imagery faces two coupled failure modes that the stock YOLOv11n inherits:

1. **Sea-clutter and speckle false alarms / missed small vessels.** Very small vessels occupy a handful of
   pixels; their only reliable signature is *local contrast against the surrounding clutter statistics* — the
   exact quantity a CFAR detector measures and a plain convolution does not.
2. **Near-shore clutter.** Harbours, breakwaters and land scatterers destroy the homogeneity assumption of
   cell-averaging CFAR (the multi-target "masking" effect) and confuse texture-based CNN features alike.

Classical maritime pipelines therefore run a *robust* CFAR prescreener (e.g. truncated-statistics CFAR, which
censors interfering targets before estimating clutter) followed by a CNN discriminator — a two-stage design
that is accurate but wasteful. **CFAR-YOLOv11 embeds the robust CFAR prescreening philosophy inside the
network as a differentiable, learnable operator**, retaining single-pass efficiency ("resource optimised")
while gaining CFAR's clutter adaptivity.

## 2. Novel components

### 2.1 RobustCFARGate — parameterised robust CFAR as differentiable attention

`ultralytics/nn/modules/cfar.py::RobustCFARGate`

For each cell under test, background statistics are estimated over a hollow annulus (background window `k_bg`,
guard window `k_guard`, both box filters → O(1) per pixel):

```
pass 1:  μ₀, σ₀  ← annulus mean/std of x
censor:  x̃ = soft-min(x, μ₀ + t·σ₀)          # truncated statistics (TS-CFAR), robust to interferers
pass 2:  μ, σ    ← annulus mean/std of x̃
statistic: z = (x − μ) / σ
decision:  g = sigmoid(α_c · (z − τ_c))       # α, τ learnable per channel → parameterised CFAR
output:    y = x · (1 + γ_c · g)              # residual amplification of target-like responses
```

Properties: 3 parameters per channel, **zero convolutions**, fully differentiable, AMP-safe (statistics in
fp32). The learnable threshold `τ` means the false-alarm operating point is optimised end-to-end by the
detection loss instead of being hand-tuned per sea state — the "parameterized CFAR algorithms" milestone of
the proposal.

Placement (see `ultralytics/cfg/models/11/yolo11-cfar.yaml`):

- after the P2/4 stem convolution, where activations still track SAR backscatter statistics — suppresses sea
  clutter and speckle before deep feature extraction;
- on the P3/8 head branch feeding `Detect`, sharpening the small-vessel path.

### 2.2 C3k2DCA — Dense Context-Aware C3k2

`ultralytics/nn/modules/cfar.py::DCABlock, C3k2DCA`

The proposal's "modified C2f / dense context-aware module" rebuilt for YOLOv11. Each inner `DCABlock`
replaces the C3k2 Bottleneck with:

- a **multi-dilation depthwise pyramid** (3×3, dilation 1/2/3 → effective receptive fields 3/5/7) capturing
  local scattering evidence *and* the wider sea/land context needed to reject near-shore clutter, at
  depthwise cost;
- a **pointwise fusion** of the pyramid;
- a **global-context channel gate** (squeeze-excitation on the fused response) that suppresses channels
  dominated by clutter;
- a residual connection.

The CSP split/dense concatenation of the C2f/C3k2 topology is preserved, so every intermediate DCA output is
densely aggregated ("Dense"). `C3k2DCA` replaces `C3k2` only where small-ship evidence lives: backbone P2 and
P3 stages and the P3 head branch; P4/P5 stages keep stock C3k2 to hold the parameter budget.

### 2.3 Lightweight budget

Both modules are deliberately parameter-frugal (gates: 3·C params; DCA: depthwise + 1×1 only). At scale `n`
the model stays in the ~2.6–3.0 M parameter / <8 GFLOPs class of YOLOv11n, honouring the "lightweight,
resource-optimised" constraint (the exact summary is printed by `model.info()` in the notebook).

## 3. Files

| File | Purpose |
|---|---|
| `ultralytics/nn/modules/cfar.py` | RobustCFARGate, DCABlock, C3k2DCA |
| `ultralytics/cfg/models/11/yolo11-cfar.yaml` | CFAR-YOLOv11 architecture |
| `research/cfar_demo.py` | classical TS-CFAR on image chips (analysis / paper figures) |
| `research/coco_to_yolo.py` | raw HRSID (COCO JSON) → YOLO layout converter |
| `notebooks/CFAR_YOLOv11_Colab.ipynb` | end-to-end Colab: setup → data → train → evaluate → compare |

## 4. Usage

```python
from ultralytics import YOLO

model = YOLO("yolo11n-cfar.yaml")          # scale letter n/s/m/l/x selects compound scaling
model.train(data="hrsid.yaml", epochs=150, imgsz=800, batch=16)
model.val(split="test")
```

Baseline comparison (train both from scratch with identical hyperparameters and seed):

```python
YOLO("yolo11n.yaml").train(data="hrsid.yaml", epochs=150, imgsz=800, batch=16, seed=0, name="baseline")
YOLO("yolo11n-cfar.yaml").train(data="hrsid.yaml", epochs=150, imgsz=800, batch=16, seed=0, name="cfar")
```

### Ablations

- *gates only*: replace `C3k2DCA` with `C3k2` in the YAML;
- *DCA only*: delete the two `RobustCFARGate` lines (fix `Concat`/`Detect` indices accordingly);
- *gate windows*: `RobustCFARGate` YAML args are `[k_bg, k_guard]` (optionally `t_trunc`).

## 5. Mapping to the proposal

| Proposal promise | Realisation |
|---|---|
| Robust CFAR + YOLO integration (novelty claim, §3) | RobustCFARGate: differentiable truncated-statistics CFAR inside YOLOv11 |
| "Developing Parameterized CFAR algorithms" (Gantt) | learnable per-channel threshold τ, slope α, gain γ |
| Modified C2f / dense context-aware module (Fig. 3) | C3k2DCA on YOLOv11's C3k2 |
| Lightweight models, resource optimisation (§4.5) | n/s scales, parameter-frugal modules, single-pass detector |
| Small ships & cluttered background failures (Figs. 1–2) | CFAR gate on P3 head branch + multi-dilation context in DCA |
| Anchor-free detection (§4.4) | YOLOv11 anchor-free decoupled head (inherited) |

Vessel-wake detection and navigation-direction prediction (proposal objective 3) operate on the same
detector: the wake class is simply an additional dataset class, and the CFAR gate's sensitivity to extended
low-contrast structures makes it a suitable front end — planned as follow-up work on the custom dataset.
