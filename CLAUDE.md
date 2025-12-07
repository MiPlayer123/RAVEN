# CLAUDE.md - AI Assistant Guide for RAVEN

## Project Overview

RAVEN (**R**elational and **A**nalogical **V**isual r**E**aso**N**ing) is a dataset and benchmarking framework for visual reasoning research, presented at CVPR 2019. The project generates and evaluates Raven's Progressive Matrices (RPM) - visual IQ test puzzles where models must identify patterns and select the correct answer from multiple choices.

**Paper**: "RAVEN: A Dataset for Relational and Analogical Visual rEasoNing" (Zhang et al., CVPR 2019)

## Repository Structure

```
RAVEN/
├── src/
│   ├── dataset/           # Dataset generation code
│   │   ├── main.py        # Entry point for dataset generation
│   │   ├── AoT.py         # And-Or Tree (grammar) implementation
│   │   ├── Rule.py        # Rule definitions (Constant, Progression, etc.)
│   │   ├── Attribute.py   # Entity attributes (Type, Size, Color, etc.)
│   │   ├── build_tree.py  # Tree structure builders for configurations
│   │   ├── rendering.py   # Image rendering utilities
│   │   ├── sampling.py    # Attribute/rule sampling logic
│   │   ├── solver.py      # Rule-based answer solver
│   │   ├── serialize.py   # Serialization to NPZ/XML formats
│   │   ├── constraints.py # Rule constraint logic
│   │   ├── api.py         # Utility APIs (e.g., RLE decoding)
│   │   └── const.py       # Constants and configuration values
│   │
│   └── model/             # Neural network benchmarking code
│       ├── main.py        # Entry point for training/evaluation
│       ├── basic_model.py # Base model class with train/validate/test
│       ├── resnet18.py    # ResNet18 + MLP model (Resnet18_MLP)
│       ├── cnn_mlp.py     # CNN + MLP model (CNN_MLP)
│       ├── cnn_lstm.py    # CNN + LSTM model (CNN_LSTM)
│       ├── fc_tree_net.py # FC Tree Network for structure reasoning
│       ├── const/         # Model constants
│       └── utility/       # Dataset loading utilities
│
├── assets/
│   ├── embedding.npy      # Pre-trained GloVe word embeddings
│   └── README.md          # Dataset format documentation
│
├── requirements.txt       # Python dependencies
├── LICENSE               # GPL-3.0 license
└── README.md             # Project documentation
```

## Key Concepts

### And-Or Tree (AoT) Grammar

The dataset uses an attributed stochastic image grammar based on And-Or Trees:

- **Root** → **Structure** → **Component** → **Layout** → **Entity**
- **And-nodes**: All children must be included
- **Or-nodes**: One child is sampled
- **Leaf nodes**: Entities with attributes (Type, Size, Color, Angle)

### Figure Configurations (7 types)

| Config Name | Description |
|-------------|-------------|
| `center_single` | Single object in center |
| `distribute_four` | 2x2 grid layout |
| `distribute_nine` | 3x3 grid layout |
| `left_center_single_right_center_single` | Left-Right split |
| `up_center_single_down_center_single` | Up-Down split |
| `in_center_single_out_center_single` | Inner-Outer (center) |
| `in_distribute_four_out_center_single` | Inner-Outer (grid) |

### Rule Types

Rules define how attributes change across the 3x3 matrix:

| Rule | Description |
|------|-------------|
| `Constant` | Attribute remains unchanged |
| `Progression` | Attribute changes by fixed increment (+1, +2, -1, -2) |
| `Arithmetic` | Third panel = First ± Second |
| `Distribute_Three` | Three distinct values across each row |

### Attributes

- **Number**: 1-9 entities per panel
- **Position**: Entity placement on grid
- **Type**: Shape (triangle, square, pentagon, hexagon, circle)
- **Size**: 6 levels (0.4 to 0.9 scale)
- **Color**: 10 grayscale levels (0-255)
- **Angle**: 8 rotation values (-135° to 180°)

## Development Environment

### Requirements

- **Python 3.7+** (dataset generation) / Python 2.7 (model training - legacy)
- OpenCV (`opencv-contrib-python`)
- PyTorch with CUDA support (for model training)
- NumPy, SciPy, Matplotlib, Pillow, scikit-image, tqdm

### Installation

```bash
pip install -r requirements.txt
```

### Important Notes

- Dataset generation code (`src/dataset/`) is Python 3 compatible
- Model training code (`src/model/`) uses Python 2 syntax and may require porting
- CUDA is expected for model training
- The `embedding.npy` file must be placed in the dataset directory for training

## Common Commands

### Dataset Generation

```bash
# Generate dataset with default settings (20,000 samples per configuration)
python src/dataset/main.py --num-samples 20000 --save-dir /path/to/dataset

# Key arguments:
#   --num-samples: Samples per configuration (default: 20000)
#   --save-dir: Output directory
#   --seed: Random seed (default: 1234)
#   --fuse: Fuse configurations (0/1, default: 0)
#   --val: Validation set proportion (default: 2 = 20%)
#   --test: Test set proportion (default: 2 = 20%)
```

### Model Training

```bash
# Train a model
python src/model/main.py --model Resnet18_MLP --path /path/to/dataset

# Key arguments:
#   --model: Model name (Resnet18_MLP, CNN_MLP, CNN_LSTM)
#   --path: Path to dataset
#   --epochs: Training epochs (default: 200)
#   --batch_size: Batch size (default: 32)
#   --lr: Learning rate (default: 1e-4)
#   --device: CUDA device ID (default: 0)
#   --resume: Resume from checkpoint (True/False)
#   --save: Checkpoint save directory
#   --meta_alpha, --meta_beta: Auxiliary loss weights
```

## Dataset Format

### NPZ Files

Each `.npz` file contains:
- `image`: (16, 160, 160) array - 8 context panels + 8 answer choices
- `target`: Correct answer index (0-7)
- `structure`: Serialized tree structure
- `meta_matrix`: Rule encoding matrix
- `meta_target`: Bitwise-OR of meta_matrix rows
- `meta_structure`: Structure encoding

### XML Files

Companion XML files provide detailed annotations:
- Panel decomposition (Struct → Component → Layout → Entity)
- Entity attributes with value indices
- Rule groups per component

## Code Patterns

### Model Architecture Pattern

All models inherit from `BasicModel`:

```python
class CustomModel(BasicModel):
    def __init__(self, args):
        super(CustomModel, self).__init__(args)
        # Define layers
        self.optimizer = optim.Adam(self.parameters(), ...)

    def compute_loss(self, output, target, meta_target, meta_structure):
        # Return loss tensor

    def forward(self, x, embedding, indicator):
        # Return (prediction, meta_target_pred, meta_struct_pred)
```

### Rule Implementation Pattern

Rules inherit from the `Rule` base class:

```python
class CustomRule(Rule):
    def __init__(self, name, attr, param, component_idx):
        super(CustomRule, self).__init__(name, attr, param, component_idx)

    def apply_rule(self, aot, in_aot=None):
        # Transform AoT and return modified copy
        return modified_aot
```

## Testing and Validation

- The dataset includes a built-in solver (`src/dataset/solver.py`) that achieves 100% accuracy
- Human baseline: ~84% accuracy
- Best model (ResNet+DRT): ~60% accuracy

## Important Files to Understand

1. **`src/dataset/AoT.py`**: Core grammar implementation - understand the tree structure
2. **`src/dataset/Rule.py`**: Rule logic - understand how patterns are generated
3. **`src/dataset/const.py`**: All constant values and attribute ranges
4. **`src/model/basic_model.py`**: Base training loop interface
5. **`src/model/fc_tree_net.py`**: Novel DRT (Dynamic Residual Tree) module

## Conventions

### Naming Conventions

- Classes: `PascalCase` (e.g., `AoTNode`, `BasicModel`)
- Functions/methods: `snake_case` (e.g., `apply_rule`, `compute_loss`)
- Private methods: Prefixed with `_` (e.g., `_sample`, `_resample`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `IMAGE_SIZE`, `COLOR_VALUES`)

### Code Style

- Python 2.7 compatible syntax
- NumPy for array operations
- PyTorch for neural network operations
- Deep copying used extensively for tree manipulation

## Common Issues

1. **Dataset generation**: Fully compatible with Python 3.7+
2. **Model training**: Uses Python 2 syntax - may need porting for Python 3
3. **scipy.misc.imresize**: Deprecated in newer SciPy - use `skimage.transform.resize`
4. **CUDA required**: Models expect GPU acceleration

## External Resources

- [Project Page](http://wellyzhang.github.io/project/raven.html)
- [Paper PDF](http://wellyzhang.github.io/attach/cvpr19zhang.pdf)
- [WReN Implementation](https://github.com/Fen9/WReN) (alternative model)
