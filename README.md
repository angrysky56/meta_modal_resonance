# Meta-Modal Resonance Theory Framework

## Overview

This repository implements a computational framework for modeling the "Meta-Modal Resonance Theory" (MMRT) of meaning-construction in human consciousness. The theory proposes that profound meaning emerges through the dynamic integration of three fundamental experiential modalities:

1. **Hedonic Processing** - Systems related to pleasure, pain, and immediate reward/threat evaluation
2. **Eudaimonic Processing** - Systems related to purpose, growth, achievement, and virtuous action
3. **Transcendent Processing** - Systems related to boundary dissolution, connection, and integration with larger wholes

The computational framework simulates how these three processing modes interact within individual agents and across multi-agent systems to create emergent meaning structures that are more robust than those produced by any single modality alone.

## Key Features

- **Multi-Modal Agent Architecture**: Agents with parallel processing across hedonic, eudaimonic, and transcendent domains
- **Oscillatory Control System**: Dynamic shifting of processing emphasis between modalities
- **Meaning Structure Construction**: Temporal integration of modal outputs into coherent structures
- **Environmental Simulation**: Variable contexts that create different demands on processing systems
- **Integration Metrics**: Quantitative measures for meaning coherence, resilience, and integration
- **Rich Visualization**: Tools for visualizing meaning structures and their properties

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/username/meta_modal_resonance.git
   cd meta_modal_resonance
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Quick Start

Run a basic experiment comparing different agent types:

```bash
python experiments/modal_comparison.py --agents-per-type 3 --duration 100 --output-dir results
```

This will:
1. Create agents of each type (hedonic optimizer, eudaimonic optimizer, transcendent optimizer, and meta-modal integrator)
2. Expose them to identical environmental sequences
3. Compare their meaning construction capabilities
4. Generate visualizations and metrics

## Repository Structure

```
meta_modal_resonance/
├── docs/                      # Theoretical documentation
│   ├── theoretical_foundation.md
│   ├── implementation_plan.md
│   └── experimental_design.md
├── src/                       # Source code
│   ├── agents/                # Agent definitions
│   │   ├── meta_modal_agent.py
│   │   └── oscillatory_controller.py
│   ├── environment/           # Environment simulation
│   │   └── context_generator.py
│   ├── metrics/               # Evaluation metrics
│   │   └── integration_metrics.py
│   └── visualization/         # Visualization tools
│       └── meaning_structure_visualizer.py
├── experiments/               # Experiment scripts
│   ├── modal_comparison.py
│   └── resilience_testing.py
└── README.md
```

## Experiments

### Modal Comparison

This experiment tests the primary hypothesis that agents capable of dynamic integration across modalities will develop more coherent and resilient meaning structures than agents primarily operating within a single modality.

```bash
python experiments/modal_comparison.py --help
```

Parameters:
- `--agents-per-type`: Number of agents per type to create (default: 3)
- `--duration`: Number of time steps to run the experiment (default: 100)
- `--complexity`: Environmental complexity level, 0.0-1.0 (default: 0.7)
- `--volatility`: Environmental volatility level, 0.0-1.0 (default: 0.5)
- `--output-dir`: Directory for result files (default: 'results')

### Perturbation Resilience

This experiment tests how different types of meaning structures respond to various perturbations, examining the hypothesis that integrated structures demonstrate greater resilience to disruption.

```bash
python experiments/resilience_testing.py --help
```

## Citation

If you use this framework in your research, please cite:

```
@misc{mmrt2025,
  author = {Author},
  title = {Meta-Modal Resonance Theory: A Computational Framework for Meaning Construction},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/username/meta_modal_resonance}}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
