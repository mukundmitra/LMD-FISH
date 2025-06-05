
# LMD-FISH: Language Model Driven - Framework for Intelligent Scheduling of Heterogenous Systems

This repository implements a framework for real-time object detection, multi-agent task planning planning using LLMs, and robotic manipulation with XArm.

## Directory Structure

```
.
├── main.py                 # Main script
├── config.py               # Configuration and constants
├── utils/
│   ├── vision_utils.py     # PCA, rotation, vector alignment
│   ├── homography_utils.py # Coordinate transforms and homography
│   ├── robot_utils.py      # XArm controller class
│   └── ollama_utils.py     # Prompt creation and LLM response parsing
```

## Requirements
- Download 'sam2.1_b.pt'
- Python 3.8+
- PyTorch
- OpenCV
- scikit-learn
- scipy
- ultralytics (for SAM)
- xArm Python SDK
- ollama (for LLM queries)

## Quick Start

```bash
python main.py
```

This will initiate the robot planning and execution pipeline including detection, segmentation, LLM interaction, and object pick place.

# Citation
If you use this code or build upon this work, please cite:

[1] Mukund Mitra, Yashaswi Sinha, Arushi Khokhar, Sairam Jinkala, and Pradipta Biswas. 2025. LMD-FISH: Language Model Driven - Framework for Intelligent Scheduling of Heterogenous Systems. In Companion Proceedings of the 30th International Conference on Intelligent User Interfaces (IUI '25 Companion). Association for Computing Machinery, New York, NY, USA, 74–77. https://doi.org/10.1145/3708557.3716333
