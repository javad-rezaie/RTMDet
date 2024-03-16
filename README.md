# RTMDet Object Detection with MMDetection and Docker

This repository contains instructions for performing object detection tasks by training an RTMDet model using MMDetection with Docker.

## Prerequisites

Before you begin, ensure you have Docker installed on your system. If not, you can install it by following the instructions [here](https://docs.docker.com/get-docker/).

## Getting Started

Clone this GitHub repository:

```bash
git clone https://github.com/javad-rezaie/RTMDet
cd RTMDet
```

## Setting Up the Docker Environment

There are two options for setting up the Docker environment:

### Option 1: Use a Pre-built MMDetection Docker Image

Run the following command to pull the pre-built MMDetection Docker image:

```bash
make docker-pull-mmdetection
```

### Option 2: Build Your Own Docker Image

Build your own Docker image with MMDetection and your project code by running:

```bash
make docker-build-mmdetection
```

## Dataset Preparation

1. Download the Kvasir-Instrument dataset from [simula](https://datasets.simula.no/kvasir-instrument/).
2. Unzip the dataset and place it in a suitable directory.
3. Convert the dataset format to COCO format by following the steps described in `Data_Preparation.ipynb` Jupyter notebook.

To run the Jupyter notebook from the terminal, execute the `jupyter.sh` script:

```bash
bash jupyter.sh
```

## Modifying the Paths and GPU Configuration

1. Update the `DATA_DIR` path inside the `train.sh` script to your appropriate local path where the dataset is located.
2. Update the `GPU` variable to the number of installed GPUs on your PC.

## Tips
Ensure that the `train.sh` and  `jupyter.sh` bash scripts has executable permissions. If not, grant execute permission by running `chmod u+x train.sh`.

# Model Conversion to OpenVINO and Hugging Face Integration
## Converting to OpenVINO
Our trained PyTorch model was converted to OpenVINO format using the Model Optimizer tool. This streamlined the deployment process for various hardware platforms.

## Hugging Face Upload
We shared the original trained PyTorch model and its converted version to OpenVINO format on the Hugging Face Model Hub, making it easily accessible for developers ([here](https://huggingface.co/spaces/homai/Kvasir-Instrument-RTMDet)). This allows for straightforward integration into applications and fine-tuning on custom datasets.

## Running on Hugging Face
Instantiating the model from its unique identifier on Hugging Face enables easy execution and result visualization. Whether through the website interface or the API, running the model is intuitive and efficient.

## Disclaimer

This project is intended for educational purposes only. It is not intended to provide medical advice or any other professional advice. Any use of this project for real-world applications should be done with caution and proper consultation with relevant experts.

## License

This project is licensed under the This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
