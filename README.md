
# Article and Blog Generation System with Llama2

Welcome to the Article and Blog Generation System with Llama2 project! This project focuses on generating high-quality articles and blogs using the Llama2 model.

## Introduction

This system generates articles and blog posts based on provided titles using the Llama2 model. The model is fine-tuned to create coherent and engaging content.

## Dataset

For this project, we will use a custom dataset of article titles. You can create your own dataset and place it in the `data/sample_titles.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets
- Pandas

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/asimsultan/article_blog_generation.git
cd article_blog_generation

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes article titles. Place these files in the data/ directory.
# The data should be in a CSV file with one column: title.

# To fine-tune the Llama2 model for article generation, run the following command:
python scripts/train.py --data_path data/sample_titles.csv

# To evaluate the performance of the fine-tuned model, run:
python scripts/evaluate.py --model_path models/ --data_path data/sample_titles.csv
