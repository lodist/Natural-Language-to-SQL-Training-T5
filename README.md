# Natural Language to SQL Training with T5


## Disclaimer
This code was specifically designed to demonstrate the training process for converting natural language (primarily German) into SQL queries. While the provided dataset and training process are simplified, the project can be expanded to accommodate more complex SQL queries and use cases. Users are encouraged to adapt and extend the code to suit their needs.

## Overview
This repository demonstrates the training process for a T5 model to translate natural language (primarily German) into SQL-like conditions. The project uses a fine-tuned t5-base model to recognize the WHERE clause of SQL queries, showcasing the power and flexibility of transfer learning.

The repository includes:
- A Jupyter Notebook for training and evaluation.
- A sample dataset (Sample_Training_Data.csv) with simplified input-output pairs.

## Purpose
The goal of this project is to provide an example of how natural language can be translated into SQL conditions using machine learning. This project focuses on generating SQL-like WHERE clauses (excluding the WHERE keyword itself). The approach can be scaled to handle more complex queries with additional training data and examples.

## Real-World Dataset
The real dataset used in the full project includes:

10,000+ examples distributed across 60–80 parameters.
Queries ranging from simple conditions to complex logic involving multiple parameters.
With as few as 2,000 examples, the fine-tuned model started delivering reliable results for generating SQL-like WHERE conditions. However, the more complex the SQL structure, the larger the training dataset and examples required.

## Dataset

### Sample Dataset (`Sample_Training_Data.csv`)

The sample dataset provided in this repository demonstrates the structure of the input-output pairs used for training. Each row contains:

- **Input**: A natural language command (e.g., `"Einschluss: Ist Mitglied"`).
- **Value**: The corresponding SQL condition (e.g., `"IsMember = true"`).

Sample rows:

| **Input**                                      | **Value**                                                              |
|-----------------------------------------------|------------------------------------------------------------------------|
| Einschluss: Ist Mitglied                      | IsMember = true                                                       |
| Ausschluss: Kampagne 23987                    | listCampaigns in ('23987')                                            |
| Einschluss: Wohnhaft im Kanton ZH oder WT     | addressForCampaign_canton in ('ZH','WT')                              |
| Einschluss: Alter 26-75                       | (DATEDIFF(CURDATE(), birthdate) / 365.25) BETWEEN 26 AND 75           |

This dataset focuses on training the model to generate SQL conditions for filtering and segmenting data.

## Training Process
The training is performed using the t5-base model from Hugging Face. Key steps include:

1.Dataset Preparation:
- Preprocess the dataset into Input (natural language) and Value (SQL condition).
- Tokenize using the Hugging Face T5 tokenizer.

2. Fine-Tuning:
- Fine-tune the model using the Hugging Face Trainer API with the following configurations:
    - Batch Size: 8
    - Epochs: 30
    - Learning Rate: 3e-4

3. Evaluation:
- Validate the model's performance on unseen inputs to measure accuracy and token-level correctness.

4. Model Outputs:
- Translate unseen natural language inputs into accurate SQL-like conditions.

## Files
1. Train_T5_HF.ipynb: A Jupyter Notebook demonstrating the training process with the Hugging Face Transformers library.
2. Sample_Training_Data.csv: A sample dataset showcasing input-output pairs for training.


## Usage
## Running the Notebook

1. Clone the repository:

```bash
git clone https://github.com/lodist/Natural-Language-to-SQL-Training-T5.git
```

2. Install the required libraries:
   
```bash
pip install transformers datasets pandas
```

3. Open and run the notebook:

```bash
jupyter notebook Train_T5_HF.ipynb
```

## Scalability and Limitations
This project focuses on generating WHERE clauses for SQL queries.
Expanding the project to cover complete SQL queries, including SELECT, JOIN, and GROUP BY, would require significantly larger datasets and additional training.
The complexity of the SQL conditions is directly proportional to the number of training examples required.

## Results

### Evaluation Metrics:
- **Exact Match (EM):** 89.04%
- **Average BLEU Score:** 58.57%
- **ROUGE Scores:**
  - **ROUGE-1:** 0.9636
  - **ROUGE-2:** 0.9476
  - **ROUGE-L:** 0.9635
- **Perplexity:** 1.1105

### Summary:
The model demonstrates strong performance across all metrics, with ROUGE scores and Perplexity being production-ready. 
Improvements in Exact Match (EM) and BLEU Score are minimal, and the model is on the verge of meeting production-grade standards. 
A final fine-tuning or dataset refinement could push these metrics to fully meet production requirements.


## License
This repository is licensed under the MIT License. See LICENSE for details.

## Acknowledgments
Hugging Face Transformers for providing the pre-trained T5 model.
The machine learning community for advancing research in NLP and SQL generation.

Let me know if you need further adjustments or additional details! 
