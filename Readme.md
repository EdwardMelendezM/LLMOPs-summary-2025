# MLLOps (Google summary)

ML Ops is an ML engineering culture and practice that aims at unifying ML development (Dev) and ML Operation (Ops)

Automation and Monitoring at allsteps of ML system construction, including:
- Integration
- Testing
- Releasing
- Deployment
- Infrastructure management

## MLOps framework
1. Data Ingestion
2. Data Validation
3. Data Transformacion
4. Model
5. Model Analysis
6. Serving
7. Logging

# MLOps Vs System Design for LLMs
## MLOps for LLMs (LLMOps)
Focus on the LLM development and managing the model in production.
- Experiment on foundation models
- Prompt design and managment
- Supervised tuning
- Monitoring
- Evaluate generative output

## LLM System design
Broader design of the entire end to end application (front-end, back-end, data engineering etc.)
- Chain multiple LLM together
- Grounding
- Track history

## High level example of a LLM driven application
![Image1](/src/assets/img/example-of-llm-driven-application.png)

# LLMOps Pipeline

1. Data preparation and versioning
2. Pipeline design (Supervised tuning)
3. Configuration and Workflow (Artifact)
4. Pipeline Execution
5. Deploy LLM
6. Prompting and Predictions
7. Resposible IA

Where:
- Orchestration are 1 and 2 steps
- Automation are 4 and 5 steps
- This is a simplified diagram

![Image2](/src/assets/img/simplified-diagram.png)

# Data Preparation

- We need to install all this SDK
```python
import vertexai
from utils import authenticate
from google.cloud import bigquery

credentials, PROJECT_ID = authenticate()
REGION = "us-central1"

vertexai.init(project = PROJECT_ID,
              location = REGION,
              credentials = credentials)

bq_client = bigquery.Client(project=PROJECT_ID,
                            credentials = credentials)
```

- We can do many querys with bq_client
```python
QUERY_TABLES = """
SELECT
  table_name
FROM
  `bigquery-public-data.stackoverflow.INFORMATION_SCHEMA.TABLES`
"""

query_job = bq_client.query(QUERY_TABLES)

for row in query_job:
    for value in row.values():
        print(value)
```

- Data retrieval
```python
import pandas as pd

INSPECT_QUERY = """
SELECT
    *
FROM
    `bigquery-public-data.stackoverflow.posts_questions`
LIMIT 3
"""
query_job = bq_client.query(INSPECT_QUERY)

stack_overflow_df = query_job\
    .result()\
    .to_arrow()\
    .to_pandas()
stack_overflow_df.head()
```

- Dealing with large databasets
```python
QUERY_ALL = """
SELECT
    *
FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
"""
query_job = bq_client.query(QUERY_ALL)

try:
    stack_overflow_df = query_job\
    .result()\
    .to_arrow()\
    .to_pandas()
except Exception as e:
    print('The DataFrame is too large to load into memory.', e)
```

- Joining tables and query optimization
```python
QUERY = """
SELECT
    CONCAT(q.title, q.body) as input_text,
    a.body AS output_text
FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
JOIN
    `bigquery-public-data.stackoverflow.posts_answers` a
ON
    q.accepted_answer_id = a.id
WHERE
    q.accepted_answer_id IS NOT NULL AND
    REGEXP_CONTAINS(q.tags, "python") AND
    a.creation_date >= "2020-01-01"
LIMIT
    10000
"""

query_job = bq_client.query(QUERY)
### this may take some seconds to run
stack_overflow_df = query_job.result()\
                        .to_arrow()\
                        .to_pandas()

stack_overflow_df.head(2)
```

Adding instructions
- Instructions for LLMs have been shown to improve model performance and generalization to unseen tasks (Google, 2022).
- Wihtout the instruction, it is only question and answer. Model might not understand what to do.
- With the instructions, the model gets a guideline as to what task to perform.
```python
INSTRUCTION_TEMPLATE = f"""\
Please answer the following Stackoverflow question on Python. \
Answer it like you are a developer answering Stackoverflow questions.

Stackoverflow question:
"""
```

## Dataset for Tuning
- Divide the data into a training and evaluation. By default, 80/20 split is used.
- This (80/20 split) allows for more data to be used for tuning. The evaluation split is used as unseen data during tuning to evaluate performance.
- The random_state parameter is used to ensure random sampling for a fair comparison.
```python
from sklearn.model_selection import train_test_split

train, evaluation = train_test_split(
    stack_overflow_df,
    ### test_size=0.2 means 20% for evaluation
    ### which then makes train set to be of 80%
    test_size=0.2,
    random_state=42
)
```

### Diferent Datasets and Flow
- Versioning data is important.
- It allows for reproducibility, traceability, and maintainability of machine learning models.
- Get the timestamp.
```python
import datetime

date = datetime.datetime.now().strftime("%H:%d:%m:%Y")

cols = ['input_text_instruct','output_text']
tune_jsonl = train[cols].to_json(orient="records", lines=True)

training_data_filename = f"tune_data_stack_overflow_\
                            python_qa-{date}.jsonl"

with open(training_data_filename, "w") as f:
    f.write(tune_jsonl)
```

# Automation and Orchestration With Pipelines
We will use Kubeflow pipelines to orchestrat and automate a workflow.

```python
from kfp import dsl
from kfp import compiler

# Ignore FutureWarnings in kfp
import warnings
warnings.filterwarnings("ignore", 
                        category=FutureWarning, 
                        module='kfp.*')
```

## Kubeflow pipelines
Kubeflow pipelines consist of two key concepts: Components and pipelines.

- Simple pipeline example
```python
### Simple example: component 1
@dsl.component
def say_hello(name: str) -> str:
    hello_text = f'Hello, {name}!'
    
    return hello_text
```
- Second simple pipeline example
```python
### Simple example: component 2
@dsl.component
def how_are_you(hello_text: str) -> str:
    
    how_are_you = f"{hello_text}. How are you?"
    
    return how_are_you
```
- Combine simple pipeline example
```python
### Simple example: pipeline
@dsl.pipeline
def hello_pipeline(recipient: str) -> str:
    
    # notice, just recipient and not recipient.output
    hello_task = say_hello(name=recipient)
    
    # notice .output
    how_task = how_are_you(hello_text=hello_task.output)
    
    # notice .output
    return how_task.output 
```

- Implement the pipeline
```python
compiler.Compiler().compile(hello_pipeline, 'pipeline.yaml')
```

- Define the arguments
```javascript
pipeline_arguments = {
    "recipient": "World!",
}
```

- View the pipeline.yaml
```
!cat pipeline.yaml
```

- Import pipeline job
```python
### import `PipelineJob` 
from google.cloud.aiplatform import PipelineJob

job = PipelineJob(
        ### path of the yaml file to execute
        template_path="pipeline.yaml",
        ### name of the pipeline
        display_name=f"deep_learning_ai_pipeline",
        ### pipeline arguments (inputs)
        ### {"recipient": "World!"} for this example
        parameter_values=pipeline_arguments,
        ### region of execution
        location="us-central1",
        ### root is where temporary files are being 
        ### stored by the execution engine
        pipeline_root="./",
)

### submit for execution
job.submit()

### check to see the status of the job
job.state
```

- We can see how it works  
- ![image3](/src/assets/img/job-worked.png)

You can see a real example in the course


# Predictions, Prompts and Safety
## Predictions
- We can test a prediction
```python
import random
import vertexai
from vertexai.language_models import TextGenerationModel

vertexai.init(project = PROJECT_ID,
              location = REGION,
              credentials = credentials)
model = TextGenerationModel.from_pretrained("text-bison@001")
list_tuned_models = model.list_tuned_model_names()
tuned_model_select = random.choice(list_tuned_models)
deployed_model = TextGenerationModel.get_tuned_model\
(tuned_model_select)

PROMPT = "How can I load a csv file using Pandas?"
### depending on the latency of your prompt
### it can take some time to load
response = deployed_model.predict(PROMPT)
print(response)
```

- We can use format for any response
```python
from pprint import pprint

### load the first object of the response
output = response._prediction_response[0]
### print the first object of the response
pprint(output)
```

## Prompt Management and Templates
- Remember that the model was trained on data that had an Instruction and a Question as a Prompt (Lesson 2).
- In the example above, only a Question as a Prompt was used for a response.
- It is important for the production data to be the same as the training data. Difference in data can effect the model performance.
- Add the same Instruction as it was used for training data, and combine it with a Question to be used as a Prompt.

```python
INSTRUCTION = """\
Please answer the following Stackoverflow question on Python.\
Answer it like\
you are a developer answering Stackoverflow questions.\
Question:
"""

QUESTION = "How can I store my TensorFlow checkpoint on\
Google Cloud Storage? Python example?"

PROMPT = f"""
{INSTRUCTION} {QUESTION}
"""

final_response = deployed_model.predict(PROMPT)
output = final_response._prediction_response[0][0]["content"]
print(output)
```

## Safety
- The reponse also includes safety scores.
- These scores can be used to make sure that the LLM's response is within the boundries of the expected behaviour.
- The first layer for this check, blocked, is by the model itself.

```python
### retrieve the "blocked" key from the 
### "safetyAttributes" of the response
blocked = response._prediction_response[0][0]\
['safetyAttributes']['blocked']

print(blocked) #false

### retrieve the "safetyAttributes" of the response
safety_attributes = response._prediction_response[0][0]\
['safetyAttributes']
```