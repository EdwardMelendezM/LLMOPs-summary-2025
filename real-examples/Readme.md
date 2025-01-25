# Real-life pipeline Example
## Automation and Orchestration of a Supervised Tuning Pipeline
- Reuse an existing Kubeflow Pipeline for Parameter-Efficient Fine-Tuning (PEFT) for a foundation model from Google, called PaLM 2.
- Advantage of reusing a pipleline means you do not have to build it from scratch, you can only specify some of the parameters

```python
### these are the same 
### jsonl files from the previous lab

### time stamps have been removed so that 
### the files are consistent for all learners
TRAINING_DATA_URI = "./tune_data_stack_overflow_python_qa.jsonl" 
EVAUATION_DATA_URI = "./tune_eval_data_stack_overflow_python_qa.jsonl"  
```
- Provide the model with a version.
- Versioning model allows for:
  - Reproducibility: Reproduce your results and ensure your models perform as expected.
  - Auditing: Track changes to your models.
  - Rollbacks: Roll back to a previous version of your model.
```python
  ### path to the pipeline file to reuse
### the file is provided in your workspace as well
template_path = 'https://us-kfp.pkg.dev/ml-pipeline/\
large-language-model-pipelines/tune-large-model/v2.0.0'

import datetime
date = datetime.datetime.now().strftime("%H:%d:%m:%Y")

MODEL_NAME = f"deep-learning-ai-model-{date}"
```

- This example uses two PaLM model parameters:
  - TRAINING_STEPS: Number of training steps to use when tuning the model. For extractive QA you can set it from 100-500.
  - EVALUATION_INTERVAL: The interval determines how frequently a trained model is evaluated against the created evaluation set to assess its performance and identify issues. Default will be 20, which means after every 20 training steps, the model is evaluated on the evaluation dataset.
  
```python
TRAINING_STEPS = 200
EVALUATION_INTERVAL = 20

from utils import authenticate
credentials, PROJECT_ID = authenticate()

REGION = "us-central1"

pipeline_arguments = {
    "model_display_name": MODEL_NAME,
    "location": REGION,
    "large_model_reference": "text-bison@001",
    "project": PROJECT_ID,
    "train_steps": TRAINING_STEPS,
    "dataset_uri": TRAINING_DATA_URI,
    "evaluation_interval": EVALUATION_INTERVAL,
    "evaluation_data_uri": EVAUATION_DATA_URI,
}
```

Note: Due to classroom restrictions, the execution will not take place in this notebook. But, if you were to execute it in your own environment, the code is provided below (for the real-life example from above). Keep in mind, running this execution is time consuming and expensive:

```python
pipeline_root "./"

job = PipelineJob(
        ### path of the yaml file to execute
        template_path=template_path,
        ### name of the pipeline
        display_name=f"deep_learning_ai_pipeline-{date}",
        ### pipeline arguments (inputs)
        parameter_values=pipeline_arguments,
        ### region of execution
        location=REGION,
        ### root is where temporary files are being 
        ### stored by the execution engine
        pipeline_root=pipeline_root,
        ### enable_caching=True will save the outputs 
        ### of components for re-use, and will only re-run those
        ### components for which the code or data has changed.
        enable_caching=True,
)

### submit for execution
job.submit()

### check to see the status of the job
job.state
```

- This is how the successful execution of the job would display like:
![image1](/src/assets/img/pipeline-result.png)

- This is how the pipeline graph would look like:
![image1](/src/assets/img/pipelines-image.png)
