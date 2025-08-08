from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role

role = get_execution_role()

model = PyTorchModel(
    model_data='s3://sagemaker-us-west-2-010819239854/jzeng-rcnn-2025-08-07-18-20-50-001/output/model.tar.gz',  # your model path
    role=role,
    entry_point='SM_inference.py',
    source_dir='.',                      # path to inference.py
    framework_version='1.13.1',
    py_version='py39'
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'  # or 'ml.g4dn.xlarge' for GPU
)
