import os
import torch
import sagemaker
from sagemaker.pytorch import PyTorch

# Detect environment
SIM_SM = os.environ.get("SM_CHANNEL_TRAINING") is None
print("Running in Local Mode" if SIM_SM else "Running in SageMaker Cloud")

# Role (unused in Local Mode)
role = "arn:aws:iam::123456789012:role/FakeRole" if SIM_SM else "arn:aws:iam::010819239854:role/service-role/AmazonSageMaker-ExecutionRole-MyRole"

# Session
session = sagemaker.local.LocalSession() if SIM_SM else sagemaker.Session()
if SIM_SM:
    session.config = {'local': {'local_code': True}}

# Instance type
instance = "local_gpu" if SIM_SM and torch.cuda.is_available() else ("local" if SIM_SM else "ml.m5.large")
 #    ml.m5.large = cheap CPU, ml.g4dn.xlarge = GPU

estimator = PyTorch(
    entry_point="train.py",
    role=role,
    framework_version="1.13",
    py_version="py39",
    instance_count=1,
    instance_type=instance,
    sagemaker_session=session,
    dependencies=["requirements.txt"]
)

# Training
if SIM_SM:
    estimator.fit({"training": "file://./data/PennFudanPed"})
else:
    s3_input = session.upload_data(path="./data", key_prefix="pytorch-demo/data")
    estimator.fit({"training": s3_input})
