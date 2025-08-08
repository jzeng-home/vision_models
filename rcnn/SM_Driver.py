from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import sagemaker

print("Running with Sagemaker.")

role = get_execution_role()
session = sagemaker.Session()

estimator = PyTorch(
    entry_point="train.py",
    source_dir=".",                   # current directory
    role=role,
    framework_version="1.13",
    py_version="py39",
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=session,
    dependencies=["requirements.txt"],
    base_job_name="jzeng-rcnn"
)

s3_input = "s3://my-sagemaker-bucket-jzeng/PennFudanPed/"
estimator.fit({"training": s3_input})