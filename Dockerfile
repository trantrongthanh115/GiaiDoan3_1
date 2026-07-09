FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies for LightGBM (libgomp)
RUN dnf install -y libgomp && dnf clean all

# Copy requirements and install packages
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy main handler code
COPY lambda_deploy/app.py ${LAMBDA_TASK_ROOT}/app.py

# Copy the real model and preprocessor files
COPY lambda_deploy/lightgbm_model.pkl ${LAMBDA_TASK_ROOT}/lightgbm_model.pkl
COPY lambda_deploy/standard_scaler.pkl ${LAMBDA_TASK_ROOT}/standard_scaler.pkl
COPY lambda_deploy/label_encoder.pkl ${LAMBDA_TASK_ROOT}/label_encoder.pkl

# Set the command to run the lambda handler
CMD [ "app.lambda_handler" ]
