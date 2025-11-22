# ARG AIRFLOW_VERSION=3.0.2
# FROM apache/airflow:${AIRFLOW_VERSION}

# # Use major.minor for constraints (avoid picking the host's PYTHON_VERSION env)
# ARG PYTHON_MAJOR_MINOR=3.12

# COPY requirements.txt /requirements.txt

# # Install only our extra deps using the official constraints to stay compatible with the base image
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r /requirements.txt \
#       --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_MAJOR_MINOR:-3.12}.txt"

FROM apache/airflow:2.10.3
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir apache-airflow[virtualenv]==2.10.3 -r /requirements.txt