# Use the official Python image as a base
FROM python:3.10.16-slim

# Set environment variables
ENV POETRY_VERSION=1.6.0
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl gcc git wget \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Create and set the working directory
WORKDIR /app

COPY . /app
# Install dependencies including Jupyter Lab
RUN poetry config virtualenvs.create false && poetry install

# Copy the rest of the application code
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/site-packages/":$LD_LIBRARY_PATH

# RUN pip install ipykernel jupyterlab
RUN python -m ipykernel install --user --name=urbanity

# Expose the Jupyter Lab port
EXPOSE 8888

# Set the entry point
CMD ["poetry", "run", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
