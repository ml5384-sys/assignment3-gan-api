FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first (much smaller image), then the other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy the helper library, API code, and trained model weights
COPY helper_lib ./helper_lib
COPY app ./app
COPY gan_mnist.pth .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
