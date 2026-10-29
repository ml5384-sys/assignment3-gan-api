# Assignment 3: GAN for Handwritten Digit Generation

This project trains a Generative Adversarial Network (GAN) on the MNIST dataset and serves the trained generator through a FastAPI endpoint that returns generated handwritten digits as a PNG image.

## Project Structure

```
.
├── helper_lib/            # Reusable neural network library (Modules 4 & 6)
│   ├── data_loader.py     # MNIST DataLoader, pixels normalized to [-1, 1]
│   ├── model.py           # FCNN, CNN, EnhancedCNN, Generator, Discriminator, GAN
│   ├── trainer.py         # train_model (classifiers), train_gan (GAN)
│   ├── evaluator.py       # evaluate_model (loss and accuracy)
│   ├── generator.py       # generate_samples (plot generated digits)
│   └── utils.py           # save_model / load_model
├── app/
│   └── main.py            # FastAPI app with the /generate_digit endpoint
├── HW3.ipynb              # Part 1 and Part 2 notebook
├── gan_mnist.pth          # Trained GAN weights
└── requirements.txt
```

## Part 1: GAN Architecture

**Generator**
- Input: noise vector of shape (B, 100)
- Fully connected layer to 7 × 7 × 128, then reshape to (128, 7, 7)
- ConvTranspose2d: 128 → 64, kernel 4, stride 2, padding 1 → 64 × 14 × 14, followed by BatchNorm2d and ReLU
- ConvTranspose2d: 64 → 1, kernel 4, stride 2, padding 1 → 1 × 28 × 28, followed by Tanh

**Discriminator**
- Input: image of shape (1, 28, 28)
- Conv2d: 1 → 64, kernel 4, stride 2, padding 1 → 64 × 14 × 14, followed by LeakyReLU(0.2)
- Conv2d: 64 → 128, kernel 4, stride 2, padding 1 → 128 × 7 × 7, followed by BatchNorm2d and LeakyReLU(0.2)
- Flatten and Linear layer to a single output (real/fake logit)

## Part 2: Training and API

**Training**
- Dataset: MNIST training set (60,000 images), normalized to [-1, 1]
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam (lr = 0.0002, betas = (0.5, 0.999)) for both networks
- Batch size: 128
- The trained weights are saved to `gan_mnist.pth`

**API**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/generate_digit?num_images=16` | Returns a PNG grid of generated digits (1–64 images) |

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Retrain the GAN by running the training cells in `HW3.ipynb`. A trained `gan_mnist.pth` is already included.

3. Start the API from the project root:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Open `http://127.0.0.1:8000/docs`, select `/generate_digit`, click **Try it out**, then **Execute**. Or open `http://127.0.0.1:8000/generate_digit?num_images=16` directly in the browser.
