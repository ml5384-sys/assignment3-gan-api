# Assignment 3: GAN for Handwritten Digit Generation

This project trains a Generative Adversarial Network (GAN) on the MNIST dataset and serves the trained generator through a FastAPI endpoint that returns generated handwritten digits as a PNG image.

Open `http://127.0.0.1:8000/docs`, select `/generate_digit`, click **Try it out**, then **Execute**. Or open `http://127.0.0.1:8000/generate_digit?num_images=16` directly in the browser.
