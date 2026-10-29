import io
from pathlib import Path

import torch
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from torchvision.utils import save_image

from helper_lib.model import get_model
from helper_lib.utils import load_model

app = FastAPI(title="GAN Digit Generator API")

# ---- Load the trained GAN once at startup ----
MODEL_PATH = Path(__file__).resolve().parent.parent / "gan_mnist.pth"
gan_model = load_model(get_model("GAN"), MODEL_PATH, device="cpu")


@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "GAN API is running. Go to /docs to try it."}


@app.get("/generate_digit")
def generate_digit(num_images: int = Query(16, ge=1, le=64)):
    """Generate handwritten digits with the GAN and return them as a PNG grid."""
    with torch.no_grad():
        z = torch.randn(num_images, gan_model.z_dim)       # random latent vectors
        imgs = gan_model.generator(z) * 0.5 + 0.5          # rescale [-1, 1] -> [0, 1]
    buf = io.BytesIO()
    save_image(imgs, buf, format="png", nrow=8)            # write image grid to memory
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
