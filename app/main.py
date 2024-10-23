

from fastapi import FastAPI
from app.routes import inference

app = FastAPI()

# Include the inference router
app.include_router(inference.router)
