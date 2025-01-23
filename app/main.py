from fastapi import FastAPI
from app.routes import inference
from fastapi.middleware.cors import CORSMiddleware


origins = ["http://localhost", "http://localhost:4200", "https://dosetailor.com",]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the inference router
app.include_router(inference.router)
