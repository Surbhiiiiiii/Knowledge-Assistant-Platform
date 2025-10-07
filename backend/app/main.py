from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload, chat

app = FastAPI(title="IDP & Knowledge Assistant")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust for production frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(upload.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
