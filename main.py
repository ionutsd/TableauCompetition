import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query_engine import answer_question
from data_loader import load_data
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI(title="Fuel Price Chatbot API")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    load_data()
    print("Data loaded and cached.")

class QuestionRequest(BaseModel):
    question: str
    filter_years: list[int] = []


class QuestionResponse(BaseModel):
    answer: str
    engine: str
    filters: dict


@app.post("/ask", response_model=QuestionResponse)
@limiter.limit("10/minute")
def ask(request: Request, body: QuestionRequest):
    result = answer_question(
        question=body.question,
        filter_years=body.filter_years,
    )
    return QuestionResponse(**result)


@app.get("/health")
def health():
    df = load_data()
    return {
        "status": "ok",
        "rows": len(df),
        "years": f"{df['year'].min()}–{df['year'].max()}",
        "countries": df['country'].nunique(),
    }


@app.get("/")
def root():
    return {"message": "Fuel Price Chatbot API is running. POST /ask to query."}