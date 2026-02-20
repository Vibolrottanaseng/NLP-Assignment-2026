from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from model import predict

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )

@app.post("/predict", response_class=HTMLResponse)
async def run_predict(
    request: Request,
    premise: str = Form(...),
    hypothesis: str = Form(...)
):

    label, confidence = predict(premise, hypothesis)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": label,
            "premise": premise,          
            "hypothesis": hypothesis     
        }
    )