from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import requests
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# MongoDB Connection
client = MongoClient(os.getenv('MONGO_URI'))
db = client.adhd_tasks
tasks_collection = db.tasks

# Task model
class Task(BaseModel):
    title: str
    description: str
    priority: int
    due_date: str

# Add a task
@app.post("/tasks/")
async def create_task(task: Task):
    task_data = task.dict()
    task_data = task_data | await categorize_task(task)
    result = tasks_collection.insert_one(task_data)
    return {"task_id": str(result.inserted_id)}

# Get all tasks
@app.get("/tasks/")
async def get_tasks():
    tasks = list(tasks_collection.find({}, {"_id": 0}))
    return tasks

# Call DistilBERT to categorize a task
async def categorize_task(task: Task):
    model_url = "http://distilbert:8001/categorize"
    headers = {'Content-Type': 'application/json'}
    
    # Send task description to LLM
    data = {"text": task.description}
    response = requests.post(model_url, headers=headers, json=data)
    print(response)
    if response.status_code == 200:
        categories = response.json().get("categories", ["uncategorized"])
        # Optionally, update the task with the categorized result
        tasks_collection.update_one({"title": task.title}, {"$set": {"category": categories[0]}})
        return {"category": categories[0]}
    else:
        return {}
