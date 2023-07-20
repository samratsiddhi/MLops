from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()
class Employee(BaseModel):
    name: str
    salary: int=5000

@app.get("/home")
def home():
    print("hello")
    return {"response": 200, "msg":"sucess"}

@app.get("/")
def index():
    return {"response":200, "msg":"welcome"}

@app.post("/insert")
def insert(data:Employee):
    print(data.name)
    print(data.salary)
    return {"response":200, "msg":"insert sucessful"}

if __name__=="__main__":
    uvicorn.run("app:app", host = "localhost", port = 8080, reload=True)


