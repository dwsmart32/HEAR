from pydantic import BaseModel, Field

class BenchmarkResponse(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning to reach the conclusion.")
    prediction: str = Field(description="The final short answer or label (e.g., 'Angry', 'Happy').")