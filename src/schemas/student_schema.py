from pydantic import BaseModel,Field
from typing import Literal, Annotated




class InputData(BaseModel):
    gender: Annotated[Literal["male", "female"] ,Field(..., example="male")]

    race_ethnicity: Annotated[Literal[
        "group A", "group B", "group C", "group D", "group E"],Field(...,example="group A")]

    parental_level_of_education: Annotated[str ,Field(..., example="bachelor's degree")]

    lunch: Annotated[Literal["standard", "free/reduced"] ,Field(..., example="standard")]

    test_preparation_course: Annotated[Literal["none", "completed"] , Field(..., example="completed")]

    reading_score: Annotated[int,Field(..., example=72, ge=0, le=100)]

    writing_score: Annotated[int ,Field(..., example=74, ge=0, le=100)]

