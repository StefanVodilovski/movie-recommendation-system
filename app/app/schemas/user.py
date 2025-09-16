from pydantic import BaseModel


class UserCreate(BaseModel):
    gender: str
    age: int
    occupation: int
    zip_code: str


class UserRead(BaseModel):
    id: int
    gender: str
    age: int
    occupation: int
    zip_code: str

    class Config:
        from_attributes = True
