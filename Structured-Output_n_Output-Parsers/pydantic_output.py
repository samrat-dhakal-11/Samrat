from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name: str="Samrat"
    roll: Optional[int] = None
    email: EmailStr
    cpga:float=Field(gt=0,lt=10,default=5,description="The cpga of the student should be between 0 and 10")

data = {'roll':'20', 'email':'samrat@lict.edu.np'} 
new_student = Student(**data)

print(new_student)
print(new_student.name)
print(new_student.roll)
print(type(new_student.roll))
print(type(new_student))
print(new_student.email)
print(type(new_student.email))
print(new_student.cpga)
print(type(new_student.cpga))

student_dict=dict(new_student)
print(student_dict)
print(type(student_dict))
print(student_dict['name'])
print(student_dict['roll'])


student_json=new_student.model_dump_json()
print(student_json)
print(type(student_json))