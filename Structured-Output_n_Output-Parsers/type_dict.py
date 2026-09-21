from typing import TypedDict

class MovieReviewDict(TypedDict):
    title: str
    rating: int
    
    
new_pred: MovieReviewDict = {'title':'Ramayanan', 'rating': 10}

print(new_pred)