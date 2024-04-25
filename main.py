from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI()

# Load clinical data
clinical_data = pd.read_excel('clinical.xlsx')
clinical_data['features'] = clinical_data.astype(str).agg(' '.join, axis=1)
tfidf_vectorizer_clinical = TfidfVectorizer()
tfidf_matrix_clinical = tfidf_vectorizer_clinical.fit_transform(clinical_data['features'])

# Load food data
food_data = pd.read_excel('Food Dataset.xlsx')
food_data['features'] = food_data['Meal Type'] + ' ' + food_data['Preference'] + ' ' + food_data['Allergy'] + ' ' + food_data['GI']
tfidf_vectorizer_food = TfidfVectorizer(vocabulary=tfidf_vectorizer_clinical.vocabulary_)
tfidf_matrix_food = tfidf_vectorizer_food.fit_transform(food_data['features'])

# Calculate cosine similarity
cosine_similarities = cosine_similarity(tfidf_matrix_clinical, tfidf_matrix_food)
similar_indices_clinical = [list(similarities.argsort()[-20:][::-1]) for similarities in cosine_similarities]

# Define input data schema
class UserData(BaseModel):
    age: int
    gender: int
    hba1c_levels: int
    exercise: int
    glucose_level: int
    allergy: int
    preference: int
    meal_type: int

# Define endpoint for recommending dishes
@app.post("/recommend-dishes/")
async def recommend_dishes(user_data: UserData):
    # Construct user input
    user_input = f"{user_data.age} {user_data.gender} {user_data.hba1c_levels} {user_data.exercise} {user_data.glucose_level} {user_data.allergy} {user_data.preference}"

    # Transform user input using TF-IDF Vectorizer
    user_tfidf = tfidf_vectorizer_clinical.transform([user_input])

    # Calculate cosine similarity between user input and all dishes in food dataset
    similarity_scores_food = cosine_similarity(user_tfidf, tfidf_matrix_food)

    # Get indices of similar dishes
    similar_indices_food = similarity_scores_food.argsort()[0][-20:][::-1]

    # Filter dishes based on user preferences and other factors, excluding dishes with allergens
    filtered_indices = []
    for idx in similar_indices_food:
        dish = food_data.iloc[idx]
        if (dish['Preference'] == user_data.preference and
            dish['Meal Type'] == user_data.meal_type and
            idx in similar_indices_clinical):
            # Check if the dish contains any ingredient that the user is allergic to
            if user_data.allergy == 6 or user_data.allergy not in dish['Allergy']:
                filtered_indices.append(idx)

    # If no dishes match the user's input or preferences, select random dishes from the dataset
    if not filtered_indices:
        filtered_indices = random.sample(range(len(food_data)), min(3, len(food_data)))
    else:
        filtered_indices = random.sample(filtered_indices, min(3, len(filtered_indices)))

    # Get recommendations from filtered dishes
    recommended_dishes = food_data.iloc[filtered_indices][['Dish Name', 'Carbs', 'Protein', 'Fats', 'Fiber', 'Calories', 'Image']].values.tolist()

    return {"recommended_dishes": recommended_dishes}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
