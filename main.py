from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import requests
import os
import pandas as pd
import random

app = FastAPI()

# Load the clinical dataset
clinical_data = pd.read_excel('clinical.xlsx')

# Map categorical GI values to numerical values in clinical dataset
gi_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
clinical_data['GI'] = clinical_data['GI'].map(gi_mapping)

# Fill missing values with the mean
clinical_data['GI'] = clinical_data['GI'].fillna(clinical_data['GI'].mean())

# Concatenate relevant columns into a single feature
clinical_data['features'] = clinical_data.astype(str).agg(' '.join, axis=1)

# Initialize the TF-IDF Vectorizer for clinical data
tfidf_vectorizer_clinical = TfidfVectorizer()

# Fit and transform the feature for clinical data
tfidf_matrix_clinical = tfidf_vectorizer_clinical.fit_transform(clinical_data['features'])

# Load the food dataset
food_data = pd.read_excel('Food Dataset.xlsx')

# Map categorical GI values to numerical values in food dataset
food_data['GI'] = food_data['GI'].map(gi_mapping)

# Fill missing values with an empty string
food_data['GI'] = food_data['GI'].fillna('')

# Convert 'GI' column to string type
food_data['GI'] = food_data['GI'].astype(str)

# Concatenate relevant columns into a single feature
food_data['features'] = food_data['Meal Type'] + ' ' + food_data['Preference'] + ' ' + food_data['Allergy'] + ' ' + food_data['GI']

# Initialize the TF-IDF Vectorizer for food data
tfidf_vectorizer_food = TfidfVectorizer(vocabulary=tfidf_vectorizer_clinical.vocabulary_)

# Fit and transform the feature for food data
tfidf_matrix_food = tfidf_vectorizer_food.fit_transform(food_data['features'])

# Calculate cosine similarity between clinical data and food data
cosine_similarities = cosine_similarity(tfidf_matrix_clinical, tfidf_matrix_food)

# Get indices of similar dishes for each clinical data point
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

# Function to recommend a dish based on user input
def recommend_dish(user_data: UserData):
    try:
        # Construct user input
        user_input = f"{user_data.age} {user_data.gender} {user_data.hba1c_levels} {user_data.exercise} {user_data.glucose_level} {user_data.allergy} {user_data.preference}"

        # Transform user input using TF-IDF Vectorizer
        user_tfidf = tfidf_vectorizer_clinical.transform([user_input])

        # Calculate cosine similarity between user input and all dishes in food dataset
        similarity_scores_food = cosine_similarity(user_tfidf, tfidf_matrix_food)

        # Get indices of similar dishes
        similar_indices_food = similarity_scores_food.argsort()[0][-20:][::-1]

        # Filter dishes based on user preferences and restrictions
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

        return recommended_dishes
    except Exception as e:
        return f"Error: {str(e)}"

# Function to generate a PDF of the weekly diet plan
def generate_pdf(weekly_diet_plan):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Colors for days
    colors = {'Monday': (255, 255, 0), 'Tuesday': (255, 165, 0), 'Wednesday': (255, 69, 0),
              'Thursday': (255, 0, 0), 'Friday': (128, 0, 128), 'Saturday': (0, 0, 255), 'Sunday': (0, 128, 0)}

    for day, meals in weekly_diet_plan.items():
        pdf.add_page()

        # Set background color for the day
        pdf.set_fill_color(*colors[day])

        # Add day header
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt=f"{day} Diet Plan", ln=True, align='C', fill=True)

        # Set font for dish details
        pdf.set_font("Arial", size=12)

        # Add dishes
        for meal, dish in meals.items():
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"{meal}: {dish[0]}", ln=True)
            pdf.cell(200, 10, txt=f"Carbs: {dish[1]}", ln=True)
            pdf.cell(200, 10, txt=f"Protein: {dish[2]}", ln=True)
            pdf.cell(200, 10, txt=f"Fats: {dish[3]}", ln=True)
            pdf.cell(200, 10, txt=f"Fiber: {dish[4]}", ln=True)
            pdf.cell(200, 10, txt=f"Calories: {dish[5]}", ln=True)
            # Download the image and add to PDF
            image_url = dish[6]
            print(f"Downloading image: {image_url}")
            image_path = f"image_{day}_{meal}.jpg"
            response = requests.get(image_url)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                # Calculate the x-coordinate for the image to be on the right side
                image_width = 50
                image_x = pdf.w - image_width - 10  # Adjust 10 as needed for spacing
                # Add the image to the PDF at the calculated position
                pdf.image(image_path, x=image_x, y=pdf.get_y(), w=image_width)
                os.remove(image_path)
                pdf.ln(10)
            else:
                print(f"Failed to download image: {image_url}")

    pdf_file_path = "weekly_diet_plan.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path

# Function to recommend a weekly diet plan based on user input
def recommend_weekly_diet(user_data: UserData):
    try:
        # Define the preferences and requirements for each day of the week
        weekly_plan = {
            'Monday': 'Breakfast,Lunch,Dinner',
            'Tuesday': 'Breakfast,Lunch,Dinner',
            'Wednesday': 'Breakfast,Lunch,Dinner',
            'Thursday': 'Breakfast,Lunch,Dinner',
            'Friday': 'Breakfast,Lunch,Dinner',
            'Saturday': 'Breakfast,Lunch,Dinner',
            'Sunday': 'Breakfast,Lunch,Dinner',
        }

        # Initialize an empty diet plan for the week
        weekly_diet_plan = {}

        # Generate diet plan for each day of the week
        for day, meals in weekly_plan.items():
            meals = meals.split(',')
            daily_dishes = {}
            for meal in meals:
                # Get recommendations based on user inputs for the current day and meal
                user_data.meal_type = meal
                recommended_dishes = recommend_dish(user_data)
                # Randomly select one dish from recommendations
                selected_dish = random.choice(recommended_dishes) if recommended_dishes else ['No dish found', '', '', '', '', '', '']
                daily_dishes[meal] = selected_dish

            # Add daily diet plan to the weekly diet plan
            weekly_diet_plan[day] = daily_dishes

        return weekly_diet_plan
    except Exception as e:
        return f"Error: {str(e)}"

# Define endpoint for recommending a dish
@app.post("/recommend-dish/")
async def get_dish_recommendation(user_data: UserData):
    recommended_dish = recommend_dish(user_data)
    return recommended_dish

# Define endpoint for generating the weekly diet plan PDF
@app.post("/generate-pdf/")
async def generate_weekly_diet_pdf(user_data: UserData):
    weekly_diet_plan = recommend_weekly_diet(user_data)
    pdf_file_path = generate_pdf(weekly_diet_plan)
    return {"pdf_file_path": pdf_file_path}
