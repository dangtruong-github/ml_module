# Overview
The repository contains code of the collaborative filtering for user ratings

Our plan is to update the model when an user rates a film, or modifies the rating of a film

The links to the frontend/backend repo:
https://github.com/dangtruong-github/mlops-frontend

# How to run
Step 1: Run ```git clone``` then ```pip install requirements.txt``` to install required packages

Step 2: Modify the ```.env``` file to connect to your database

Step 3: Run the following code:
```
zenml up
python run_pipeline.py
```

