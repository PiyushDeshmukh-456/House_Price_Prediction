🏡 Pune House Price Prediction
Welcome to the Pune House Price Prediction Tool! This web application uses machine learning to help users estimate house prices in Pune based on various features, such as location, size, and amenities. Whether you're buying or selling property, this tool provides accurate price predictions to guide your decisions.

🚀 Features
1. User-Friendly Interface
A clean and intuitive form built with Flask for inputting house details effortlessly.

2. Dynamic Location Search
Quickly find and select desired locations within Pune through an intelligent dropdown.

3. Accurate Predictions
Powered by a robust machine learning model trained on real housing data for precise price estimates.

4. Responsive Design
Optimized for seamless use across devices, including mobile and desktop platforms.

📊 Machine Learning Model
The model is trained on an open-source dataset of Pune housing prices. It was initially built using Linear Regression but can be upgraded with advanced algorithms like Random Forests or XGBoost for better accuracy.

✨ Key Preprocessing Steps:
Converting mixed formats of square footage into a unified numeric value.
Handling missing values to ensure reliable predictions.
Encoding categorical data, such as location, for use in the model.
🧠 How It Works
1. Input Features
Users provide details such as:

Number of Bedrooms
Total Area in Square Feet
Number of Bathrooms
Number of Balconies
Location within Pune
2. Predict Home Prices
Upon submission, the app processes the input through the trained model to calculate and display the predicted house price.

3. Smart Location Search
Effortlessly find your desired location with a dynamic search field.

📷 Screenshots
Home Page:


Prediction Result:


📝 Documentation
Code Structure:
app.py: The primary Flask application file managing user interactions and routes.
train_model.py: Contains the script for training the machine learning model and saving it as model.pkl.
run.py: A lightweight script for launching the Flask application.
templates/: Folder containing HTML files for rendering web pages.
static/: Directory for CSS and JavaScript files to enhance the UI.
requirements.txt: Lists all dependencies required to run the project seamlessly.
🛠️ Installation and Usage
Prerequisites:
Python 3.8+
Flask
Pandas, Scikit-learn, NumPy
Installed packages from requirements.txt
Steps:
Clone this repository:
bash
Copy code
git clone https://github.com/your-repo/pune-house-price-prediction.git
Navigate to the project directory:
bash
Copy code
cd pune-house-price-prediction
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Train the model (optional):
bash
Copy code
python train_model.py
Run the application:
bash
Copy code
python run.py
Access the app at: http://127.0.0.1:5000
🔗 Future Enhancements
Upgrade the model to Random Forest or Gradient Boosting for better accuracy.
Add data visualization to display market trends.
Introduce authentication for user-specific data analysis.
