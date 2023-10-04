# PDPR_GPT

Project Setup Guide
Follow these steps to set up and run the project:

1. Clone the Repository

git clone <repository_url>
cd <repository_directory>  # Replace <repository_directory> with the name of your cloned repository
Replace <repository_url> with the URL of your Git repository.

2. Set Up a Virtual Environment
To isolate our project dependencies, we'll create a virtual environment using virtualenv.


virtualenv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate



3. Install Dependencies
With the virtual environment activated, install the required packages using the following command:


pip install -r requirements.txt



4. Configure API Key and Environment Variables
Open the example.env file.
Replace the placeholder for the API key with your actual API key.
Rename the file from example.env to .env.



5. Run the App using Streamlit
With everything set up, you can now run the application using:


streamlit run app.py
That's it! The app should now be running in your default browser.