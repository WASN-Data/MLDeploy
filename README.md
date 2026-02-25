🚀 TO TEST EVERYTHING
Step 1: Start Docker Desktop

Step 2: Build and run the API
cd "c:\Users\sophi\Desktop\Ynov\Cours\Concepts_technos_IA\MLDeploy"
docker compose -f serving/docker-compose.yml up --build

Step 3: Open a NEW terminal and run the webapp
cd "c:\Users\sophi\Desktop\Ynov\Cours\Concepts_technos_IA\MLDeploy"
docker compose -f webapp/docker-compose.yml up --build

Step 4: Access the apps
API docs: http://localhost:8080/docs
Web app: http://localhost:8081

🚀 To Run the Notebook/retrain models
Install dependencies (in your venv):
pip install jupyter librosa matplotlib seaborn scikit-learn

Open and run:
cd notebooks
jupyter notebook 01_EDA_and_Model_Training.ipynb

After training - rebuild Docker containers to use the new model:
$env:PATH = "C:\Program Files\Docker\Docker\resources\bin;$env:PATH"
docker compose -f serving/docker-compose.yml down
docker compose -f serving/docker-compose.yml up --build -d

Docker commands
Quick restart (just API, keeps webapp running):
docker compose -f serving/docker-compose.yml restart

Full rebuild (after code changes):
docker compose -f serving/docker-compose.yml up --build -d


