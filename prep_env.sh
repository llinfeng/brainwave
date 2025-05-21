python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Let's check if the keys are stored"
echo $OPENAI_API_KEY
