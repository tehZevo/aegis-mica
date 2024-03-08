FROM python:3.11
WORKDIR /app
RUN apt update
RUN apt install git -y
#break out tf; sb3 for layer caching
run pip install tensorflow stable-baselines3
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-u", "main.py"]