#this is python image
FROM python:3.12.13-slim

#use uv for venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

#make a dir for my code
WORKDIR /app

#pull the requirements
COPY requirements.txt .

#To run requirements file
RUN uv pip install --system --no-cache -r requirements.txt

#copy my code
COPY . .

# expose where your app will run(port mapping)
EXPOSE 8501

#how to run my app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]






