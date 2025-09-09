# FROM python:3.11
#
#
# WORKDIR /code
#
#
# COPY ./requirements.txt /code/requirements.txt
#
#
# RUN pip install --no-cache-dir  -r /code/requirements.txt
#
#
# COPY ./ /code/.
#
# EXPOSE 8000
#
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#
FROM python:3.11


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir  -r /code/requirements.txt


COPY ./app /code/app


CMD ["fastapi", "run", "app/main.py", "--port", "80"]
