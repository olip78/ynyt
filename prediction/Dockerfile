FROM python:3.9

ENV PYTHONUNBUFFERED 1

RUN mkdir -p /usr/src/app/

COPY ./prediction/requirements.txt /usr/src/app/
WORKDIR /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ynyt/ /usr/src/app/ynyt
COPY ./prediction/ /usr/src/app/prediction
COPY ./artifacts/ /usr/src/app/artifacts

WORKDIR ./prediction

CMD ["python", "predict.py"]