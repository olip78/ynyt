FROM python:3.10
    
ENV DASH_DEBUG_MODE=False
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y  g++ llvm lcov && apt-get install --yes gdal-bin && apt-get install --yes libgdal-dev

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

ENV PATH_DATA=./data/
ENV PATH_CONFIG=./

RUN mkdir -p /usr/src/app/
WORKDIR /usr/src/app/

COPY ./dashboard/requirements.txt /usr/src/app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt && pip install --upgrade dash flask

COPY ./dashboard/ /usr/src/app/
COPY ./data/taxi_zones/ /usr/src/app/data/taxi_zones
COPY ./config.json /usr/src/app/config.json

EXPOSE 8080

HEALTHCHECK --interval=15s --retries=2 \
  CMD curl -f 'http://localhost:8080/healthcheck'

CMD ["python", "dashboard.py"]