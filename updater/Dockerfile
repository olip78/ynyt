FROM python:3.10
    
ENV PYTHONUNBUFFERED 1

RUN mkdir -p /usr/src/app/
WORKDIR /usr/src/app/

COPY ./updater/requirements.txt /usr/src/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./updater/ /usr/src/app/

ENV PATH_DATA=../results/
RUN mkdir -p /usr/src/results/

EXPOSE 8081

HEALTHCHECK --interval=15s --retries=2 \
  CMD curl -f 'http://localhost:8081/healthcheck'

CMD ["python", "updater.py"]