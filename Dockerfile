FROM tiangolo/meinheld-gunicorn-flask:python3.7 as partypi
MAINTAINER Justin Shenk <shenkjustin@gmail.com>

RUN mkdir -p /app
WORKDIR /app

COPY ./app/requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

#Install partypi
#RUN pip install .

COPY ./gunicorn_conf.py /
ENV MODULE_NAME="partypi.main" 
#ENV GUNICORN_CMD_ARGS="chdir=/app/partypi/partypi"
# Expose the web port
EXPOSE 80
