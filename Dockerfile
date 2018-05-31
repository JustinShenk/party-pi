FROM python:3
MAINTAINER Justin Shenk "shenk.justin@gmail.com"

RUN apt-get update -y
RUN apt-get install -y build-essential openssl
RUN mkdir -p /partypi
WORKDIR /partypi
COPY requirements.txt /partypi
RUN pip install --no-cache-dir -r requirements.txt

COPY . /partypi

RUN ./docker-ssl-cert-generate

# Expose the web port
EXPOSE 5000

CMD ["gunicorn",  "-c", "gunicorn.conf", "main:app"]
