FROM python:3.6
MAINTAINER Justin Shenk "shenk.justin@gmail.com"

RUN apt-get update -y && apt-get install -y \
        build-essential \
        openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /partypi
RUN pip3 install --no-cache-dir pipenv

ADD Pipfile /partypi/Pipfile
ADD Pipfile.lock /partypi/Pipfile.lock
RUN pipenv install --system

ADD . /partypi

RUN openssl req \
    -new \
    -newkey rsa:4096 \
    -days 365 \
    -nodes \
    -x509 \
    -subj "/C=US/ST=Denial/L=Springfield/O=Dis/CN=www.example.com" \
    -keyout www.example.com.key \
    -out www.example.com.cert

# Expose the web port
EXPOSE 5000

CMD ["gunicorn",  "-c", "gunicorn.conf", "main:app"]
