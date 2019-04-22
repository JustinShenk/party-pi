FROM alpine:edge as openssl
RUN apk upgrade --update-cache --available && \
    apk add openssl && \
    rm -rf /var/cache/apk/*

# Generate self-signed certificate
RUN openssl req \
    -new \
    -newkey rsa:4096 \
    -days 365 \
    -nodes \
    -x509 \
    -subj "/C=US/ST=Denial/L=Springfield/O=Dis/CN=www.partypi.net" \
    -keyout /www.partypi.net.key \
    -out /www.partypi.net.cert

FROM python:3.6
MAINTAINER Justin Shenk <shenkjustin@gmail.com>

RUN apt-get update
RUN rm -rf /var/lib/apt/lists*

RUN mkdir -p /partypi

WORKDIR /partypi

COPY requirements.txt /partypi
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /partypi

# Install partypi
RUN pip install .

# Expose the web port
EXPOSE 5000

COPY --from=openssl /www.partypi.* /partypi/partypi/
CMD ["gunicorn",  "-c", "gunicorn.conf", "main:app"]
