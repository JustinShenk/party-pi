FROM python:3.6
MAINTAINER Justin Shenk <shenkjustin@gmail.com>

RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-pip
#RUN apt-get update -y && apt-get install -y \
        #build-essential \
        #openssl \
    #&& rm -rf /var/lib/apt/lists/*

RUN mkdir -p /partypi

WORKDIR /partypi
COPY requirements.txt /partypi
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /partypi

# Install partypi
RUN pip install .

#RUN openssl req \
    #-new \
    #-newkey rsa:4096 \
    #-days 365 \
    #-nodes \
    #-x509 \
    #-subj "/C=US/ST=Denial/L=Springfield/O=Dis/CN=www.partypi.net" \
    #-keyout www.partypi.net.key \
    #-out www.partypi.net.cert

# Expose the web port
EXPOSE 5000

CMD ["gunicorn",  "-c", "gunicorn.conf", "main:app"]
