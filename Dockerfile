FROM python:3.6 as partypi
MAINTAINER Justin Shenk <shenkjustin@gmail.com>

RUN mkdir -p /partypi
WORKDIR /partypi

COPY requirements.txt /partypi
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /partypi

# Install partypi
RUN pip install .

# Expose the web port
EXPOSE 8000
