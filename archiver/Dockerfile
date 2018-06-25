# To use it, run:
#   $ docker build -t growth-ingest -f Dockerfile.ingest .
#   $ docker run -d --restart always --name growth-ingest --link growth-db:growth-db -v growth-data:/growth-data -v alerts:/alerts -it growth-ingest

FROM python:3.6

# Install vim and git
RUN apt-get update && apt-get -y install apt-file && apt-file update && apt-get -y install vim && \
    apt-get install -y git

# place to keep our app:
RUN mkdir -p /app

# install python libs

# penquins to talk to Kowalski:
RUN pip install pymongo>=3.6.1 socketIO_client>=0.7.2 \
    && pip install git+https://github.com/dmitryduev/broker.git

# install supervisord
RUN pip install git+https://github.com/Supervisor/supervisor.git


ADD archiver/ /app
ADD Marshal/ingest/ /ingest

# set up secrets
# make sure your secret file in the root:
COPY secret.json /ingest
COPY Marshal/secret.py /ingest
RUN python /ingest/secret.py

# change working directory to /ingest
WORKDIR /ingest
RUN git clone https://github.com/ZwickyTransientFacility/ztf-avro-alert.git

# init filter interpreter and run alert watcher
#CMD /bin/bash
CMD /usr/local/bin/supervisord -n -c supervisord.conf
#CMD python init_filter_interp.py && python alert_watcher_kafka.py config.json > alert_watcher_kafka.log
