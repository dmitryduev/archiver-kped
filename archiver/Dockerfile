# To use it, run:
#   $ docker build -t archiver-kped -f archiver/Dockerfile .

FROM python:3.6

# Install vim, git, fftw3, sextractor
RUN apt-get update && apt-get -y install apt-file && apt-file update && apt-get -y install vim && \
    apt-get install -y git && apt-get install -y fftw3 && apt-get install -y libfftw3-dev && \
    apt-get install -y sextractor

# place to keep our app, raw data, and processed data:
RUN mkdir -p /app
RUN mkdir -p /data
RUN mkdir -p /archive

# install python libs
COPY code/requirements.txt /app/
RUN pip install Cython && pip install numpy
RUN pip install -r /app/requirements.txt

# copy over the secrets:
COPY secrets.json /app/

# copy over the master calibration data
RUN mkdir -p /app/calib
ADD calib/ /app/calib/

# get list of observed targets from Michael's github:
ADD https://raw.githubusercontent.com/mcoughlin/kp84/master/input/observed.dat /app/

# copy over the code
ADD code/ /app/

# change working directory to /app
WORKDIR /app

# run archiver
#CMD /usr/local/bin/supervisord -n -c supervisord.conf
#CMD /bin/bash
CMD python archiver.py config.json
