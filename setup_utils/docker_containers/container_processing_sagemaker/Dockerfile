FROM centos:7

ARG username_proxy
ARG password_proxy
ARG git_token

MAINTAINER Amazon AI <sage-learner@amazon.com>


RUN yum -y install \
         wget \
         gcc-c++ libgcc libstdc++ \
         python3 python3-devel \
         build-essential \
         nginx \
         git \
         ca-certificates \
         libgomp \
    && rm -rf /var/lib/apt/lists/*

RUN yum -y groupinstall 'Development Tools'

RUN yum -y install java-1.8.0-openjdk

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
COPY requirements.txt /
COPY setup.py /
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py \
#RUN wget https://bootstrap.pypa.io/get-pip.py \
&& python3 get-pip.py && \
python3 -m pip install ipykernel setuptools wheel && \
python3 -m pip install -r requirements.txt && \
    rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY container /opt/program

WORKDIR /opt/program
RUN ls


ENTRYPOINT ["python3"]
