FROM python:3.8-slim
LABEL maintainer="dmitriyburenok@gmail.com"
ENV PYTHONUNBUFFERED=1
RUN apt update -y
RUN apt install -y libgl1-mesa-glx gcc curl libglib2.0-0
COPY . /yolov7-api
WORKDIR /yolov7-api
RUN pip install -r requirements.txt
RUN curl -LO https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
CMD gunicorn --bind 0.0.0.0:14641 api:app -w 1 --threads 4
