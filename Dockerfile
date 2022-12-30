FROM continuumio/anaconda3

WORKDIR /home/jacopo/Documents/internship/content/projects/thesis-mlops-platform-poc/

COPY . .

RUN apt-get update
RUN apt-get install make

RUN pip install -r requirements.txt

CMD make 

