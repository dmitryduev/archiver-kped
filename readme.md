### KPED Archiver

_Dockerized Archiver for Kitt Peak EMCCD Demonstrator project_

Clone the repo and cd to the directory:
```bash
git clone https://github.com/mansikasliwal/archiver-kped.git
cd archiver-kped
```

Create a persistent Docker volume for MongoDB:
```bash
docker volume create archiver-kped-mongo-volume
```

Launch the MongoDB container. Feel free to change u/p for the admin
```bash
# docker build -t archiver-kped-mongo -f database/Dockerfile .
docker run -d --restart always --name archiver-kped-mongo -p 27018:2017 -v archiver-kped-mongo-volume:/db \
       -e MONGO_INITDB_ROOT_USERNAME=mongoadmin -e MONGO_INITDB_ROOT_PASSWORD=mongoadminsecret \
       mongo:latest
```

