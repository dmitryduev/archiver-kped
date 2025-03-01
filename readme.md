### KPED Archiver

_Dockerized Archiver for Kitt Peak EMCCD Demonstrator project_

Clone the repo and cd to the directory:
```bash
git clone https://github.com/dmitryduev/archiver-kped.git
cd archiver-kped
```

Create a persistent Docker volume for MongoDB:
```bash
docker volume create archiver-kped-mongo-volume
```

Launch the MongoDB container. Feel free to change u/p for the admin
```bash
# docker build -t archiver-kped-mongo -f database/Dockerfile .
docker run -d --restart always --name archiver-kped-mongo -p 27018:27017 -v archiver-kped-mongo-volume:/data/db \
       -e MONGO_INITDB_ROOT_USERNAME=mongoadmin -e MONGO_INITDB_ROOT_PASSWORD=mongoadminsecret \
       mongo:latest
```

Feel free to change u/p for the mongo admin. If you choose to do so, change the corresponding fields in the archiver config file `archiver/code/config.json`: `"database" -> "admin"` and `"admin_pwd"`. Also feel free to change the the default mongo u/p specified in the archiver config file.

Create file archiver/secrets.json with the Kowalski login credentials:
```json
{
  "kowalski": {
    "user": "USER",
    "password": "PASSWORD"
  }
}
```

Bulid and launch the archiver container. Bind-mount the raw/processed data directories on the host machine:
```bash
cd archiver
docker build -t archiver-kped -f Dockerfile .
docker run -v /path/to/raw/data:/data \
           -v /path/to/archive:/archive \
           --name archiver-kped -d --link archiver-kped-mongo:mongo --restart always archiver-kped
```
