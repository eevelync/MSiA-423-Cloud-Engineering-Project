# Cloud Engineering Project
- Lambda Functions/: Contains the lambda functions used for this project using a docker image
- models/: Contains the base and final models python scripts and the model objects uploaded and retrieved from s3, then running the model training locally. However it isn't necessary since we also used lambda function to perform the model operations
- notebooks/: Contains notebooks that were used for the general outline of our project. More importantly the script that cleaned and uploaded the original data from kaggle to an s3 bucket
- DockerfileApp: Used to build app docker image
- DockerfilePres.txt: Used to build presentation image
- presentation_app.py: This is our first streamlit app, we are using it in replace of a slide deck, since we believe that it will be easier to showcase our points and make it interactive
- app.py: This is our final app our end user will interact with, since we assume they wouldn't be interested in all the background information, and would only want the fair prce predictions

# Docker Usage (Works for team members with AWS permissions)
### Run the following commmands in terminal first to store aws keys locally. Keys will be stored in ~/.aws/credentials locally
- aws configure
- AWS ACCESS KEY ID = [input access key]
- AWS SECREY ACCESS KEY ID = [input secret access key]
- Default region name [us-east-2]: [click enter]
- Default output format [None]: [click enter]

- export AWS_ACCESS_KEY_ID
- export AWS_SECRET_ACCESS_KEY_ID

### Once Aws configuration is set run the following

- docker build -t demo-app -f DockerfileApp .
- docker run demo-app

- docker build -t demo-app -f DockerfilePres.txt .
- docker run demo-app


# Presentation Link
http://presentation4-267781137.us-east-2.elb.amazonaws.com/

# Demo Link
http://demo-app-1129752538.us-east-2.elb.amazonaws.com/
