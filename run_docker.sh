# This runs a jupyter in docker container with project folder mounted.
# This container has all the necessary software set up.
sudo docker run --runtime=nvidia -it --net=host -v $PWD:/user_data iaroslavai/scuda bash
