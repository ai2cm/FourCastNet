FROM nvcr.io/nvidia/pytorch:21.11-py3

# update repo info
RUN apt update -y

# install mpi4py
RUN pip install mpi4py

# h5py
RUN pip install h5py

# other python stuff
RUN pip install wandb && \
    pip install ruamel.yaml && \
    pip install --upgrade tqdm && \
    pip install timm && \
    pip install einops && \
    pip install moviepy imageio && \
    pip install netCDF4

# benchy
RUN pip install git+https://github.com/romerojosh/benchy.git

# set wandb to offline
#ENV WANDB_MODE offline

# copy source code
RUN mkdir -p /opt/ERA5_wind
COPY config /opt/ERA5_wind/config
COPY copernicus /opt/ERA5_wind/copernicus
COPY docker /opt/ERA5_wind/docker
COPY networks /opt/ERA5_wind/networks
COPY utils /opt/ERA5_wind/utils
COPY inference /opt/ERA5_wind/inference
COPY *.py /opt/ERA5_wind/
COPY *.sh /opt/ERA5_wind/

# create dummy git image
RUN cd /opt/ERA5_wind && git init
