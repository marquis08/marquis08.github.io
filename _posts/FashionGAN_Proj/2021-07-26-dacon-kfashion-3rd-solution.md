---
date: 2021-07-26 00:31
title: "Examine Dacon K-fashion 3rd solution with Docker"
categories: Docker FashionGAN_Proj
tags: Docker FashionGAN_Proj
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# git repo
<https://github.com/dacon-ai/K-Fashion-3rd>  

- `./docker/build.sh` 
    ```sh
    #!/usr/bin/env bash
    set -ex

    CUDA_VERSION_MAJOR_MINOR="10.2"

    docker build \
        --build-arg CUDA_VERSION_MAJOR_MINOR=${CUDA_VERSION_MAJOR_MINOR} \
        -t "${D2HUB_IMAGE}" . -f docker/Dockerfile
    ```
    - `set -e` : 오류가 발생하면 중단
    - `set -x` : 디버그 용도로 자주 사용하는 옵션. 실행되는 명령어와 인수들을 출력



# Dockerfile reference
<https://docs.docker.com/engine/reference/builder/>

Docker can build images automatically by reading the instructions from a `Dockerfile`. A `Dockerfile` is a text document that contains all the commands a user could call on the command line to assemble an image. Using `docker build` users can create an automated build that executes several command-line instructions in succession.  

## Usage
The docker build command builds an image from a `Dockerfile` and a context. The build’s context is the set of files at a specified location `PATH` or `URL`. The `PATH` is a directory on your local filesystem. The `URL` is a Git repository location.  

The build context is processed recursively. So, a PATH includes any subdirectories and the URL includes the repository and its submodules. This example shows a build command that uses the current directory (.) as build context:  

The build is run by the **Docker daemon**, **not by the CLI**. The first thing a build process does is send the entire context (recursively) to the daemon. In most cases, it’s best to start with an empty directory as context and keep your Dockerfile in that directory. Add only the files needed for building the Dockerfile.

> ⛔ Warning  
>  
> Do not use your root directory, /, as the PATH for your build context, as it causes the build to transfer the entire contents of your hard drive to the Docker daemon.  

To use a file in the build context, the Dockerfile refers to the file specified in an instruction, for example, a COPY instruction. To increase the build’s performance, exclude files and directories by adding a .dockerignore file to the context directory. For information about how to create a .dockerignore file see the documentation on this page.

Traditionally, the Dockerfile is called Dockerfile and located in the root of the context. You use the -f flag with docker build to point to a Dockerfile anywhere in your file system.
```sh
$ docker build -f /path/to/a/Dockerfile .
```

You can specify a repository and tag at which to **save the new image** if the build succeeds:
```sh
$ docker build -t shykes/myapp .
```  
To tag the image **into multiple repositories after the build**, add multiple -t parameters when you run the build command:
```sh
$ docker build -t shykes/myapp:1.0.2 -t shykes/myapp:latest .
```

## Format
Here is the format of the Dockerfile:  

```dockerfile
# Comment
INSTRUCTION arguments
```
The instruction is not case-sensitive. However, convention is for them to be UPPERCASE to distinguish them from arguments more easily.  

Docker runs instructions in a `Dockerfile` **in order**. A `Dockerfile` **must begin with a `FROM` instruction**. This may be after parser directives, comments, and globally scoped ARGs. The `FROM` instruction specifies the **Parent Image** from which you are building. `FROM` may only be preceded by one or more `ARG` instructions, which declare arguments that are used in `FROM` lines in the Dockerfile.

Docker treats lines that begin with # as a comment, unless the line is a valid parser directive. A # marker anywhere else in a line is treated as an argument. This allows statements like:  
```dockerfile
# Comment
RUN echo 'we are running some # of cool things'
```  
Comment lines are removed before the Dockerfile instructions are executed, which means that the comment in the following example is not handled by the shell executing the echo command, and both examples below are equivalent:  
```dockerfile
RUN echo hello \
# comment
world
```
```dockerfile
RUN echo hello \
world
```  

## Environment replacement
Environment variables (declared with [the `ENV` statement](https://docs.docker.com/engine/reference/builder/#env)) can also be used in certain instructions as variables to be interpreted by the `Dockerfile`. Escapes are also handled for including variable-like syntax into a statement literally.

Environment variables are supported by the following list of instructions in the Dockerfile:
- ADD
- COPY
- ENV
- EXPOSE
- FROM
- LABEL
- STOPSIGNAL
- USER
- VOLUME
- WORKDIR
- ONBUILD (when combined with one of the supported instructions above)

## .dockerignore file
Before the docker CLI sends the context to the docker daemon, it looks for a file named `.dockerignore` in the root directory of the context. If this file exists, the CLI modifies the context to exclude files and directories that match patterns in it. This helps to avoid unnecessarily sending large or sensitive files and directories to the daemon and potentially adding them to images using `ADD` or `COPY`.

The CLI interprets the `.dockerignore` file as a newline-separated list of patterns similar to the file globs of Unix shells. For the purposes of matching, the root of the context is considered to be both the working and the root directory. For example, the patterns `/foo/bar` and `foo/bar` both exclude a file or directory named `bar` in the `foo` subdirectory of PATH or in the root of the git repository located at `URL`. Neither excludes anything else.

If a line in `.dockerignore` file starts with `#` in column 1, then this line is considered as a comment and is ignored before interpreted by the CLI.

Here is an example `.dockerignore` file:
```dockerfile
# comment
*/temp*
*/*/temp*
temp?
```  

This file causes the following build behavior:

|Rule	|Behavior|
|---|---|
|`# comment`	|Ignored.|
|`*/temp*`	|Exclude files and directories whose names start with `temp` in any immediate subdirectory of the root. For example, the plain file `/somedir/temporary.txt` is excluded, as is the directory `/somedir/temp.`|
|`*/*/temp*`|	Exclude files and directories starting with `temp` from any subdirectory that is two levels below the root. For example, `/somedir/subdir/temporary.txt` is excluded.|
|`temp?`	|Exclude files and directories in the root directory whose names are a one-character extension of `temp`. For example, `/tempa` and `/tempb` are excluded.|

## FROM
```dockerfile
FROM [--platform=<platform>] <image> [AS <name>]
```  
Or  
```dockerfile
FROM [--platform=<platform>] <image>[:<tag>] [AS <name>]
```  
Or  
```dockerfile
FROM [--platform=<platform>] <image>[@<digest>] [AS <name>]
```  



# Take a look Dockerfile
- `./docker/Dockerfile`

```dockerfile
ARG CUDA_VERSION_MAJOR_MINOR

FROM nvidia/cuda:${CUDA_VERSION_MAJOR_MINOR}-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN \
    apt-get update -qq && apt-get install -yq --no-install-recommends \
        build-essential git curl wget cmake vim ssh bzip2 ca-certificates \
        ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# https://hub.docker.com/r/continuumio/miniconda3/dockerfile
RUN \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

# https://plotly.com/python/getting-started/
#RUN \
#    conda install -c plotly plotly=4.8.2 && \
#    conda install nodejs jupyterlab "ipywidgets=7.5" && \
#    jupyter labextension install jupyterlab-plotly@4.8.2 && \
#    jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.8.2

RUN conda install pytorch==1.6.0 torchvision cudatoolkit=10.2 -c pytorch -y
RUN conda install pandas

RUN pip install mmcv-full==latest+torch1.6.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

#RUN conda clean --all
#RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
#WORKDIR /mmdetection
#RUN pip install -r requirements/build.txt
#RUN pip install --no-cache-dir -e .

# environment variables
ENV LC_ALL=C.UTF-8
ENV TZ="Asia/Seoul"
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/extras/CUPTI/lib64/
```
