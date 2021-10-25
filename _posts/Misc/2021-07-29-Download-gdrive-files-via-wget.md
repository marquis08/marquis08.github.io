---
date: 2021-07-29 00:41
title: "Download Gdrive file via wget or gdown"
categories: Gdrive Misc
tags: Gdrive Misc
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Wget
1. 링크 생성(링크가 있는 모든 사용자에게 공개)
2. wget 쿠키 설정으로 다운로드
    ```sh
    wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILEID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILEID}" -O {FILENAME} && rm -rf ~/cookies.txt
    ```
# Gdown
- CLI
    ```sh
    $ gdown --help
    usage: gdown [-h] [-V] [-O OUTPUT] [-q] [--fuzzy] [--id]  [--proxy PROXY]
                [--speed SPEED] [--no-cookies] [--no-check-certificate]
                [--continue] [--folder]
                url_or_id
    ```
- Python
    ```py
    import gdown

    url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2vNm9zMTJWOGxobkU'
    output = '20150428_collected_images.tgz'
    gdown.download(url, output, quiet=False)
    ```

# Reference
> <https://deeplify.dev/server/bash/download-google-drive-file-in-terminal>  
> <https://github.com/wkentaro/gdown>

