import requests
import os.path


def download_file(filename, url):
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)


def download_if_not_exists(filename, url):
    if not os.path.exists(filename):
        download_file(filename, url)

download_if_not_exists('cfg/yolov3.weights', 'https://pjreddie.com/media/files/yolov3.weights')
