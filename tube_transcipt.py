import requests
from bs4 import BeautifulSoup
import re
from pytube import YouTube
import ssl

# install
#  pip install git+https://github.com/nficano/pytube.git
#  pip install beautifulsoup4

requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

urlstr = "https://www.youtube.com/c/yanfookchurch/videos"
r = requests.get(urlstr)
html_doc = r.text
soup = BeautifulSoup(html_doc, 'html.parser')
count = 0
for line in soup.prettify().splitlines():
    if re.match(r".*2020-08-09 恩福中心主早堂崇拜.*", line):
        str = line[29:]
        tokens = str.split(':')
        for i,tok in enumerate(tokens):
            if re.match(r".*2020-08-09 恩福中心主早堂崇拜 .*", tok):
                print(tok)
                for j in range(i,i+50):
                    if re.match(r".*url.*", tokens[j]):
                        if re.match(r".*watch.*", tokens[j+1]):
                            targets = tokens[j+1].split(',')
                            videourl = "https://youtube.com"+targets[0].replace('\"','')
                            print(videourl)
                            YouTube(videourl).streams.first().download('sermon.mp4')

print(count)