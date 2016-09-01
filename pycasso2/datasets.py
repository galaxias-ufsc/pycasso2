import urllib2
import time
from os import listdir
from termcolor import colored


def download_manga_cube(plate, plateifu, data_dir, wait = True, wait_time = 120
    , overwrite = False):

    """
    Downloads MaNGA datacube of given plate and plateifu to data_dir.
    """

    plate = str(plate)

    file_name = 'manga-' + plateifu + '-LINCUBE.fits.gz'

    if overwrite == False and file_name in listdir(data_dir):
        print colored('File already downloaded', 'red')

    else:

        url =  'https://data.sdss.org/sas/dr13/manga/spectro/redux/v1_5_4/' + plate
        url += '/stack/manga-' + plateifu + '-LINCUBE.fits.gz'

        u = urllib2.urlopen(url)

        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        print "Downloading: %s Bytes: %s" % (file_name, file_size)

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl
            , file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print status,

        f.close()

        if wait == True:
            #Wait some time:
            time.sleep(wait_time)