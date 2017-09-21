import time
from os import listdir



def download_manga_cube(plate, plateifu, data_dir, wait=True, wait_time=120, overwrite=False):
    """
    Downloads MaNGA datacube of given plate and plateifu to data_dir.
    """
    from six.moves import urllib

    plate = str(plate)

    file_name = 'manga-' + plateifu + '-LINCUBE.fits.gz'

    if overwrite == False and file_name in listdir(data_dir):
        print('File already downloaded')

    else:

        url = 'https://data.sdss.org/sas/dr14/manga/spectro/redux/v2_1_2/' + \
            plate
        url += '/stack/manga-' + plateifu + '-LINCUBE.fits.gz'

        u = urllib.request.urlopen(url)

        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        print("Downloading: %s Bytes: %s" % (file_name, file_size))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (
                file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8) * (len(status) + 1)
            print(status, end=' ')

        f.close()

        if wait == True:
            # Wait some time:
            time.sleep(wait_time)


def download_manga_image(plate, ifudsgn, objid, imdir, thumb=False):
    from six.moves import urllib
   
    if str(objid) + '.png' in listdir(imdir):

        print('Already downloaded')

    else:

        url = 'https://data.sdss.org/sas/dr14/manga/spectro/redux/v2_1_2/'
        if thumb:
            image_name = str(ifudsgn).strip() + '_thumb.png'
        else:
            image_name = str(ifudsgn).strip() + '.png'
        url += str(plate).strip() + '/stack/images/' + image_name

        print(url)
        file_name = imdir + str(objid) + '.png'
        u = urllib.request.urlopen(url)
        file_size = int(u.getheader('Content-Length'))
        print('Downloading: %s Bytes: %s' % (file_name, file_size))

        file_size_dl = 0
        block_sz = 8192
        with open(file_name, 'wb') as f:
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break
    
                file_size_dl += len(buffer)
                f.write(buffer)
                status = r'%10d  [%3.2f%%]' % (
                    file_size_dl, file_size_dl * 100. / file_size)
                status = status + chr(8) * (len(status) + 1)
                print(status)
