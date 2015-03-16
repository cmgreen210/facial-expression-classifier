from distutils.core import setup


def _get_data_files(key):
    files = []
    if key == 'haar':
        files.append('haarcascade_frontalface_alt.xml')
        files.append('haarcascade_frontalface_alt2.xml')
        files.append('haarcascade_frontalface_alt_tree.xml')
        files.append('haarcascade_frontalface_default.xml')
    return files

setup(
    # Application name:
    name="Face Emotion Classifier",

    # Version number (initial):
    version="0.1.0",

    # Application author details:
    author="Chris Green",
    author_email="cmgreen210@gmail.com",

    # Packages
    packages=["fec", "fec.classifier", "fec.media", "fec.test"],

    # Include additional files into the package
    package_data={
        'fec.classifier': ['data/*.xml'],
        'fec.test': ['data/*.txt', 'data/*.jpg', 'data/*.npx',
                     'data/gl_mdl/dir_archive.ini',
                     'data/gl_mdl/objects.bin', 'data/img_10/*']
    },
    # Details
    url="https://github.com/cmgreen210/fec",

    #
    # license="LICENSE.txt",
    description="Facial django_expression classification from images and video",

    long_description=open("README.md").read(),
)
