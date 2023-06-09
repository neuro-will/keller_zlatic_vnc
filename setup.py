from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='keller_zlatic_vnc',
    version='0.1.0',
    author='William Bishop',
    author_email='bishopw@hhmi.org',
    packages=['keller_zlatic_vnc'],
    python_requires='>=3.7.0',
    description='Core code which supports William Bishops work with the Keller and Zlatic lab VNC data.',
    long_description = long_description,
    install_requires=[
	"findspark",
	"fpdf",
	"h5py",
    "imageio",
	"matplotlib",
    "numpy",
    "opencv_python",
    "openpyxl",
	"pandas",
	"pyqtgraph",
	"pyspark",
    "ruamel.yaml",
	"scikit-image",
    "sphinx",
    "tifffile",
    "xlrd",
    ],
)
