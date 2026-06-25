# Imports
import os
import sys
import getopt
from typing import List, Optional, Any, Callable, Dict, Tuple
from zipfile import ZipFile
from io import BytesIO
import requests
from tqdm import tqdm
from urllib.error import HTTPError
from multiprocessing import Pool

# Constants
OUTPUT_PATH = './env'
RAW_FOLDER = f'{OUTPUT_PATH}/raw'
DATA_FOLDER = f'{OUTPUT_PATH}/data'

def download_data(path: str, remote: str, name: Optional[str]=None,
        loc: Optional[str]=None, chunk_size: int=1024):
    """Downloads a file from a remote location to the specified path
    using a GET request. 
    
    Parameters
    ----------
    path : str
        The path on the machine where the file should be downloaded to.
    remote : str
        The remote location which can be queried with a GET request.
    name : str (default path)
        The name of the file that is being pulled from the remote.
    loc : str (default remote)
        The location where the remote file lives.
    chunk_size : int (default 1024)
        The number of bytes to read when streaming the content into a file.
    
    Raises
    ------
    HTTPError
        If the file could not be downloaded from the remote location.
    
    Source
    ------
    [qualtrics_data_compiler.py](https://github.com/ahaim5357/10.17605-osf.io-74bzs/blob/main/qualtrics_data_compiler.py#L109-L144) (MIT License)
    """
    if loc is None:
        loc = remote
    if name is None:
        name = path

    if not os.path.exists(path):
        print(f'Downloading {name} from \'{loc}\'.')

        # Download data
        response: requests.Response = requests.get(remote, stream=True)
        if not response.ok:
            raise HTTPError(f'Failed to download {name} from \'{loc}\'.')

        # Write data to location
        with open(path, 'wb') as data, tqdm(
                desc=f'Download {name}', unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            # Content iterator for progress information
            for bytes in response.iter_content(chunk_size=chunk_size):
                written: int = data.write(bytes)
                bar.update(written)

def zip_extractor(path: str, transformers: Dict[str, Callable[[ZipFile, str], Any]]):
    """Extracts and transforms the data in a zip file based upon their
    extensions.

    Parameters
    ----------
    path : str
        The path to the zip file.
    transformers : dict
        A dictionary of file extensions ('.' included) to runnables that
        take in the in the zip file operated on and the name of the file
        being extracted.
    """
    with ZipFile(path, 'r') as zip_ref:
        # Use tqdm as tracker for file extraction
        for file_name in tqdm(iterable=zip_ref.namelist(), desc=f'Extract {path}', unit='files', total=len(zip_ref.namelist())):
            transformers[os.path.splitext(file_name)[1]](zip_ref, file_name)

def csv_handler(zip_ref: ZipFile, file_name: str):
    """A transformer to extract CSV files to the data folder.

    Parameters
    ----------
    zip_ref : `zipfile.ZipFile`
        The zip file being operated on.
    file_name : str
        The name of the file being extracted.
    """
    zip_ref.extract(file_name, path=DATA_FOLDER)

def pdf_handler(zip_ref: ZipFile, file_name: str):
    """A transformer to extract PDF files to the root folder.

    PDFs with spaces in the name are considered to be nonessential and
    not extracted from the zip.

    Parameters
    ----------
    zip_ref : `zipfile.ZipFile`
        The zip file being operated on.
    file_name : str
        The name of the file being extracted.
    """
    if ' ' not in file_name:
        zip_ref.extract(file_name, path=OUTPUT_PATH)

def txt_handler(zip_ref: ZipFile, file_name: str, license_prefix: Optional[str] = None):
    """A transformer to extract TXT files to the root folder.

    Files named `license.txt` are assumed to be data licenses, and are renamed
    as such. Data licenses can be assigned a prefix.

    Parameters
    ----------
    zip_ref : `zipfile.ZipFile`
        The zip file being operated on.
    file_name : str
        The name of the file being extracted.
    license_prefix : str, optional (default None)
        A prefix to assign to the beginning of a license.txt file.
    """
    # Assume all license.txt files are data licenses (which they are in the two cases)
    if file_name == 'license.txt':
        license_name: str = f'{license_prefix}-DATA-LICENSE' if license_prefix is not None else 'DATA-LICENSE'
        with open(f'{OUTPUT_PATH}/{license_name}', mode='wb') as el:
            el.write(zip_ref.read(file_name))
    else:
        zip_ref.extract(file_name, path=OUTPUT_PATH)

def empty_handler(zip_ref: ZipFile, file_name: str, prefix: Optional[str] = None):
    """A transformer to extract files with no extensions to the root folder.

    Files with no extensions are typically licenses, so they will be treated
    similarly to those.

    Parameters
    ----------
    zip_ref : `zipfile.ZipFile`
        The zip file being operated on.
    file_name : str
        The name of the file being extracted.
    prefix : str, optional (default None)
        A prefix to assign to the beginning of the file.
    """
    name: str = f'{prefix}-{file_name}' if prefix is not None else file_name
    with open(name, mode='wb') as el:
        el.write(zip_ref.read(file_name))

def zip_handler(zip_ref: ZipFile, file_name: str):
    """A transformer to extract ZIP files to the root folder.

    This method handles two additional layers from the root ZIP and extracts
    them into directories based upon the name of the ZIP.

    Parameters
    ----------
    zip_ref : `zipfile.ZipFile`
        The zip file being operated on.
    file_name : str
        The name of the file being extracted.
    """
    # Read first zip file
    with ZipFile(BytesIO(zip_ref.read(file_name))) as exps_zip:
        for exp in tqdm(iterable=exps_zip.namelist(), desc=f'Extract {file_name}', unit='files', total=len(exps_zip.namelist())):
            name, ext = os.path.splitext(exp)
            dir: str = DATA_FOLDER
            # If the file in the zip is a zip file
            if ext == '.zip':
                # Use zip file name as directory name
                dir = f'{dir}/{name}'
                # Use makedirs for multiprocessing support
                os.makedirs(dir, exist_ok=True)
                with ZipFile(BytesIO(exps_zip.read(exp))) as exp_zip:
                    exp_zip.extractall(dir)
            # Just extract as normal not a zip file
            else:
                exps_zip.extract(exp, dir)

def none_handler(zip_ref: ZipFile, file_name: str):
    """A transformer which does nothing.

    Parameters
    ----------
    zip_ref : `zipfile.ZipFile`
        The zip file being operated on.
    file_name : str
        The name of the file being extracted.
    """
    pass

def setup_dataset_packed(packed: Tuple[str, str, str, bool]):
    """Downloads and extracts a dataset from an OSF project into the
    approriate environment.

    Parameters
    ----------
    packed : tuple
        A tuple containing the variables necessary to set up the environment.

    See
    ---
    setup_dataset(project, name, dir, docs)
    """
    setup_dataset(packed[0], packed[1], packed[2], packed[3])

def setup_dataset(project: str, name: str, dir: str='./', docs: bool=True):
    """Downloads and extracts a dataset from an OSF project into the
    approriate environment.

    Parameters
    ----------
    project : str
        The five character alphanumeric code uniquely representing an OSF project.
    name : str
        The name of the zip file to extract to, in snake case.
    dir : str (default `./`)
        The directory to extract the OSF ZIP file to.
    docs : bool (default `True`)
        Whether the associated documentation and licenses should be provided.
    """
    path: str = f'{dir}/{name}.zip'
    # Download data from the OSF project
    download_data(path, f'https://files.osf.io/v1/resources/{project}/providers/osfstorage/?zip=', f'{name.capitalize()} Dataset', f'https://osf.io/{project}/')
    
    # Setup transformers
    transformers: Dict[str, Callable[[ZipFile, str], Any]] = {
        '.csv': csv_handler,
        '.zip': zip_handler,
        '.pdf': none_handler,
        '.txt': none_handler,
        '': none_handler
    }
    if docs:
        transformers['.pdf'] = pdf_handler
        transformers['.txt'] = lambda zip_ref, file_name: txt_handler(zip_ref, file_name, license_prefix=name.upper())
        transformers[''] = lambda zip_ref, file_name: empty_handler(zip_ref, file_name, prefix=name.upper())

    # Extract zip to appropriate locations
    zip_extractor(path, transformers)

def env_var_check(name: str, default: bool=False):
    """Checks whether an environment variable is present and is set to
    some representation of `True`.

    The values of `True` accepted in any case are `true`, `t`, `yes`,
    `y`, and `1`.

    Parameters
    ----------
    name : str
        The name of the environment variable.
    default : bool (default False)
        The value to default if the environment variable is not present.

    See
    ---
    [qualtrics_data_compiler.py](https://github.com/ahaim5357/10.17605-osf.io-74bzs/blob/main/qualtrics_data_compiler.py#L151) (MIT License)
    """
    return default if os.getenv(name) is None else os.getenv(name).lower() in ['true', 't', 'yes', 'y', '1']

# Main Code

def main(args: List[str]) -> int:
    """Executes the environment setup script for the current environment.

    Parameters
    ----------
    args : str, list-like
        The list of arguments to supply to the script.

    Returns
    -------
    int
        The exit status of the function, where zero is a "successful termination" and any nonzero value is an "abnormal termination".
    
    See
    ---
    [sys.exit](https://docs.python.org/3.8/library/sys.html#sys.exit)
    """

    # Get environment variables and command line args
    raw_zip: bool = env_var_check('M2JQE_RAW_ZIP')
    docs: bool = env_var_check('M2JQE_DOCS', True)
    multiprocess: bool = env_var_check('M2JQE_MULTIPROCESS')
    try:
        opts, _ = getopt.getopt(args, 'hznm', ['help', 'raw-zip', 'no-docs', 'multiprocess'])
    except getopt.GetoptError:
        print('env_setup.py [-h | --help] [-z | --raw-zip] [-n | --no-docs] [-m | --multiprocess]')
        return 2
    for opt, _ in opts:
        if opt in ('-h', '--help'):
            print('env_setup.py [-h | --help] [-z | --raw-zip] [-n | --no-docs] [-m | --multiprocess]')
            return 0
        elif opt in ('-z', '--raw-zip'):
            raw_zip = True
        elif opt in ('-n', '--no-docs'):
            docs = False
        elif opt in ('-m', '--multiprocess'):
            multiprocess = True
    
    # Set up relative directories
    os.makedirs(DATA_FOLDER, exist_ok=True)

    zip_dir: str = './'
    if raw_zip:
        zip_dir = RAW_FOLDER
        os.makedirs(zip_dir, exist_ok=True)

    # Setup project
    projects: List[Tuple[str, str, str, bool]] = [
        ('59shv', 'original', zip_dir, docs),
        ('m2jqe', 'expansion', zip_dir, docs)
    ]

    if multiprocess:
        with Pool(processes=2) as pool:
            pool.map(setup_dataset_packed, projects)
    else:
        for project in projects:
            setup_dataset_packed(project)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
