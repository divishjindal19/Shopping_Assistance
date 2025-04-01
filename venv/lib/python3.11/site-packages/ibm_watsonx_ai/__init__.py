#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

pkg_name = "ibm-watsonx-ai"

try:
    from importlib.metadata import version

    ver = version(pkg_name)

except (ModuleNotFoundError, AttributeError):
    from importlib_metadata import version as imp_lib_ver

    ver = imp_lib_ver(pkg_name)

from ibm_watsonx_ai.credentials import Credentials
from ibm_watsonx_ai.client import APIClient

APIClient.version = ver
__version__ = ver
