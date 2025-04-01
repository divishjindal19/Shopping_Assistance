#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

"""
.. module:: APIClient
   :platform: Unix, Windows
   :synopsis: IBM watsonx.ai API Client.

.. moduleauthor:: IBM
"""

import copy
import logging
import os
from warnings import warn
from typing import Any, cast, TypeAlias
import json
import base64

import ibm_watsonx_ai.utils
from ibm_watsonx_ai.utils import get_user_agent_header
from ibm_watsonx_ai.utils.utils import _APIClientSession
from ibm_watsonx_ai.Set import Set
from ibm_watsonx_ai.assets import Assets
from ibm_watsonx_ai.connections import Connections
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.deployments import Deployments
from ibm_watsonx_ai.experiments import Experiments
from ibm_watsonx_ai.export_assets import Export
from ibm_watsonx_ai.factsheets import Factsheets
from ibm_watsonx_ai.foundation_models_manager import FoundationModelsManager
from ibm_watsonx_ai.functions import Functions
from ibm_watsonx_ai.ai_services import AIServices
from ibm_watsonx_ai.hw_spec import HwSpec
from ibm_watsonx_ai.import_assets import Import
from ibm_watsonx_ai.service_instance import ServiceInstance
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.model_definition import ModelDefinition
from ibm_watsonx_ai.models import Models
from ibm_watsonx_ai.parameter_sets import ParameterSets
from ibm_watsonx_ai.pipelines import Pipelines
from ibm_watsonx_ai.pkg_extn import PkgExtn
from ibm_watsonx_ai.spaces import Spaces
from ibm_watsonx_ai.remote_training_system import RemoteTrainingSystem
from ibm_watsonx_ai.repository import Repository
from ibm_watsonx_ai.script import Script
from ibm_watsonx_ai.shiny import Shiny
from ibm_watsonx_ai.sw_spec import SwSpec
from ibm_watsonx_ai.task_credentials import TaskCredentials
from ibm_watsonx_ai.training import Training
from ibm_watsonx_ai.utils import CPDVersion
from ibm_watsonx_ai.volumes import Volume
from ibm_watsonx_ai.wml_client_error import NoWMLCredentialsProvided
from ibm_watsonx_ai.wml_client_error import WMLClientError


# requests module or requests.Session
RequestsLikeType: TypeAlias = Any


class APIClient:
    """The main class of ibm_watsonx_ai. The very heart of the module. APIClient contains objects that manage the service reasources.

    To explore how to use APIClient, refer to:
     - :ref:`Setup<setup>` - to check correct initialization of APIClient for a specific environment.
     - :ref:`Core<core>` - to explore core properties of an APIClient object.

    :param url: URL of the service
    :type url: str

    :param credentials: credentials used to connect with the service
    :type credentials: Credentials

    :param project_id: ID of the project that is used
    :type project_id: str, optional

    :param space_id: ID of deployment space that is used
    :type space_id: str, optional

    :param verify: certificate verification flag, deprecated, use Credentials(verify=...) to set `verify`
    :type verify: bool, optional

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials

        credentials = Credentials(
            url = "<url>",
            api_key = "<api_key>"
        )

        client = APIClient(credentials, space_id="<space_id>")

        client.models.list()
        client.deployments.get_details()

        client.set.default_project("<project_id>")

        ...

    """

    version: str | None = None
    _internal: bool = False

    def __init__(
        self,
        credentials: Credentials | dict[str, str] | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        verify: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        if (wml_credentials := kwargs.get("wml_credentials")) is not None:
            warn("`wml_credentials` parameter is deprecated, please use `credentials`")
            if not credentials:
                credentials = wml_credentials
        if wml_credentials is None and credentials is None:
            raise TypeError("APIClient() missing 1 required argument: 'credentials'")

        self._logger = logging.getLogger(__name__)

        if verify is not None:
            warn(
                "`verify` parameter is deprecated. Use `ibm_watsonx_ai.Credentials` for passing `verify` parameter.",
                category=DeprecationWarning,
            )

        if isinstance(credentials, dict):
            warn(
                "`credentials` parameter as dict is deprecated. Use `ibm_watsonx_ai.Credentials` for passing parameters.",
                category=DeprecationWarning,
            )
            credentials = Credentials.from_dict(credentials, _verify=verify)

        if project_id is not None and space_id is not None:
            raise WMLClientError(
                "`project_id` parameter and `space_id` parameter cannot be set at the same time."
            )

        credentials._set_env_vars_from_credentials()

        if isinstance(credentials.verify, str):
            credentials.verify = True

        from ibm_watsonx_ai._wrappers import requests

        if credentials.proxies is not None:
            requests.additional_settings["proxies"] = credentials.proxies
        elif requests.additional_settings.get("proxies") is not None:
            del requests.additional_settings["proxies"]

        # At this stage `credentials` has type Dict[str, str]
        credentials = cast(Credentials, credentials)
        self.credentials = copy.deepcopy(credentials)
        self.default_space_id = None
        self.default_project_id = None
        self.project_type = None
        self.CLOUD_PLATFORM_SPACES = False
        self.PLATFORM_URL = None
        self.version_param = self._get_api_version_param()
        self.ICP_PLATFORM_SPACES = False  # This will be applicable for 3.5 and later and specific to convergence functionalities
        self.CPD_version = CPDVersion()
        self._iam_id = None
        self._spec_ids_per_state: dict = {}
        self.generate_ux_tag = True
        self.WCA: bool = False
        self._user_headers: dict | None = None  # Used in set_headers() method
        self.__session = None

        self.PLATFORM_URLS_MAP = {
            "https://wxai-qa.ml.cloud.ibm.com": "https://api.dai.test.cloud.ibm.com",
            "https://private.wxai-qa.ml.cloud.ibm.com": "https://private.api.dai.test.cloud.ibm.com",
            "https://wml-mcsp-dev.ml.test.cloud.ibm.com": "https://api.dai.dev.cloud.ibm.com",
            "https://wml-dev.ml.test.cloud.ibm.com": "https://api.dataplatform.dev.cloud.ibm.com",
            "https://wml-fvt.ml.test.cloud.ibm.com": "https://api.dataplatform.dev.cloud.ibm.com",
            "https://private.wml-fvt.ml.test.cloud.ibm.com": "https://api.dataplatform.dev.cloud.ibm.com",
            "https://us-south.ml.test.cloud.ibm.com": "https://api.dataplatform.dev.cloud.ibm.com",
            "https://yp-qa.ml.cloud.ibm.com": "https://api.dataplatform.test.cloud.ibm.com",
            "https://private.yp-qa.ml.cloud.ibm.com": "https://api.dataplatform.test.cloud.ibm.com",
            "https://yp-cr.ml.cloud.ibm.com": "https://api.dataplatform.test.cloud.ibm.com",
            "https://private.yp-cr.ml.cloud.ibm.com": "https://api.dataplatform.test.cloud.ibm.com",
            "https://jp-tok.ml.cloud.ibm.com": "https://api.jp-tok.dataplatform.cloud.ibm.com",
            "https://eu-gb.ml.cloud.ibm.com": "https://api.eu-gb.dataplatform.cloud.ibm.com",
            "https://eu-de.ml.cloud.ibm.com": "https://api.eu-de.dataplatform.cloud.ibm.com",
            "https://us-south.ml.cloud.ibm.com": "https://api.dataplatform.cloud.ibm.com",
            "https://private.jp-tok.ml.cloud.ibm.com": "https://api.jp-tok.dataplatform.cloud.ibm.com",
            "https://private.eu-gb.ml.cloud.ibm.com": "https://api.eu-gb.dataplatform.cloud.ibm.com",
            "https://private.eu-de.ml.cloud.ibm.com": "https://api.eu-de.dataplatform.cloud.ibm.com",
            "https://private.us-south.ml.cloud.ibm.com": "https://api.dataplatform.cloud.ibm.com",
            "https://private.us-south.ml.test.cloud.ibm.com": "https://api.dataplatform.dev.cloud.ibm.com",
        }

        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

        if self.credentials.token is not None:
            if not (self.credentials.api_key or self.credentials.password):
                self.proceed = True
            else:
                # _is_env_token is used for initialising client on cluster with USER_ACCESS_TOKEN environment variable.
                self.proceed = not self.credentials._is_env_token
        else:
            self.proceed = False

        self.token: str | None = None
        self.__predefined_instance_type_list = ["icp", "openshift"]
        if credentials is None:
            raise NoWMLCredentialsProvided()
        if self.credentials.url is None:
            raise WMLClientError(Messages.get_message(message_id="url_not_provided"))
        if not self.credentials.url.startswith("https://"):
            raise WMLClientError(Messages.get_message(message_id="invalid_url"))
        if self.credentials.url[-1] == "/":
            self.credentials.url = self.credentials.url.rstrip("/")
        with self._session:
            if self.credentials.instance_id is None:
                self.CLOUD_PLATFORM_SPACES = True
                self.ICP_PLATFORM_SPACES = False

                if self._internal:
                    self.PLATFORM_URL = self.credentials.url

                else:
                    if self.credentials.platform_url:
                        self.PLATFORM_URL = self.credentials.platform_url
                    elif self.credentials.url in self.PLATFORM_URLS_MAP.keys():
                        self.PLATFORM_URL = self.PLATFORM_URLS_MAP[self.credentials.url]
                    else:
                        raise WMLClientError(
                            Messages.get_message(
                                message_id="invalid_cloud_scenario_url"
                            )
                        )

                if not self._is_IAM():
                    raise WMLClientError(
                        Messages.get_message(message_id="apikey_not_provided")
                    )
            else:
                if (
                    "icp" == self.credentials.instance_id.lower()
                    or "openshift" == self.credentials.instance_id.lower()
                ):
                    self.ICP_PLATFORM_SPACES = True
                    os.environ["DEPLOYMENT_PLATFORM"] = "private"

                    # Validate the cpd version:
                    response_get_wml_services = self._session.get(
                        f"{self.credentials.url}/ml/wml_services/version",
                        headers={"User-Agent": get_user_agent_header()},
                    )
                    if (
                        response_get_wml_services.status_code != 200
                    ):  # retry with endpoint for cpd 4.8 and higher
                        response_get_wml_services = self._session.get(
                            f"{self.credentials.url}/ml/wml_services/v2/version",
                            headers={"User-Agent": get_user_agent_header()},
                        )

                    if response_get_wml_services.status_code == 200:
                        wml_version = response_get_wml_services.json().get(
                            "version", ""
                        )
                        if wml_version:
                            wml_version = ".".join(wml_version.split(".")[:2])
                            if self.credentials.version is None:
                                self.credentials.version = wml_version
                            elif self.credentials.version != wml_version:
                                warn(
                                    f"The provided version: {self.credentials.version} "
                                    f"is different from the current CP4D version: {wml_version}. "
                                    f"Correct the credentials with proper CP4D version number."
                                )

                            if (
                                self.credentials.version
                                not in CPDVersion.supported_version_list
                            ):
                                raise WMLClientError(
                                    Messages.get_message(
                                        self.credentials.version,
                                        self.version,
                                        message_id="invalid_version_from_automated_check",
                                    )
                                )
                    else:
                        self._logger.debug(
                            f"GET /ml/wml_services/version failed with status code: {response_get_wml_services.status_code}."
                        )

                    # Condition for CAMS related changes to take effect (Might change)
                    if self.credentials.version is None:
                        raise WMLClientError(
                            Messages.get_message(
                                CPDVersion.supported_version_list,
                                message_id="version_not_provided",
                            )
                        )

                    if (
                        self.credentials.version.lower()
                        in CPDVersion.supported_version_list
                    ):
                        self.CPD_version.cpd_version = self.credentials.version.lower()
                        os.environ["DEPLOYMENT_PRIVATE"] = "icp4d"

                        if self.credentials.bedrock_url is None and self.CPD_version:
                            if self.CPD_version < 4.7:
                                bedrock_prefix = "https://cp-console"
                            else:
                                namespace_from_url = "-".join(
                                    self.credentials.url.split(".")[0].split("-")[1:]
                                )
                                route = (
                                    "cpd" if self.CPD_version >= 5.1 else "cp-console"
                                )
                                bedrock_prefix = f"https://{route}-{namespace_from_url}"
                            self.credentials.bedrock_url = ".".join(
                                [bedrock_prefix] + self.credentials.url.split(".")[1:]
                            )
                            self._is_bedrock_url_autogenerated = True

                    else:
                        self.ICP_PLATFORM_SPACES = False
                        raise WMLClientError(
                            Messages.get_message(
                                ", ".join(CPDVersion.supported_version_list),
                                message_id="invalid_version",
                            )
                        )

                else:
                    if self.credentials.url in self.PLATFORM_URLS_MAP:
                        raise WMLClientError(
                            Messages.get_message(
                                message_id="instance_id_in_cloud_scenario"
                            )
                        )
                    else:
                        raise WMLClientError(
                            Messages.get_message(message_id="invalid_instance_id")
                        )

            if (
                self.credentials.instance_id is not None
                and (
                    self.credentials.instance_id.lower()
                    not in self.__predefined_instance_type_list
                )
                and self.credentials.version is not None
            ):
                raise WMLClientError(
                    Messages.get_message(message_id="provided_credentials_are_invalid")
                )
            self._use_fm_ga_api = self.CLOUD_PLATFORM_SPACES or (
                self._check_if_fm_ga_api_available()
                if self.CPD_version <= 4.8
                else True
            )

            self._use_pta_ga_api = self.CLOUD_PLATFORM_SPACES or (
                self.CPD_version >= 5.0
            )
            self.token = None

            # For cloud, service_instance.details will be set during space creation( if instance is associated ) or
            # while patching a space with an instance

            self.service_instance: ServiceInstance = ServiceInstance(self)
            self.volumes = Volume(self)
            if self._use_fm_ga_api:
                self.foundation_models = FoundationModelsManager(self)

            if self.ICP_PLATFORM_SPACES:
                self.service_instance._refresh_details = True

            self.set = Set(self)

            if project_id:
                self.set.default_project(project_id)  # recognizes project type
            elif space_id:
                self.set.default_space(space_id)

            self.spaces = Spaces(self)

            self.export_assets = Export(self)
            self.import_assets = Import(self)

            if self.ICP_PLATFORM_SPACES:
                self.shiny = Shiny(self)

            self.script = Script(self)
            self.model_definitions = ModelDefinition(self)

            self.package_extensions = PkgExtn(self)
            self.software_specifications = SwSpec(self)

            self.hardware_specifications = HwSpec(self)

            self.connections = Connections(self)
            self.training: Training = Training(self)

            self.data_assets = Assets(self)

            self.deployments = Deployments(self)

            if self.CLOUD_PLATFORM_SPACES:
                self.factsheets = Factsheets(self)
                self.task_credentials = TaskCredentials(self)

                # AI service available only on CLOUD
                self.__ai_services = AIServices(self)

            self.remote_training_systems = RemoteTrainingSystem(self)

            self.repository = Repository(self)
            self._models = Models(self)

            self.pipelines = Pipelines(self)
            self.experiments = Experiments(self)
            self._functions = Functions(self)

            self.parameter_sets = ParameterSets(self)
            self._logger.info(
                Messages.get_message(message_id="client_successfully_initialized")
            )

    @property
    def wml_credentials(self) -> dict[str, str]:
        warn(
            "`wml_credentials` attribute is deprecated, please use `client.credentials` instead"
        )
        return self.credentials.to_dict()

    @wml_credentials.setter
    def wml_credentials(self, value: dict[str, str]) -> None:
        warn(
            "`wml_credentials` attribute is deprecated, please use `client.credentials` instead"
        )
        self.credentials = Credentials.from_dict(value)

    @property
    def wml_token(self) -> str | None:
        warn("`wml_token` attribute is deprecated, please use `client.token` instead")
        return self.token

    @wml_token.setter
    def wml_token(self, value: str) -> None:
        warn("`wml_token` attribute is deprecated, please use `client.token` instead")
        self.token = value

    @property
    def _session(self) -> RequestsLikeType:
        if self.__session is None:
            self.__session = _APIClientSession(self)
        return self.__session

    @_session.setter
    def _session(self, value: RequestsLikeType):
        self.__session = value

    @property
    def _ai_services(self) -> AIServices:
        if self.CLOUD_PLATFORM_SPACES:
            return self.__ai_services
        else:
            raise WMLClientError(
                error_msg="AI service is unsupported for this release."
            )

    @staticmethod
    def _get_api_version_param() -> str:
        try:
            file_name = "API_VERSION_PARAM"
            path = os.path.dirname(ibm_watsonx_ai.utils.__file__)
            with open(os.path.join(path, file_name)) as file:
                return file.read().strip()
        except Exception:
            return "2021-06-21"

    def _check_if_either_is_set(self) -> None:
        if self.default_space_id is None and self.default_project_id is None:
            raise WMLClientError(
                Messages.get_message(
                    message_id="it_is_mandatory_to_set_the_space_project_id"
                )
            )

    def _check_if_space_is_set(self) -> None:
        if self.default_space_id is None:
            raise WMLClientError(
                Messages.get_message(message_id="it_is_mandatory_to_set_the_space_id")
            )

    def _params(
        self,
        skip_space_project_chk: bool = False,
        skip_for_create: bool = False,
        skip_userfs: bool = False,
    ) -> dict:
        params = {}
        params.update({"version": self.version_param})
        if not skip_for_create:
            if self.default_space_id is not None:
                params.update({"space_id": self.default_space_id})
            elif self.default_project_id is not None:
                params.update({"project_id": self.default_project_id})
            else:
                # For system software/hardware specs
                if skip_space_project_chk is False:
                    raise WMLClientError(
                        Messages.get_message(
                            message_id="it_is_mandatory_to_set_the_space_project_id"
                        )
                    )

        if (
            self.default_project_id
            and self.project_type == "local_git_storage"
            and not skip_userfs
        ):
            params.update({"userfs": "true"})
            if self._iam_id:
                params.update({"iam_id": str(self._iam_id)})

        if (
            not self.default_project_id
            or self.project_type != "local_git_storage"
            or skip_userfs
        ) and "userfs" in params:
            del params["userfs"]

        return params

    def _get_headers(
        self,
        content_type: str = "application/json",
        no_content_type: bool = False,
        zen: bool = False,
        projects_token: bool = False,
    ) -> dict:

        if zen:
            headers = {"Content-Type": content_type}
            token = self.service_instance._create_token()
            if len(token.split(".")) == 1:
                headers.update({"Authorization": "Basic " + token})

            else:
                headers.update({"Authorization": "Bearer " + token})
        else:
            if self.proceed is True:
                token_to_use = (
                    self.credentials.projects_token
                    if projects_token and self.credentials.projects_token is not None
                    else self.credentials.token
                )
                if len(token_to_use.split(".")) == 1:
                    token = "Basic " + token_to_use

                else:
                    token = "Bearer " + token_to_use
            else:
                token = "Bearer " + self.service_instance._get_token()
            headers = {
                "Authorization": token,
                "User-Agent": get_user_agent_header(),
            }

            if not self.ICP_PLATFORM_SPACES:
                if self.default_project_id is not None:
                    headers.update({"X-Watson-Project-ID": self.default_project_id})

            if not no_content_type:
                headers.update({"Content-Type": content_type})

        if not self.generate_ux_tag:
            headers.update({"X-WX-UX": "true"})
            self.generate_ux_tag = True

        if self.WCA:
            headers.update({"IBM-WATSONXAI-CONSUMER": "wca"})

        if (env_variable := os.environ.get("IBM_SDK_API_CLIENT_HEADERS")) is not None:
            headers = headers | json.loads(
                base64.b64decode(env_variable).decode("utf-8")
            )
        if self._user_headers:
            headers = headers | self._user_headers

        return headers

    def set_token(self, token: str) -> None:
        """
        Method which allows refresh/set new User Authorization Token.

        :param token: User Authorization Token
        :type token: str

        **Examples**

        .. code-block:: python

            client.set_token("<USER AUTHORIZATION TOKEN>")

        """
        self.proceed = True
        self.token = token
        self.credentials.token = token

    def set_headers(self, headers: dict) -> None:
        """
        Method which allows refresh/set new User Request Headers.

        :param headers: User Request Headers
        :type headers: dict

        **Examples**

        .. code-block:: python

            headers = {
                'Authorization': 'Bearer <USER AUTHORIZATION TOKEN>',
                'User-Agent': 'ibm-watsonx-ai/1.0.1 (lang=python; arch=x86_64; os=darwin; python.version=3.10.13)',
                'X-Watson-Project-ID': '<PROJECT ID>',
                'Content-Type': 'application/json'
            }

            client.set_headers(headers)

        """
        self._user_headers = headers

    def _get_icptoken(self) -> str:
        return self.service_instance._create_token()

    def _is_default_space_set(self) -> bool:
        if self.default_space_id is not None:
            return True
        return False

    def _is_IAM(self) -> bool:
        if self.credentials.api_key is not None:
            if self.credentials.api_key != "":
                return True
            else:
                raise WMLClientError(
                    Messages.get_message(message_id="apikey_value_cannot_be_empty")
                )
        elif self.credentials.token is not None:
            if self.credentials.token != "":
                return True
            else:
                raise WMLClientError(
                    Messages.get_message(message_id="token_value_cannot_be_empty")
                )
        else:
            return False

    def _check_if_fm_ga_api_available(self) -> bool:
        response_ga_api = self._session.get(
            url="{}/ml/v1/foundation_model_specs?limit={}".format(
                self.credentials.url, "1"
            ),
            params={"version": self.version_param},
        )
        return response_ga_api.status_code == 200
