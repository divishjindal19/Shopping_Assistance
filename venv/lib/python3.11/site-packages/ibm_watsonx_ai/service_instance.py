#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
import base64
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from warnings import warn

from ibm_watsonx_ai._wrappers import requests

from ibm_watsonx_ai.href_definitions import HrefDefinitions
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    ApiRequestFailure,
    NoWMLCredentialsProvided,
    CannotAutogenerateBedrockUrl,
    InvalidCredentialsError,
)

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class ServiceInstance:
    """Connect, get details, and check usage of a Watson Machine Learning service instance."""

    def __init__(self, client: APIClient) -> None:
        self._logger = logging.getLogger(__name__)
        self._client = client
        self._credentials = client.credentials
        self._expiration_datetime: datetime | None = None
        self._min_expiration_time = timedelta(minutes=15)

        self._instance_id = self._client.credentials.instance_id

        if self._client.ICP_PLATFORM_SPACES:
            if self.get_instance_id() == "openshift":
                self._credentials.url = self.get_url()
            else:
                self._credentials.url = self.get_url() + ":31843"

        # This is used in connections.py
        self._href_definitions = HrefDefinitions(
            self._client,
            self._client.CLOUD_PLATFORM_SPACES,
            self._client.PLATFORM_URL,
            self._client.ICP_PLATFORM_SPACES,
        )

        self._client.token = self._get_token()

        if not self._client.proceed:  # there is no 'token' in credentials
            delta = self._get_expiration_datetime() - datetime.now()
            if delta < self._min_expiration_time:
                self._min_expiration_time = (
                    delta - timedelta(minutes=1)
                    if delta > timedelta(minutes=1)
                    else delta
                )

        # ml_repository_client is initialized in repo
        self._details = None
        self._refresh_details = False

    @property
    def instance_id(self):
        if self._instance_id is None:
            raise WMLClientError(
                (
                    "instance_id for this plan is picked up from the space or project with which "
                    "this instance_id is associated with. Set the space or project with associated "
                    "instance_id to be able to use this function"
                )
            )
        return self._instance_id

    @property
    def details(self):
        warn(
            "Attribute `details` is deprecated. Please use method `get_details()` instead.",
            category=DeprecationWarning,
        )
        if self._details is None or self._refresh_details:
            self._details = self.get_details()
            self._refresh_details = False
        return self._details

    @details.setter
    def details(self, value: dict | None):
        self._details = value

    def get_instance_id(self) -> str:
        """Get the instance ID of a Watson Machine Learning service.

        :return: ID of the instance
        :rtype: str

        **Example:**

        .. code-block:: python

            instance_details = client.service_instance.get_instance_id()
        """
        if self._instance_id is None:
            raise WMLClientError(
                "instance_id for this plan is picked up from the space or project with which "
                "this instance_id is associated with. Set the space or project with associated "
                "instance_id to be able to use this function"
            )

        return self.instance_id

    def get_api_key(self) -> str:
        """Get the API key of a Watson Machine Learning service.

        :return: API key
        :rtype: str

        **Example:**

        .. code-block:: python

            instance_details = client.service_instance.get_api_key()
        """
        return self._credentials.api_key

    def get_url(self) -> str:
        """Get the instance URL of a Watson Machine Learning service.

        :return: URL of the instance
        :rtype: str

        **Example:**

        .. code-block:: python

            instance_details = client.service_instance.get_url()
        """
        return self._credentials.url

    def get_username(self) -> str:
        """Get the username for the Watson Machine Learning service. Applicable only for IBM Cloud Pak® for Data.

        :return: username
        :rtype: str

        **Example:**

        .. code-block:: python

            instance_details = client.service_instance.get_username()
        """
        if self._client.ICP_PLATFORM_SPACES:
            if self._credentials.username is not None:
                return self._credentials.username
            else:
                raise WMLClientError("`username` missing in credentials.")
        else:
            raise WMLClientError("Not applicable for Cloud")

    def get_password(self) -> str:
        """Get the password for the Watson Machine Learning service. Applicable only for IBM Cloud Pak® for Data.

        :return: password
        :rtype: str

        **Example:**

        .. code-block:: python

            instance_details = client.service_instance.get_password()
        """
        if self._client.ICP_PLATFORM_SPACES:
            if self._credentials.password is not None:
                return self._credentials.password
            else:
                raise WMLClientError("`password` missing in credentials.")
        else:
            raise WMLClientError("Not applicable for Cloud")

    def get_details(self) -> dict:
        """Get information about the Watson Machine Learning instance.

        :return: metadata of the service instance
        :rtype: dict

        **Example:**

        .. code-block:: python

            instance_details = client.service_instance.get_details()

        """

        if self._client.CLOUD_PLATFORM_SPACES:
            if self._credentials is not None:

                if self._instance_id is None:
                    raise WMLClientError(
                        "instance_id for this plan is picked up from the space or project with which "
                        "this instance_id is associated with. Set the space or project with associated "
                        "instance_id to be able to use this function"
                    )

                    # /ml/v4/instances will need either space_id or project_id as mandatory params
                # We will enable this service instance class only during create space or
                # set space/project. So, space_id/project_id would have been populated at this point
                headers = self._client._get_headers()

                del headers["User-Agent"]
                if "ML-Instance-ID" in headers:
                    headers.pop("ML-Instance-ID")
                response_get_instance = self._client._session.get(
                    self._href_definitions.get_v4_instance_id_href(self.instance_id),
                    params=self._client._params(skip_space_project_chk=True),
                    headers=headers,
                )

                if response_get_instance.status_code == 200:
                    return response_get_instance.json()
                else:
                    raise ApiRequestFailure(
                        "Getting instance details failed.", response_get_instance
                    )
            else:
                raise NoWMLCredentialsProvided
        else:
            return {}

    def _get_token(self) -> str:
        if self._client.token is None:
            return self._create_token()

        if self._is_token_refresh_possible():
            if self._client.ICP_PLATFORM_SPACES:
                if self._get_expiration_datetime():
                    if (
                        self._get_expiration_datetime() - timedelta(minutes=50)
                        < datetime.now()
                    ):
                        self._client.token = self._get_cpd_token_from_request()
                        self._client.repository._refresh_repo_client()
                else:
                    self._client.token = self._get_cpd_token_from_request()
                    self._client.repository._refresh_repo_client()
            elif self._client._is_IAM():
                if (
                    self._get_expiration_datetime() - self._min_expiration_time
                    < datetime.now()
                ):
                    self._client.token = self._get_IAM_token()
                    self._client.repository._refresh_repo_client()
            elif (
                self._get_expiration_datetime() - timedelta(minutes=30) < datetime.now()
            ):
                self._client.repository._refresh_repo_client()
                self._refresh_token()

        return self._client.token

    def _create_token(self) -> str:

        if self._client.proceed is True:
            return self._credentials.token

        if self._client.CLOUD_PLATFORM_SPACES:
            if self._client._is_IAM():
                return self._get_IAM_token()
            else:
                raise WMLClientError(
                    "api_key for IAM token is not provided in credentials for the client."
                )
        else:
            return self._get_cpd_token_from_request()

    def _refresh_token(self) -> None:
        if self._client.proceed is True:
            self._client.token = self._credentials.token

        self._client.token = self._get_cpd_token_from_request()

    def _get_expiration_datetime(self) -> datetime:
        if self._expiration_datetime is not None:
            return self._expiration_datetime

        token_parts = self._client.token.split(".")
        token_padded = token_parts[1] + "==="
        try:
            token_info = json.loads(
                base64.b64decode(token_padded).decode("utf-8", errors="ignore")
            )
        except ValueError:
            # If there is a problem with decoding (e.g. special char in token), add altchars
            token_info = json.loads(
                base64.b64decode(token_padded, altchars="_-").decode(
                    "utf-8", errors="ignore"
                )
            )
        token_expire = token_info.get("exp")

        return datetime.fromtimestamp(token_expire)

    def _is_iam(self) -> str | bool:
        try:
            token_parts = self._client.token.split(".")
            token_padded = token_parts[1] + "==="
            try:
                token_info = json.loads(
                    base64.b64decode(token_padded).decode("utf-8", errors="ignore")
                )
            except ValueError:
                # If there is a problem with decoding (e.g. special char in token), add altchars
                token_info = json.loads(
                    base64.b64decode(token_padded, altchars="_-").decode(
                        "utf-8", errors="ignore"
                    )
                )
            instanceId = token_info.get("instanceId")

            return instanceId
        except:
            return False

    def _get_IAM_token(self) -> str:
        if self._client.proceed is True:
            return self._credentials.token
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic Yng6Yng=",
        }

        mystr = "apikey=" + self._href_definitions.get_iam_token_api()
        response = self._client._session.post(
            self._href_definitions.get_iam_token_url(), data=mystr, headers=headers
        )
        if response.status_code == 200:
            token = response.json().get("access_token")
            self._expiration_datetime = None
        else:
            raise WMLClientError("Error getting IAM Token.", response)
        return token

    def _is_token_refresh_possible(self) -> bool:
        """Check if necessary credentials were passed for the token refresh.
        For CP4D, you need the username and password, or the username and api_key.
        For Cloud, you need the api_key.

        :return: `True` if token refresh can be performed. `False` if it cannot.
        :rtype: bool
        """
        if self._client._is_IAM():
            return self._credentials.api_key is not None
        else:
            return self._credentials.username is not None and (
                self._credentials.password is not None
                or self._credentials.api_key is not None
            )

    def _get_cpd_auth_pair(self) -> str:
        """Get a pair of credentials required for the token generation.

        :return: string representing a dictionary of authentication credentials
                         (username & password) or (username & api_key).
        :rtype: str
        """
        if self._credentials.api_key is not None:
            return f'{{"username": "{self.get_username()}", "api_key": "{self.get_api_key()}"}}'
        else:
            return f'{{"username": "{self.get_username()}", "password": "{self.get_password()}"}}'

    def _get_cpd_bedrock_auth_data(self) -> str:
        """Get the data required for the token generation.

        :return: string representing a dictionary of authentication credentials
        :rtype: str
        """
        return f"grant_type=password&username={self.get_username()}&password={self.get_password()}&scope=openid"

    def _get_cpd_token_from_request_old_auth_flow(self) -> str:
        token_url = self._href_definitions.get_cpd_token_endpoint_href()
        response = self._client._session.post(
            token_url,
            headers={"Content-Type": "application/json"},
            data=self._get_cpd_auth_pair(),
        )

        if response.status_code == 200:
            self._expiration_datetime = None
            return response.json().get("token")
        else:
            raise InvalidCredentialsError(reason=response.text)

    def _get_cpd_token_from_request_new_auth_flow(self) -> str:
        bedrock_url = self._href_definitions.get_cpd_bedrock_token_endpoint_href()
        response = self._client._session.post(
            bedrock_url,
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
            data=self._get_cpd_bedrock_auth_data(),
        )

        if response.status_code != 200:
            raise InvalidCredentialsError(reason=response.text, logg_messages=False)

        iam_token = response.json()["access_token"]
        self._expiration_datetime = datetime.now() + timedelta(
            seconds=response.json()["expires_in"]
        )
        # refresh_token = response.json()['refresh_token']

        token_url = self._href_definitions.get_cpd_validation_token_endpoint_href()
        response = self._client._session.get(
            token_url,
            headers={"username": self.get_username(), "iam-token": iam_token},
        )

        if response.status_code == 200:
            return response.json()["accessToken"]
        else:
            raise InvalidCredentialsError(reason=response.text)

    def _get_cpd_token_from_request(self) -> str:
        """Send a request for the token on CPD.

        :return: newly created token is returned if no errors occurred
        :rtype: str
        """
        if (
            self._client.ICP_PLATFORM_SPACES
            and self._client.credentials.bedrock_url is not None
            and self._client.credentials.password is not None
        ):
            try:
                return self._get_cpd_token_from_request_new_auth_flow()
            except Exception as e1:
                if not hasattr(self._client, "_is_bedrock_url_autogenerated"):
                    raise e1

                try:
                    res = self._get_cpd_token_from_request_old_auth_flow()
                    # if it worked when iamintegration=False, then removing bedrock_url will shorten the path
                    self._client.credentials.bedrock_url = None
                    return res
                except Exception as e2:
                    if (
                        hasattr(self._client, "_is_bedrock_url_autogenerated")
                        and self._client._is_bedrock_url_autogenerated
                    ):
                        raise CannotAutogenerateBedrockUrl(e1, e2)
                    else:
                        raise e2
        else:
            return self._get_cpd_token_from_request_old_auth_flow()
