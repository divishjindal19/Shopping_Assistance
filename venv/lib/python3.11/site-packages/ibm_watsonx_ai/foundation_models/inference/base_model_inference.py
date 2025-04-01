#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import json
import warnings
import copy

from functools import partial
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generator, Any, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields

import httpx
import requests as _requests

from ibm_watsonx_ai.foundation_models.utils.utils import (
    HAPDetectionWarning,
    PIIDetectionWarning,
)
from ibm_watsonx_ai._wrappers.requests import get_httpx_client, get_async_client
from ibm_watsonx_ai.foundation_models.schema import (
    TextChatParameters,
    TextGenParameters,
    BaseSchema,
)
import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.wml_client_error import WMLClientError

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient

__all__ = ["BaseModelInference"]

_RETRY_STATUS_CODES = [429, 503, 504, 520]


class BaseModelInference(WMLResource, ABC):
    """Base interface class for the model interface."""

    def __init__(
        self, name: str, client: APIClient, persistent_connection: bool = True
    ):
        self._persistent_connection = persistent_connection

        WMLResource.__init__(self, name, client)
        if self._persistent_connection:
            self._http_client = get_httpx_client(
                transport=requests.HTTPXRetryTransport(
                    _retry_status_codes=_RETRY_STATUS_CODES,
                    verify=client.credentials.verify,
                    limits=requests.HTTPX_DEFAULT_LIMIT,
                )
            )
        else:
            self._http_client = requests

        self._async_http_client = get_async_client(
            transport=requests.AsyncHTTPXRetryTransport(
                _retry_status_codes=_RETRY_STATUS_CODES,
                verify=self._client.credentials.verify,
                limits=requests.HTTPX_DEFAULT_LIMIT,
            )
        )

    @abstractmethod
    def get_details(self) -> dict:
        """Get model interface's details

        :return: details of model or deployment
        :rtype: dict
        """
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        messages: list[dict] | None = None,
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
    ) -> dict:
        """
        Given a messages as input, and parameters the selected inference
        will generate a chat response.
        """
        raise NotImplementedError

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict] | None = None,
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
    ) -> Generator:
        """
        Given a messages as input, and parameters the selected inference
        will generate a chat as generator.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str | list | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = 10,
        async_mode: bool = False,
        validate_prompt_variables: bool = True,
    ) -> dict | list[dict] | Generator:
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generated_text response.
        """
        raise NotImplementedError

    @abstractmethod
    async def _agenerate_single(
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
    ) -> dict:
        """
        Given a text prompt as input, and parameters the selected inference
        will return async generator with response.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_text_stream(
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
    ) -> Generator:
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generator.
        """
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, prompt: str, return_tokens: bool = False) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_identifying_params(self) -> dict:
        """Represent Model Inference's setup in dictionary"""
        raise NotImplementedError

    def _prepare_chat_payload(
        self,
        messages: list[dict] | None,
        params: dict | TextChatParameters | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: str | None = None,
    ) -> dict:
        raise NotImplementedError

    def _prepare_inference_payload(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> dict:
        raise NotImplementedError

    def _prepare_beta_inference_payload(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> dict:
        raise NotImplementedError

    def _send_inference_payload(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None,
        generate_url: str,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        _http_client: requests.HTTPXClient | None = None,
    ) -> dict:
        if self._client._use_fm_ga_api:
            payload = self._prepare_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        else:  # Remove on CPD 5.0 release
            payload = self._prepare_beta_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        http_client = _http_client or self._http_client

        post_params: dict[str, Any] = dict(
            url=generate_url,
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )
        if not isinstance(http_client, httpx.Client):
            post_params["_retry_status_codes"] = _RETRY_STATUS_CODES

        response_scoring = http_client.post(**post_params)

        return self._handle_response(
            200,
            "generate",
            response_scoring,
            _field_to_hide="generated_text",
        )

    def _send_chat_payload(
        self,
        messages: list[dict] | None,
        params: dict | TextChatParameters | None,
        generate_url: str,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: str | None = None,
    ) -> dict:
        payload = self._prepare_chat_payload(
            messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

        post_params: dict[str, Any] = dict(
            url=generate_url,
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )
        if not isinstance(self._http_client, httpx.Client):
            post_params["_retry_status_codes"] = _RETRY_STATUS_CODES

        response_scoring = self._http_client.post(**post_params)

        return self._handle_response(
            200,
            "generate",
            response_scoring,
            _field_to_hide="generated_chat",
        )

    async def _asend_inference_payload(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None,
        generate_url: str,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        async_client: requests.HTTPXAsyncClient | None = None,
    ) -> dict:
        if self._client._use_fm_ga_api:
            payload = self._prepare_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        else:  # Remove on CPD 5.0 release
            payload = self._prepare_beta_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )

        _async_http_client = async_client or self._async_http_client
        response = await _async_http_client.post(
            url=generate_url,
            json=payload,
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )

        return self._handle_response(200, "agenerate", response)

    def _generate_chat_stream_with_url(
        self,
        messages: list[dict] | None,
        params: dict | TextChatParameters | None,
        chat_stream_url: str,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: str | None = None,
    ) -> Generator:

        payload = self._prepare_chat_payload(
            messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

        if isinstance(self._http_client, requests.HTTPXClient):
            stream_function = self._http_client.post_stream
            kw_args: dict = dict(
                method="POST",
                url=chat_stream_url,
                json=payload,
                headers=self._client._get_headers(),
                params=self._client._params(skip_for_create=True, skip_userfs=True),
            )
        else:
            stream_function = requests.Session().post  # type: ignore[assignment]
            kw_args = dict(
                url=chat_stream_url,
                json=payload,
                headers=self._client._get_headers(),
                params=self._client._params(skip_for_create=True, skip_userfs=True),
                stream=True,
                _retry_status_codes=_RETRY_STATUS_CODES,
            )

        with stream_function(**kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = (
                    resp.iter_lines()
                    if isinstance(resp, httpx.Response)
                    else resp.iter_lines(decode_unicode=False)  # type: ignore[call-arg]
                )
                for chunk in resp_iter:
                    if isinstance(resp, _requests.Response):
                        chunk = chunk.decode("utf-8")
                    if "delta" in chunk:
                        response = chunk.replace("data: ", "")
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            raise Exception(f"Could not parse {response} as json")
                        yield parsed_response

            elif resp.status_code != 200:
                if isinstance(resp, httpx.Response):
                    resp.read()
                raise WMLClientError(
                    f"Request failed with: {resp.text} ({resp.status_code})"
                )

    def _generate_with_url(
        self,
        prompt: list | str | None,
        params: dict | TextGenParameters | None,
        generate_url: str,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = 10,
    ) -> list | dict:
        """
        Helper method which implements multi-threading for with passed generate_url.
        """
        if isinstance(prompt, list):
            try:
                # For batch of prompts use keep-alive connection even if persistent_connection=False
                http_client: requests.HTTPXClient = (
                    get_httpx_client(
                        transport=requests.HTTPXRetryTransport(
                            _retry_status_codes=_RETRY_STATUS_CODES,
                            verify=self._client.credentials.verify,
                            limits=requests.HTTPX_DEFAULT_LIMIT,
                        )
                    )
                    if not self._persistent_connection
                    else self._http_client
                )
                _send_inference_payload_partial = partial(
                    self._send_inference_payload,
                    params=params,
                    generate_url=generate_url,
                    guardrails=guardrails,
                    guardrails_hap_params=guardrails_hap_params,
                    guardrails_pii_params=guardrails_pii_params,
                    _http_client=http_client,
                )
                if (prompt_length := len(prompt)) <= concurrency_limit:
                    with ThreadPoolExecutor(max_workers=prompt_length) as executor:
                        generated_responses = list(
                            executor.map(_send_inference_payload_partial, prompt)
                        )
                else:
                    with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                        generated_responses = list(
                            executor.map(_send_inference_payload_partial, prompt)
                        )
                return generated_responses
            except Exception:
                http_client = None  # type: ignore[assignment]
                raise
            finally:
                if not self._persistent_connection and isinstance(
                    http_client, httpx.Client
                ):
                    http_client.close()

        else:

            response = self._send_inference_payload(
                prompt,
                params,
                generate_url,
                guardrails,
                guardrails_hap_params,
                guardrails_pii_params,
            )
        return response

    def _generate_with_url_async(
        self,
        prompt: list | str | None,
        params: dict | TextGenParameters | None,
        generate_url: str,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = 10,
    ) -> Generator:
        """
        Helper method which implements multi-threading for with passed generate_url.
        """
        async_params = params or {}
        async_params = copy.deepcopy(async_params)

        if isinstance(async_params, BaseSchema):
            async_params = async_params.to_dict()

        async_params["return_options"] = {"input_text": True}
        if isinstance(prompt, list):
            try:
                # For batch of prompts use keep-alive connection even if persistent_connection=False
                http_client: requests.HTTPXClient = (
                    get_httpx_client(
                        transport=requests.HTTPXRetryTransport(
                            _retry_status_codes=_RETRY_STATUS_CODES,
                            verify=self._client.credentials.verify,
                            limits=requests.HTTPX_DEFAULT_LIMIT,
                        )
                    )
                    if not self._persistent_connection
                    else self._http_client
                )
                _send_inference_payload_partial = partial(
                    self._send_inference_payload,
                    params=async_params,
                    generate_url=generate_url,
                    guardrails=guardrails,
                    guardrails_hap_params=guardrails_hap_params,
                    guardrails_pii_params=guardrails_pii_params,
                    _http_client=http_client,
                )
                if (prompt_length := len(prompt)) <= concurrency_limit:
                    with ThreadPoolExecutor(max_workers=prompt_length) as executor:
                        generate_futures = [
                            executor.submit(
                                _send_inference_payload_partial, single_prompt
                            )
                            for single_prompt in prompt
                        ]
                        try:
                            for future in as_completed(generate_futures):
                                yield future.result()
                        except:
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise

                else:
                    with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                        generate_futures = [
                            executor.submit(
                                _send_inference_payload_partial, single_prompt
                            )
                            for single_prompt in prompt
                        ]
                        try:
                            for future in as_completed(generate_futures):
                                yield future.result()
                        except:
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
            except Exception:
                http_client = None  # type: ignore[assignment]
                raise
            finally:
                if not self._persistent_connection and isinstance(
                    http_client, httpx.Client
                ):
                    http_client.close()

        else:

            response = self._send_inference_payload(
                prompt,
                async_params,
                generate_url,
                guardrails,
                guardrails_hap_params,
                guardrails_pii_params,
            )
            yield response

    def _generate_stream_with_url(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None,
        generate_stream_url: str,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> Generator:
        if self._client._use_fm_ga_api:
            payload = self._prepare_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        else:  # Remove on CPD 5.0 release
            payload = self._prepare_beta_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )

        if isinstance(self._http_client, requests.HTTPXClient):
            stream_function = self._http_client.post_stream
            kw_args: dict = dict(
                method="POST",
                url=generate_stream_url,
                json=payload,
                headers=self._client._get_headers(),
                params=self._client._params(skip_for_create=True, skip_userfs=True),
            )
        else:
            stream_function = requests.Session().post  # type: ignore[assignment]
            kw_args = dict(
                url=generate_stream_url,
                json=payload,
                headers=self._client._get_headers(),
                params=self._client._params(skip_for_create=True, skip_userfs=True),
                stream=True,
                _retry_status_codes=_RETRY_STATUS_CODES,
            )

        with stream_function(**kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = (
                    resp.iter_lines()
                    if isinstance(resp, httpx.Response)
                    else resp.iter_lines(decode_unicode=False)  # type: ignore[call-arg]
                )
                for chunk in resp_iter:
                    if isinstance(resp, _requests.Response):
                        chunk = chunk.decode("utf-8")
                    if "generated_text" in chunk:
                        response = chunk.replace("data: ", "")
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            raise Exception(f"Could not parse {response} as json")
                        if raw_response:
                            yield parsed_response
                            continue
                        yield self._return_guardrails_stats(parsed_response)[
                            "generated_text"
                        ]

            elif resp.status_code != 200:
                if isinstance(resp, httpx.Response):
                    resp.read()
                raise WMLClientError(
                    f"Request failed with: {resp.text} ({resp.status_code})"
                )

    def _tokenize_with_url(
        self,
        prompt: str,
        tokenize_url: str,
        return_tokens: bool,
    ) -> dict:
        payload = self._prepare_inference_payload(prompt)

        parameters = payload.get("parameters", {})
        parameters.update({"return_tokens": return_tokens})
        payload["parameters"] = parameters

        post_params: dict[str, Any] = dict(
            url=tokenize_url,
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )
        if not isinstance(self._http_client, httpx.Client):
            post_params["_retry_status_codes"] = _RETRY_STATUS_CODES

        response_scoring = self._http_client.post(**post_params)

        if response_scoring.status_code == 404:
            raise WMLClientError("Tokenize is not supported for this release")
        return self._handle_response(200, "tokenize", response_scoring)

    def _return_guardrails_stats(self, single_response: dict) -> dict:
        results = single_response["results"][0]
        hap_details = (
            results.get("moderations", {}).get("hap")
            if self._client._use_fm_ga_api
            else results.get("moderation", {}).get("hap")
        )  # Remove 'else' on CPD 5.0 release
        if hap_details:
            if hap_details[0].get("input"):
                warnings.warn(
                    next(
                        warning.get("message")
                        for warning in single_response.get("system", {}).get("warnings")
                        if warning.get("id") == "UNSUITABLE_INPUT"
                    ),
                    category=HAPDetectionWarning,
                )
            else:
                warnings.warn(
                    f"Potentially harmful text detected: {hap_details}",
                    category=HAPDetectionWarning,
                )
        pii_details = (
            results.get("moderations", {}).get("pii")
            if self._client._use_fm_ga_api
            else results.get("moderation", {}).get("pii")
        )  # Remove 'else' on CPD 5.0 release
        if pii_details:
            if pii_details[0].get("input"):
                warnings.warn(
                    next(
                        warning.get("message")
                        for warning in single_response.get("system", {}).get("warnings")
                        if warning.get("id") == "UNSUITABLE_INPUT"
                    ),
                    category=PIIDetectionWarning,
                )
            else:
                warnings.warn(
                    f"Personally identifiable information detected: {pii_details}",
                    category=PIIDetectionWarning,
                )
        return results

    @staticmethod
    def _update_moderations_params(additional_params: dict) -> dict:
        default_params = {"input": {"enabled": True}, "output": {"enabled": True}}
        if additional_params:
            for key, value in default_params.items():
                if key in additional_params:
                    if additional_params[key]:
                        if "threshold" in additional_params:
                            default_params[key]["threshold"] = additional_params[
                                "threshold"
                            ]
                    else:
                        default_params[key]["enabled"] = False
                else:
                    if "threshold" in additional_params:
                        default_params[key]["threshold"] = additional_params[
                            "threshold"
                        ]
            if "mask" in additional_params:
                default_params.update({"mask": additional_params["mask"]})
        return default_params

    @staticmethod
    def _validate_and_overwrite_params(
        params: dict[str, Any], valid_param: TextChatParameters | TextGenParameters
    ) -> dict[str, Any]:
        """Validate and fix parameters"""
        chat_valid_params = {field.name.lower() for field in fields(valid_param)}
        valid_params = {}
        invalid_params = {}

        for param, value in params.items():
            if param.lower() in chat_valid_params:
                valid_params[param] = value
            else:
                invalid_params[param] = value

        if invalid_params:
            warnings.warn(
                f"Parameters [{', '.join(invalid_params)}] is/are not recognized and will be ignored.",
                UserWarning,
            )

        return valid_params
