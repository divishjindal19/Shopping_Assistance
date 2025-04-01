#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .prompt_template import PromptTemplateLock
from datetime import datetime


from ibm_watsonx_ai.wml_client_error import (
    ValidationError,
)
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.foundation_models.utils.utils import TemplateFormatter


class BasePromptTemplate(ABC):
    """Base class for Prompt Template Asset."""

    def __init__(self, input_mode: str, **kwargs: Any) -> None:
        self.name = kwargs.get("name")
        self.description = kwargs.get("description")
        self.task_ids = (
            task_ids.copy()
            if (task_ids := kwargs.get("task_ids")) is not None
            else task_ids
        )
        self.model_id: ModelTypes | str | None = kwargs.get("model_id")
        if isinstance(self.model_id, Enum):
            self.model_id = self.model_id.value
        self.model_params = (
            model_params.copy()
            if (model_params := kwargs.get("model_params")) is not None
            else model_params
        )
        self.template_version = kwargs.get("template_version")

        self._input_mode = input_mode
        self._prompt_id: str | None = None
        self._created_at: float | None = None
        self._lock: PromptTemplateLock | None = None
        self._is_template: bool | None = None
        self.input_text = kwargs.get("input_text")
        self.input_variables = (
            input_variables.copy()
            if (input_variables := kwargs.get("input_variables")) is not None
            else input_variables
        )

    @property
    def prompt_id(self) -> str | None:
        return self._prompt_id

    @property
    def created_at(self) -> str | None:
        if self._created_at is not None:
            return str(datetime.fromtimestamp(self._created_at / 1000)).split(".")[0]
        else:
            return None

    @property
    def lock(self) -> PromptTemplateLock | None:
        return self._lock

    @property
    def is_template(self) -> bool | None:
        return self._is_template

    def __repr__(self) -> str:
        args = [
            f"{key}={value!r}"
            for key, value in self.__dict__.items()
            if not key.startswith("_") and value is not None
        ]
        return f"{type(self).__name__}({ ', '.join(args)})"

    @abstractmethod
    def _validation(self) -> None:
        """Validate the consistency of the template structure with the provided input variables."""
        raise NotImplementedError

    def _validate_prompt(
        self, input_variables: Iterable[str], template_text: str
    ) -> None:
        """:raises ValidationError: When set of elements `input_variables` is not the same
        as set of placeholders in joined string input_text + template_text"""
        try:
            dummy_inputs = {input_variable: "wx" for input_variable in input_variables}
            TemplateFormatter().format(template_text, **dummy_inputs)
        except KeyError as key:
            raise ValidationError(
                str(key),
                additional_msg="One can turn off validation step setting `validate_template` to False.",
            )
