#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING, cast, Any

from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    MissingExtension,
    UnsupportedOperation,
)
from ibm_watsonx_ai.utils.autoai.errors import ContainerTypeNotSupported
from ibm_watsonx_ai.helpers.connections import (
    DataConnection,
    ContainerLocation,
    S3Connection,
    S3Location,
    FSLocation,
    AssetLocation,
)
from ibm_watsonx_ai.utils.autoai.utils import is_ipython
from ibm_watsonx_ai.foundation_models.utils import FineTuningParams
from ibm_watsonx_ai.foundation_models.utils.utils import (
    _is_fine_tuning_endpoint_available,
)

import datetime
import numpy as np


if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from pandas import DataFrame


class FineTuner:
    id: str | None = None
    _client: APIClient = None  # type: ignore[assignment]
    _training_metadata: dict | None = None

    def __init__(
        self,
        name: str,
        task_id: str,
        api_client: APIClient,
        *,
        description: str | None = None,
        base_model: str | None = None,
        num_epochs: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        max_seq_length: int | None = None,
        accumulate_steps: int | None = None,
        verbalizer: str | None = None,
        response_template: str | None = None,
        gpu: dict | None = None,
        auto_update_model: bool = True,
        group_by_name: bool = False,
    ):
        self._client = api_client

        if not _is_fine_tuning_endpoint_available(self._client):
            raise UnsupportedOperation(
                Messages.get_message(message_id="fine_tuning_not_supported")
            )

        self.name = name
        self.description = description if description else "Fine tuning with SDK"
        self.auto_update_model = auto_update_model
        self.group_by_name = group_by_name

        base_model_red: dict = {"model_id": base_model}

        self.fine_tuning_params = FineTuningParams(
            base_model=base_model_red,
            task_id=task_id,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            accumulate_steps=accumulate_steps,
            verbalizer=verbalizer,
            response_template=response_template,
            gpu=gpu,
        )

        if not isinstance(self.name, str):
            raise WMLClientError(
                f"'name' param expected string, but got {type(self.name)}: {self.name}"
            )

        if not isinstance(self.fine_tuning_params.task_id, str):
            raise WMLClientError(
                f"'task_id' param expected string, but got {type(self.fine_tuning_params.task_id)}: {self.fine_tuning_params.task_id}"
            )

        if self.description and (not isinstance(self.description, str)):
            raise WMLClientError(
                f"'description' param expected string, but got {type(self.description)}: "
                f"{self.description}"
            )

        if self.auto_update_model and (not isinstance(self.auto_update_model, bool)):
            raise WMLClientError(
                f"'auto_update_model' param expected bool, but got {type(self.auto_update_model)}: "
                f"{self.auto_update_model}"
            )

        if self.group_by_name and (not isinstance(self.group_by_name, bool)):
            raise WMLClientError(
                f"'group_by_name' param expected bool, but got {type(self.group_by_name)}: "
                f"{self.group_by_name}"
            )

    def run(
        self,
        training_data_references: list[DataConnection],
        training_results_reference: DataConnection | None = None,
        background_mode: bool = False,
    ) -> dict:
        """Run a fine-tuning process of foundation model on top of the training data referenced by DataConnection.

        :param training_data_references: data storage connection details to inform where training data is stored
        :type training_data_references: list[DataConnection]

        :param training_results_reference: data storage connection details to store pipeline training results
        :type training_results_reference: DataConnection, optional

        :param background_mode: indicator if fit() method will run in background (async) or (sync)
        :type background_mode: bool, optional

        :return: run details
        :rtype: dict

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment
            from ibm_watsonx_ai.helpers import DataConnection, S3Location

            experiment = TuneExperiment(credentials, ...)
            fine_tuner = experiment.fine_tuner(...)

            fine_tuner.run(
                training_data_references=[DataConnection(
                    connection_asset_id=connection_id,
                    location=S3Location(
                        bucket='fine_tuning_data',
                        path='ft_train_data.json')
                    )
                )]
                background_mode=False)
        """
        WMLResource._validate_type(
            training_data_references, "training_data_references", list, mandatory=True
        )
        WMLResource._validate_type(
            training_results_reference,
            "training_results_reference",
            object,
            mandatory=False,
        )

        for source_data_connection in [training_data_references]:
            if source_data_connection:
                self._validate_source_data_connections(source_data_connection)
        training_results_reference = self._determine_result_reference(
            results_reference=training_results_reference,
            data_references=training_data_references,
        )

        self._initialize_training_metadata(
            training_data_references,
            test_data_references=None,
            training_results_reference=training_results_reference,
        )

        self._training_metadata = cast(dict, self._training_metadata)
        tuning_details = self._client.training.run(
            meta_props=self._training_metadata,
            asynchronous=background_mode,
            _is_fine_tuning=True,
        )
        self.id = self._client.training.get_id(tuning_details)

        return self._client.training.get_details(self.id, _is_fine_tuning=True)

    def _initialize_training_metadata(
        self,
        training_data_references: list[DataConnection],
        test_data_references: list[DataConnection] | None = None,
        training_results_reference: DataConnection | None = None,
    ) -> None:
        self._training_metadata = {
            self._client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [
                connection._to_dict() for connection in training_data_references
            ],
            self._client.training.ConfigurationMetaNames.NAME: f"{self.name[:100]}",
            self._client.training.ConfigurationMetaNames.FINE_TUNING: self.fine_tuning_params.to_dict(),
        }
        if test_data_references:
            self._training_metadata[
                self._client.training.ConfigurationMetaNames.TEST_DATA_REFERENCES
            ] = [connection._to_dict() for connection in test_data_references]
        if training_results_reference:
            self._training_metadata[
                self._client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE
            ] = training_results_reference._to_dict()

        if self.description:
            self._training_metadata[
                self._client.training.ConfigurationMetaNames.DESCRIPTION
            ] = f"{self.description}"

        if self.auto_update_model is not None:
            self._training_metadata[
                self._client.training.ConfigurationMetaNames.AUTO_UPDATE_MODEL
            ] = self.auto_update_model

    def _validate_source_data_connections(
        self, source_data_connections: list[DataConnection]
    ) -> list[DataConnection]:
        for data_connection in source_data_connections:
            if isinstance(data_connection.location, ContainerLocation):
                if self._client.ICP_PLATFORM_SPACES:
                    raise ContainerTypeNotSupported()  # block Container type on CPD
                elif isinstance(data_connection.connection, S3Connection):
                    # note: remove S3 inline credential from data asset before training
                    data_connection.connection = None
                    if hasattr(data_connection.location, "bucket"):
                        delattr(data_connection.location, "bucket")
                    # --- end note
            if isinstance(data_connection.connection, S3Connection) and isinstance(
                data_connection.location, AssetLocation
            ):
                # note: remove S3 inline credential from data asset before training
                data_connection.connection = None

                for s3_attr in ["bucket", "path"]:
                    if hasattr(data_connection.location, s3_attr):
                        delattr(data_connection.location, s3_attr)
                # --- end note

        return source_data_connections

    def _determine_result_reference(
        self,
        results_reference: DataConnection | None,
        data_references: list[DataConnection],
        result_path: str = "default_tuning_output",
    ) -> DataConnection:
        # note: if user did not provide results storage information, use default ones
        if results_reference is None:
            if self._client.ICP_PLATFORM_SPACES:
                location = FSLocation(
                    path="/{option}/{id}/assets/wx_fine_tune"
                )  # TODO changed
                if self._client.default_project_id is None:
                    location.path = location.path.format(
                        option="spaces", id=self._client.default_space_id
                    )

                else:
                    location.path = location.path.format(
                        option="projects", id=self._client.default_project_id
                    )
                results_reference = DataConnection(connection=None, location=location)

            else:
                if isinstance(data_references[0].location, S3Location):
                    results_reference = DataConnection(
                        connection=data_references[0].connection,
                        location=S3Location(
                            bucket=data_references[0].location.bucket, path="."
                        ),
                    )

                elif isinstance(data_references[0].location, AssetLocation):
                    connection_id = data_references[0].location._get_connection_id(
                        self._client
                    )

                    if connection_id is not None:
                        results_reference = DataConnection(
                            connection_asset_id=connection_id,
                            location=S3Location(
                                bucket=data_references[0].location._get_bucket(
                                    self._client
                                ),
                                path=result_path,
                            ),
                        )

                    else:  # set container output location when default DAta Asset is as a train ref
                        results_reference = DataConnection(
                            location=ContainerLocation(path=result_path)
                        )

                else:
                    results_reference = DataConnection(
                        location=ContainerLocation(path=result_path)
                    )
        # -- end note

        # note: validate location types:
        if self._client.ICP_PLATFORM_SPACES:
            if not isinstance(results_reference.location, FSLocation):
                raise TypeError(
                    "Unsupported results location type. Results reference can be stored on FSLocation."
                )
        else:
            if not isinstance(
                results_reference.location, (S3Location, ContainerLocation)
            ):
                raise TypeError(
                    "Unsupported results location type. Results reference can be stored"
                    " only on S3Location or ContainerLocation."
                )
        # -- end note
        return results_reference

    @staticmethod
    def _get_average_loss_score_for_each_epoch(tuning_details: dict) -> list:
        scores = []
        temp_score = []
        epoch = 1  # FT starting on 1 epoch
        if "data" in tuning_details["entity"]["status"]["metrics"][0]:
            for ind, metric in enumerate(tuning_details["entity"]["status"]["metrics"]):
                if int(metric["data"]["epoch"]) == epoch:
                    temp_score.append(metric["data"]["value"])
                else:
                    epoch += 1
                    scores.append(np.average(temp_score))
                    temp_score = [metric["data"]["value"]]
            scores.append(np.average(temp_score))
        else:
            for ind, metric in enumerate(tuning_details["entity"]["status"]["metrics"]):
                if int(metric["ml_metrics"]["epoch"]) == epoch:
                    temp_score.append(metric["ml_metrics"]["loss"])
                else:
                    epoch += 1
                    scores.append(np.average(temp_score))
                    temp_score = [metric["ml_metrics"]["loss"]]
            scores.append(np.average(temp_score))
        return scores

    @staticmethod
    def _get_first_and_last_iteration_metrics_for_each_epoch(
        tuning_details: dict,
    ) -> list:
        first_and_last_iteration_metrics_for_each_epoch = []
        first_iteration = True

        tuning_metrics = tuning_details["entity"]["status"]["metrics"]
        for ind in range(len(tuning_metrics)):
            if ind == 0:
                first_and_last_iteration_metrics_for_each_epoch.append(
                    tuning_metrics[ind]
                )
                first_and_last_iteration_metrics_for_each_epoch.append(
                    tuning_metrics[ind]
                )
                first_iteration = False
            elif first_iteration:
                first_and_last_iteration_metrics_for_each_epoch.append(
                    tuning_metrics[ind]
                )
                first_iteration = False
            else:
                if (
                    tuning_metrics[ind].get(
                        "data", tuning_metrics[ind].get("ml_metrics")
                    )["epoch"]
                    == tuning_metrics[ind - 1].get(
                        "data", tuning_metrics[ind - 1].get("ml_metrics")
                    )["epoch"]
                ):
                    first_and_last_iteration_metrics_for_each_epoch.pop()
                    first_and_last_iteration_metrics_for_each_epoch.append(
                        tuning_metrics[ind]
                    )
                else:
                    first_and_last_iteration_metrics_for_each_epoch.append(
                        tuning_metrics[ind]
                    )
                    first_iteration = True
        return first_and_last_iteration_metrics_for_each_epoch

    def get_params(self) -> dict:
        """Get configuration parameters of FineTuner.

        :return: FineTuner parameters
        :rtype: dict

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            fine_tuner = experiment.fine_tuner(...)

            fine_tuner.get_params()

            # Result:
            #
            # {'base_model': {'model_id': 'bigscience/bloom-560m'},
            #  'name': 'Fine-Tuning of bloom-560m model',
            #  'auto_update_model': False,
            #  'group_by_name': False}
        """
        params = self.fine_tuning_params.to_dict()
        params["name"] = self.name
        params["description"] = self.description
        params["auto_update_model"] = self.auto_update_model
        params["group_by_name"] = self.group_by_name
        return params

    def get_run_status(self) -> str:
        """Check status/state of initialized Fine-Tuning run if ran in background mode.

        :return: Fine-tuning run status
        :rtype: str

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            fine_tuner = experiment.fine_tuner(...)
            fine_tuner.run(...)

            fine_tuner.get_run_details()

            # Result:
            # 'completed'
        """
        if self.id is None:
            raise WMLClientError(
                Messages.get_message(message_id="fm_fine_tuning_not_scheduled")
            )

        return self._client.training.get_status(training_id=self.id, _is_fine_tuning=True).get("state")  # type: ignore[return-value]

    def get_run_details(self, include_metrics: bool = False) -> dict:
        """Get fine-tuning run details.

        :param include_metrics: indicates to include metrics in the training details output
        :type include_metrics: bool, optional

        :return: Fine-tuning details
        :rtype: dict

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            fine_tuner = experiment.fine_tuner(...)
            fine_tuner.run(...)

            fine_tuner.get_run_details()
        """

        if self.id is None:
            raise WMLClientError(
                Messages.get_message(message_id="fm_fine_tuning_not_scheduled")
            )

        details = self._client.training.get_details(
            training_id=self.id, _is_fine_tuning=True
        )

        if include_metrics:
            try:
                details["entity"]["status"]["metrics"] = (
                    self._get_metrics_data_from_property_or_file(details)
                )
            except KeyError:
                pass
            finally:
                return details

        if details["entity"]["status"].get("metrics", False):
            del details["entity"]["status"]["metrics"]

        return details

    def _get_metrics_data_from_property_or_file(self, details: dict) -> dict:
        path = details["entity"]["status"]["metrics"][0]["context"]["fine_tuning"][
            "metrics_location"
        ]
        results_reference = details["entity"]["results_reference"]
        conn = DataConnection._from_dict(results_reference)
        conn._api_client = self._client
        metrics_data = conn._download_json_file(path, tuning_type="fine_tuning")

        return metrics_data

    def plot_learning_curve(self, **kwargs: Any) -> None:
        """Plot learning curves.

        .. note ::
            Available only for Jupyter notebooks.

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            fine_tuner = experiment.fine_tuner(...)
            fine_tuner.run(...)

            fine_tuner.plot_learning_curve()
        """
        skip_check = kwargs.get("skip_check", False)
        if not skip_check:
            if not is_ipython():
                raise WMLClientError(
                    "Function `plot_learning_curve` is available only for Jupyter notebooks."
                )
        from ibm_watsonx_ai.utils.autoai.incremental import plot_learning_curve

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise MissingExtension("matplotlib")

        tuning_details = self.get_run_details(include_metrics=True)

        if "metrics" in tuning_details["entity"]["status"]:
            # average loss score for each epoch
            scores = self._get_average_loss_score_for_each_epoch(
                tuning_details=tuning_details
            )

            # date_time from the first and last iteration on each epoch
            date_times = [
                datetime.datetime.strptime(
                    m_obj["data"]["timestamp"], "%Y-%m-%dT%H:%M:%S.%f"
                )
                for m_obj in self._get_first_and_last_iteration_metrics_for_each_epoch(
                    tuning_details=tuning_details
                )
            ]

            elapsed_time = []
            for i in range(1, len(date_times), 2):
                elapsed_time.append((date_times[i] - date_times[i - 1]).total_seconds())

            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            if scores:
                plot_learning_curve(
                    fig=fig,
                    axes=axes,
                    scores=scores,
                    fit_times=elapsed_time,
                    xlabels={"first_xlabel": "Epochs", "second_xlabel": "Epochs"},
                    titles={"first_plot": "Loss function"},
                )
        else:
            raise WMLClientError(
                Messages.get_message(message_id="fm_fine_tuning_no_metrics")
            )

    def summary(self, scoring: str = "loss") -> DataFrame:

        if self.id is None:
            raise WMLClientError(
                Messages.get_message(message_id="fm_fine_tuning_not_scheduled")
            )

        from pandas import DataFrame

        details = self.get_run_details(include_metrics=True)

        metrics = details["entity"]["status"].get("metrics", [{}])[0]
        is_ml_metrics = "data" in metrics or "ml_metrics" in metrics

        if not is_ml_metrics:
            raise WMLClientError(
                Messages.get_message(message_id="fm_fine_tuning_no_metrics")
            )
        columns = [
            "Model Name",
            "Enhancements",
            "Base model",
            "Auto store",
            "Epochs",
            scoring,
        ]
        values = []
        model_name = "model_" + self.id
        base_model_name = None
        epochs = None
        enhancements = []
        if scoring == "loss":
            model_metrics = [
                self._get_average_loss_score_for_each_epoch(tuning_details=details)[-1]
            ]
        else:
            model_metrics = [
                details["entity"]["status"]
                .get("metrics", [{}])[-1]
                .get("data", {})[scoring]
            ]

        if "parameters" in details["entity"]:
            enhancements = ["fine tuning"]
            base_model_name = details["entity"]["parameters"]["base_model"]["model_id"]
            epochs = details["entity"]["parameters"]["num_epochs"]

        values.append(
            (
                [model_name]
                + [enhancements]
                + [base_model_name]
                + [details["entity"]["auto_update_model"]]
                + [epochs]
                + model_metrics
            )
        )

        summary = DataFrame(data=values, columns=columns)
        summary.set_index("Model Name", inplace=True)

        return summary

    def get_model_id(self) -> str:
        """Get model id.

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            fine_tuner = experiment.fine_tuner(...)
            fine_tuner.run(...)

            fine_tuner.get_model_id()
        """
        run_details = self.get_run_details()
        if run_details["entity"]["auto_update_model"]:
            return run_details["entity"]["model_id"]
        else:
            raise WMLClientError(
                Messages.get_message(message_id="fm_fine_tuning_no_model_id")
            )

    def cancel_run(self, hard_delete: bool = False) -> None:
        """Cancels or deletes a Fine-Tuning run.

        :param hard_delete: When True then the completed or cancelled fine-tuning run is deleted,
                            if False then the current run is canceled. Default: False
        :type hard_delete: bool, optional
        """
        if self.id is None:
            raise WMLClientError(
                Messages.get_message(message_id="fm_fine_tuning_not_scheduled")
            )

        self._client.training.cancel(
            training_id=self.id, hard_delete=hard_delete, _is_fine_tuning=True
        )

    def get_data_connections(self) -> list[DataConnection]:
        """Create DataConnection objects for further user usage
            (eg. to handle data storage connection).

        :return: list of DataConnections
        :rtype: list['DataConnection']

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment
            experiment = TuneExperiment(credentials, ...)
            fine_tuner = experiment.fine_tuner(...)
            fine_tuner.run(...)

            data_connections = fine_tuner.get_data_connections()
        """
        training_data_references = self.get_run_details()["entity"][
            "training_data_references"
        ]

        data_connections = [
            DataConnection._from_dict(_dict=data_connection)
            for data_connection in training_data_references
        ]

        for data_connection in data_connections:
            data_connection.set_client(self._client)
            data_connection._run_id = self.id

        return data_connections
