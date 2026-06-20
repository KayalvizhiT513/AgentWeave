from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from agentweave.config import Settings
from agentweave.core.models import Conversation


def format_conversation_transcript(conversation: Conversation) -> str:
    lines = [
        f"{exchange.role.value.title()}: {exchange.content.strip()}"
        for exchange in conversation.exchanges
    ]
    return "\n".join(lines)


def build_futureagi_dataset_row(
    conversation: Conversation,
    provider_mode: str,
) -> dict[str, Any]:
    return {
        "conversation_id": conversation.id,
        "topic": conversation.topic,
        "scene": conversation.scene,
        "constraints": conversation.constraints,
        "status": conversation.status.value,
        "current_round": conversation.current_round,
        "exchange_count": len(conversation.exchanges),
        "evaluation_count": len(conversation.evaluations),
        "replacement_count": len(conversation.replacements),
        "provider_mode": provider_mode,
        "transcript": format_conversation_transcript(conversation),
        "final_summary": conversation.final_summary,
    }


def build_futureagi_flat_dataset_row(
    conversation: Conversation,
    provider_mode: str,
) -> dict[str, Any]:
    row = build_futureagi_dataset_row(conversation, provider_mode)
    return {
        "conversation_id": row["conversation_id"],
        "topic": row["topic"],
        "scene": row["scene"] or "",
        "constraints": json.dumps(row["constraints"], ensure_ascii=True),
        "status": row["status"],
        "current_round": row["current_round"],
        "exchange_count": row["exchange_count"],
        "evaluation_count": row["evaluation_count"],
        "replacement_count": row["replacement_count"],
        "provider_mode": row["provider_mode"],
        "transcript": row["transcript"],
        "final_summary": row["final_summary"] or "",
    }


@dataclass(slots=True)
class FutureAGIUploadResult:
    status: str
    message: str
    raw_response: dict[str, Any]


class FutureAGIConfigurationError(RuntimeError):
    pass


class FutureAGIUploadError(RuntimeError):
    pass


class FutureAGISDKUnavailableError(RuntimeError):
    pass


class FutureAGIClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def ensure_configured(self) -> None:
        if self.settings.fi_api_key is None or self.settings.fi_secret_key is None:
            raise FutureAGIConfigurationError(
                "FI_API_KEY and FI_SECRET_KEY are required for Future AGI uploads."
            )

    async def upload_conversation_to_dataset(
        self,
        conversation: Conversation,
        dataset_name: str,
    ) -> FutureAGIUploadResult:
        self.ensure_configured()
        return await asyncio.to_thread(
            self._upload_conversation_to_dataset_sync,
            conversation,
            dataset_name,
        )

    def _upload_conversation_to_dataset_sync(
        self,
        conversation: Conversation,
        dataset_name: str,
    ) -> FutureAGIUploadResult:
        try:
            from fi.datasets import Dataset, DatasetConfig
            from fi.datasets.types import (
                Cell,
                Column,
                DataTypeChoices,
                ModelTypes,
                Row,
                SourceChoices,
            )
        except ImportError as exc:
            raise FutureAGISDKUnavailableError(
                "Future AGI SDK is not installed. Install it with `pip install futureagi`."
            ) from exc

        os.environ["FI_API_KEY"] = self.settings.fi_api_key.get_secret_value()
        os.environ["FI_SECRET_KEY"] = self.settings.fi_secret_key.get_secret_value()
        os.environ["FI_BASE_URL"] = self.settings.fi_base_url

        config = DatasetConfig(
            name=dataset_name,
            model_type=ModelTypes.GENERATIVE_LLM,
        )
        dataset = Dataset(dataset_config=config)
        try:
            existing = Dataset.get_dataset_config(dataset_name)
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if "invalid api key" in lowered or "secret key" in lowered or "unauthorized" in lowered:
                raise FutureAGIConfigurationError(
                    "Future AGI rejected FI_API_KEY / FI_SECRET_KEY. "
                    "Check the values in .env, confirm they belong to the selected base URL, "
                    "and verify they have dataset access."
                ) from exc
            if "400 bad request" in lowered or "<html" in lowered:
                raise FutureAGIUploadError(
                    "Future AGI SDK returned a malformed-request error while fetching the dataset config. "
                    f"Base URL in use: {self.settings.fi_base_url}. Raw error: {message}"
                ) from exc
            existing = None

        if existing is None:
            try:
                dataset = dataset.create()
            except Exception as exc:
                message = str(exc)
                lowered = message.lower()
                if "invalid api key" in lowered or "secret key" in lowered or "unauthorized" in lowered:
                    raise FutureAGIConfigurationError(
                        "Future AGI rejected FI_API_KEY / FI_SECRET_KEY while creating the dataset."
                    ) from exc
                raise FutureAGIUploadError(f"Failed to create dataset '{dataset_name}': {message}") from exc
        else:
            dataset = existing

        columns = [
            Column(
                name="conversation_id",
                data_type=DataTypeChoices.TEXT,
                source=SourceChoices.OTHERS,
            ),
            Column(name="topic", data_type=DataTypeChoices.TEXT, source=SourceChoices.OTHERS),
            Column(name="scene", data_type=DataTypeChoices.TEXT, source=SourceChoices.OTHERS),
            Column(
                name="constraints",
                data_type=DataTypeChoices.TEXT,
                source=SourceChoices.OTHERS,
            ),
            Column(name="status", data_type=DataTypeChoices.TEXT, source=SourceChoices.OTHERS),
            Column(
                name="current_round",
                data_type=DataTypeChoices.INTEGER,
                source=SourceChoices.OTHERS,
            ),
            Column(
                name="exchange_count",
                data_type=DataTypeChoices.INTEGER,
                source=SourceChoices.OTHERS,
            ),
            Column(
                name="evaluation_count",
                data_type=DataTypeChoices.INTEGER,
                source=SourceChoices.OTHERS,
            ),
            Column(
                name="replacement_count",
                data_type=DataTypeChoices.INTEGER,
                source=SourceChoices.OTHERS,
            ),
            Column(
                name="provider_mode",
                data_type=DataTypeChoices.TEXT,
                source=SourceChoices.OTHERS,
            ),
            Column(
                name="transcript",
                data_type=DataTypeChoices.TEXT,
                source=SourceChoices.OTHERS,
            ),
            Column(
                name="final_summary",
                data_type=DataTypeChoices.TEXT,
                source=SourceChoices.OTHERS,
            ),
        ]
        try:
            dataset = dataset.add_columns(columns=columns)
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if "already" not in lowered and "exist" not in lowered and "duplicate" not in lowered:
                if "invalid api key" in lowered or "secret key" in lowered or "unauthorized" in lowered:
                    raise FutureAGIConfigurationError(
                        "Future AGI rejected FI_API_KEY / FI_SECRET_KEY while adding columns."
                    ) from exc
                raise FutureAGIUploadError(f"Failed to add columns: {message}") from exc

        row = build_futureagi_flat_dataset_row(conversation, self.settings.provider_mode)
        rows = [
            Row(
                order=1,
                cells=[
                    Cell(column_name="conversation_id", value=row["conversation_id"]),
                    Cell(column_name="topic", value=row["topic"]),
                    Cell(column_name="scene", value=row["scene"]),
                    Cell(column_name="constraints", value=row["constraints"]),
                    Cell(column_name="status", value=row["status"]),
                    Cell(column_name="current_round", value=row["current_round"]),
                    Cell(column_name="exchange_count", value=row["exchange_count"]),
                    Cell(column_name="evaluation_count", value=row["evaluation_count"]),
                    Cell(column_name="replacement_count", value=row["replacement_count"]),
                    Cell(column_name="provider_mode", value=row["provider_mode"]),
                    Cell(column_name="transcript", value=row["transcript"]),
                    Cell(column_name="final_summary", value=row["final_summary"]),
                ],
            )
        ]
        try:
            dataset = dataset.add_rows(rows=rows)
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if "invalid api key" in lowered or "secret key" in lowered or "unauthorized" in lowered:
                raise FutureAGIConfigurationError(
                    "Future AGI rejected FI_API_KEY / FI_SECRET_KEY while adding rows."
                ) from exc
            raise FutureAGIUploadError(f"Failed to add rows: {message}") from exc

        dataset_id = getattr(getattr(dataset, "dataset_config", None), "id", None) or getattr(dataset, "id", None)
        raw_response = {
            "dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "rows_added": 1,
        }
        return FutureAGIUploadResult(
            status="success",
            message=f"Uploaded conversation to Future AGI dataset '{dataset_name}'.",
            raw_response=raw_response,
        )
