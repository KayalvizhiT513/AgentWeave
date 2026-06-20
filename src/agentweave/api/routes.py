from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from agentweave.api.schemas import (
    ConversationDetailResponse,
    ConversationEventResponse,
    ConversationSummaryResponse,
    CreateConversationRequest,
    FutureAGIDatasetRowResponse,
    FutureAGIUploadRequest,
    FutureAGIUploadResponse,
)
from agentweave.dependencies import get_futureagi_client, get_orchestrator
from agentweave.core.models import Conversation
from agentweave.services.futureagi import (
    FutureAGIConfigurationError,
    FutureAGISDKUnavailableError,
    FutureAGIUploadError,
    build_futureagi_dataset_row,
)


router = APIRouter(prefix="/api/v1", tags=["conversations"])


def _summary(conversation: Conversation) -> ConversationSummaryResponse:
    return ConversationSummaryResponse(
        id=conversation.id,
        topic=conversation.topic,
        scene=conversation.scene,
        status=conversation.status,
        current_round=conversation.current_round,
        active_agent_count=len(conversation.active_agents()),
        exchange_count=len(conversation.exchanges),
        evaluation_count=len(conversation.evaluations),
        replacement_count=len(conversation.replacements),
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _detail(conversation: Conversation) -> ConversationDetailResponse:
    return ConversationDetailResponse.model_validate(conversation.model_dump())


@router.post(
    "/conversations",
    response_model=ConversationDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    request: CreateConversationRequest,
    orchestrator=Depends(get_orchestrator),
) -> ConversationDetailResponse:
    conversation = await orchestrator.create_conversation(
        topic=request.topic,
        scene=request.scene,
        constraints=request.constraints,
        runtime=request.runtime,
    )
    if request.auto_start:
        conversation = await orchestrator.start_conversation(conversation.id)
    return _detail(conversation)


@router.get("/conversations", response_model=list[ConversationSummaryResponse])
async def list_conversations(orchestrator=Depends(get_orchestrator)) -> list[ConversationSummaryResponse]:
    conversations = await orchestrator.list_conversations()
    return [_summary(item) for item in conversations]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    orchestrator=Depends(get_orchestrator),
) -> ConversationDetailResponse:
    conversation = await orchestrator.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _detail(conversation)


@router.post("/conversations/{conversation_id}/start", response_model=ConversationDetailResponse)
async def start_conversation(
    conversation_id: str,
    orchestrator=Depends(get_orchestrator),
) -> ConversationDetailResponse:
    try:
        conversation = await orchestrator.start_conversation(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return _detail(conversation)


@router.post("/conversations/{conversation_id}/step", response_model=ConversationDetailResponse)
async def step_conversation(
    conversation_id: str,
    orchestrator=Depends(get_orchestrator),
) -> ConversationDetailResponse:
    try:
        conversation = await orchestrator.step_conversation(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return _detail(conversation)


@router.post("/conversations/{conversation_id}/stop", response_model=ConversationDetailResponse)
async def stop_conversation(
    conversation_id: str,
    orchestrator=Depends(get_orchestrator),
) -> ConversationDetailResponse:
    try:
        conversation = await orchestrator.stop_conversation(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return _detail(conversation)


@router.get("/conversations/{conversation_id}/events", response_model=None)
async def stream_events(
    conversation_id: str,
    orchestrator=Depends(get_orchestrator),
):
    conversation = await orchestrator.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    queue = orchestrator.event_bus.subscribe(conversation_id)

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                payload = ConversationEventResponse.model_validate(event.model_dump())
                yield f"data: {json.dumps(payload.model_dump(mode='json'))}\n\n"
        except asyncio.CancelledError:
            raise
        finally:
            orchestrator.event_bus.unsubscribe(conversation_id, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get(
    "/conversations/{conversation_id}/futureagi/row",
    response_model=FutureAGIDatasetRowResponse,
)
async def get_futureagi_row(
    conversation_id: str,
    orchestrator=Depends(get_orchestrator),
) -> FutureAGIDatasetRowResponse:
    conversation = await orchestrator.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    row = build_futureagi_dataset_row(conversation, orchestrator.provider.settings.provider_mode if hasattr(orchestrator.provider, "settings") else "simulated")
    return FutureAGIDatasetRowResponse.model_validate(row)


@router.post(
    "/conversations/{conversation_id}/futureagi/upload",
    response_model=FutureAGIUploadResponse,
)
async def upload_conversation_to_futureagi(
    conversation_id: str,
    request: FutureAGIUploadRequest,
    orchestrator=Depends(get_orchestrator),
    futureagi_client=Depends(get_futureagi_client),
) -> FutureAGIUploadResponse:
    conversation = await orchestrator.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        result = await futureagi_client.upload_conversation_to_dataset(
            conversation=conversation,
            dataset_name=request.dataset_name,
        )
    except FutureAGIConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FutureAGISDKUnavailableError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except FutureAGIUploadError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return FutureAGIUploadResponse(
        dataset_name=request.dataset_name,
        status=result.status,
        message=result.message,
    )
