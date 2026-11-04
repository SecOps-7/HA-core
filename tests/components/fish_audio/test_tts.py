"""Tests for the Fish Audio TTS entity."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fishaudio import AsyncFishAudio, AuthenticationError, FlushEvent, RateLimitError
from fishaudio.exceptions import ServerError, WebSocketError
from httpx_ws import WebSocketUpgradeError
import pytest
from pytest_homeassistant.common import MockConfigEntry, async_mock_service
from pytest_homeassistant.typing import ClientSessionGenerator

from homeassistant.components import tts
from homeassistant.components.fish_audio.const import (
    BACKEND_MODELS,
    CONF_BACKEND,
    CONF_LATENCY,
    CONF_SPEED,
    CONF_VOICE_ID,
    DEFAULT_SPEED,
    DOMAIN,
)
from homeassistant.components.fish_audio.error import UnexpectedError
from homeassistant.components.media_player import (
    ATTR_MEDIA_CONTENT_ID,
    DOMAIN as MP_DOMAIN,
    SERVICE_PLAY_MEDIA,
)
from homeassistant.components.tts import TTSAudioRequest
from homeassistant.config_entries import SOURCE_REAUTH, ConfigSubentryData
from homeassistant.const import ATTR_ENTITY_ID, CONF_API_KEY
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.core_config import async_process_ha_core_config
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import device_registry as dr

from tests.components.tts.common import retrieve_media


@pytest.fixture(autouse=True)
def tts_mutagen_mock_fixture_autouse(tts_mutagen_mock: MagicMock) -> None:
    """Mock writing tags."""


@pytest.fixture(autouse=True)
def mock_tts_cache_dir_autouse(mock_tts_cache_dir: Path) -> None:
    """Mock the TTS cache dir with empty dir."""


@pytest.fixture(autouse=True)
async def setup_internal_url(hass: HomeAssistant) -> None:
    """Set up internal url."""
    await async_process_ha_core_config(
        hass, {"internal_url": "http://example.local:8123"}
    )


@pytest.fixture
async def calls(hass: HomeAssistant) -> list[ServiceCall]:
    """Mock media player calls."""
    return async_mock_service(hass, MP_DOMAIN, SERVICE_PLAY_MEDIA)


async def _failing_stream(error: Exception) -> AsyncGenerator[bytes]:
    if error is not None:
        raise error
    yield b""  # Keep this an async generator


async def test_tts_service_success(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test TTS service with successful audio generation."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Get the TTS entity
    entity = hass.data[tts.DOMAIN].get_entity("tts.test_voice_test_voice")
    assert entity is not None

    # Test the TTS audio generation
    extension, data = await entity.async_get_tts_audio(
        message="Hello world",
        language="en",
        options={},
    )

    # Verify the result
    assert extension == "mp3"
    assert data == b"fake_audio_data"

    # Verify the client was called
    mock_fishaudio_client.tts.convert.assert_called_once()


async def test_tts_stream_success(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test TTS stream with successful audio generation."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Get the TTS entity
    entity = hass.data[tts.DOMAIN].get_entity("tts.test_voice_test_voice")
    assert entity is not None

    # Test the TTS stream generation
    async def generate_message() -> AsyncGenerator[str]:
        yield "Hello, "
        yield "World!"

    message_gen = generate_message()

    request = TTSAudioRequest(language="en", options={}, message_gen=message_gen)

    response = await entity.async_stream_tts_audio(request)

    assert response.extension == "mp3"
    assert [chunk async for chunk in response.data_gen] == [
        b"first audio chunk",
        b"second audio chunk",
    ]

    sent = [
        item
        async for item in mock_fishaudio_client.tts.stream_websocket.call_args.args[0]
    ]
    assert sent == ["Hello, World!"]
    mock_fishaudio_client.tts.convert.assert_not_called()


async def test_tts_rate_limited(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test TTS service with rate limit error."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Get the TTS entity
    entity = hass.data[tts.DOMAIN].get_entity("tts.test_voice_test_voice")
    assert entity is not None

    # Make tts.convert() fail with rate limit error
    mock_fishaudio_client.tts.convert = AsyncMock(
        side_effect=RateLimitError(429, "Rate limited")
    )

    # Test that the error is raised
    with pytest.raises(HomeAssistantError, match="Rate limited"):
        await entity.async_get_tts_audio(
            message="Hello world",
            language="en",
            options={},
        )

    # Verify the client was called
    mock_fishaudio_client.tts.convert.assert_called_once()


async def test_tts_missing_voice_id(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test TTS service raises ServiceValidationError when voice_id is missing."""
    # Create a config entry with no voice_id
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"api_key": "test-key"},
        unique_id="test_user",
        subentries_data=[
            ConfigSubentryData(
                data={CONF_BACKEND: "s1"},  # Missing CONF_VOICE_ID
                subentry_type="tts",
                title="Test Voice",
                subentry_id="test-sub-id",
                unique_id="test-voice",
            )
        ],
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Get the TTS entity
    entity = hass.data[tts.DOMAIN].get_entity("tts.test_voice_test_voice")
    assert entity is not None

    # Test that the error is raised
    with pytest.raises(ServiceValidationError, match="Voice ID not configured"):
        await entity.async_get_tts_audio(
            message="Hello world",
            language="en",
            options={},
        )


async def test_tts_supported_languages(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test TTS entity supported languages."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Get the TTS entity
    entity = hass.data[tts.DOMAIN].get_entity("tts.test_voice_test_voice")
    assert entity is not None

    # Verify supported languages
    assert entity.supported_languages == [
        "Any",
        "en",
        "zh",
        "de",
        "ja",
        "ar",
        "fr",
        "es",
        "ko",
    ]


@pytest.mark.parametrize(
    ("options", "expected_speed"),
    [
        pytest.param({}, DEFAULT_SPEED, id="default"),
        pytest.param({CONF_SPEED: 1.5}, 1.5, id="per_call_faster"),
        pytest.param({CONF_SPEED: 0.75}, 0.75, id="per_call_slower"),
    ],
)
async def test_tts_speed_option(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    options: dict[str, float],
    expected_speed: float,
) -> None:
    """Test the speech speed per-call option is forwarded to the client."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    entity = hass.data[tts.DOMAIN].get_entity("tts.test_voice_test_voice")
    assert entity is not None

    await entity.async_get_tts_audio(
        message="Hello world",
        language="en",
        options=options,
    )

    assert mock_fishaudio_client.tts.convert.call_args.kwargs["speed"] == expected_speed


async def test_tts_speed_from_subentry(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test the configured per-voice speed is used without a per-call option.

    This is the Assist pipeline path, which does not pass per-call options.
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "test-api-key"},
        unique_id="test_user",
        subentries_data=[
            ConfigSubentryData(
                data={
                    CONF_VOICE_ID: "voice-123",
                    CONF_BACKEND: "s1",
                    CONF_LATENCY: "balanced",
                    CONF_SPEED: 1.25,
                },
                subentry_type="tts",
                title="Test Voice",
                subentry_id="test-subentry-id",
                unique_id="voice-123-s1",
            )
        ],
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    entity = hass.data[tts.DOMAIN].get_entity("tts.test_voice_test_voice")
    assert entity is not None

    # The configured speed is exposed as a default option so it is part of the cache key.
    assert entity.default_options == {CONF_SPEED: 1.25}

    await entity.async_get_tts_audio(
        message="Hello world",
        language="en",
        options={},
    )

    assert mock_fishaudio_client.tts.convert.call_args.kwargs["speed"] == 1.25


@pytest.mark.parametrize("speed", [0.1, 5.0])
async def test_tts_speed_out_of_range(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    speed: float,
) -> None:
    """Test an out-of-range per-call speed raises a validation error."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    entity = hass.data[tts.DOMAIN].get_entity("tts.test_voice_test_voice")
    assert entity is not None

    with pytest.raises(ServiceValidationError):
        await entity.async_get_tts_audio(
            message="Hello world",
            language="en",
            options={CONF_SPEED: speed},
        )

    mock_fishaudio_client.tts.convert.assert_not_called()


# Service-level integration tests


async def test_tts_service_speak(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    hass_client: ClientSessionGenerator,
    calls: list[ServiceCall],
) -> None:
    """Test TTS speak service call."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    await hass.services.async_call(
        tts.DOMAIN,
        "speak",
        {
            ATTR_ENTITY_ID: "tts.test_voice_test_voice",
            tts.ATTR_MEDIA_PLAYER_ENTITY_ID: "media_player.something",
            tts.ATTR_MESSAGE: "Hello world",
        },
        blocking=True,
    )

    assert len(calls) == 1
    assert (
        await retrieve_media(hass, hass_client, calls[0].data[ATTR_MEDIA_CONTENT_ID])
        == HTTPStatus.OK
    )


async def test_tts_service_speak_with_language(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    hass_client: ClientSessionGenerator,
    calls: list[ServiceCall],
) -> None:
    """Test TTS speak service call with language parameter."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    await hass.services.async_call(
        tts.DOMAIN,
        "speak",
        {
            ATTR_ENTITY_ID: "tts.test_voice_test_voice",
            tts.ATTR_MEDIA_PLAYER_ENTITY_ID: "media_player.something",
            tts.ATTR_MESSAGE: "Hola mundo",
            tts.ATTR_LANGUAGE: "es",
        },
        blocking=True,
    )

    assert len(calls) == 1
    assert (
        await retrieve_media(hass, hass_client, calls[0].data[ATTR_MEDIA_CONTENT_ID])
        == HTTPStatus.OK
    )


async def test_tts_service_speak_server_error(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    hass_client: ClientSessionGenerator,
    calls: list[ServiceCall],
) -> None:
    """Test TTS speak service call with server error."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Get the client from runtime_data and make it fail with ServerError
    entry = hass.config_entries.async_get_entry(mock_config_entry.entry_id)
    assert entry is not None
    mock_fishaudio_client.tts.stream_websocket.side_effect = lambda *_args, **_kwargs: (
        _failing_stream(ServerError(500, "Internal server error"))
    )

    await hass.services.async_call(
        tts.DOMAIN,
        "speak",
        {
            ATTR_ENTITY_ID: "tts.test_voice_test_voice",
            tts.ATTR_MEDIA_PLAYER_ENTITY_ID: "media_player.something",
            tts.ATTR_MESSAGE: "Test server error",
        },
        blocking=True,
    )

    assert len(calls) == 1
    assert (
        await retrieve_media(hass, hass_client, calls[0].data[ATTR_MEDIA_CONTENT_ID])
        == HTTPStatus.INTERNAL_SERVER_ERROR
    )


async def test_tts_service_speak_rate_limit_error(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    hass_client: ClientSessionGenerator,
    calls: list[ServiceCall],
) -> None:
    """Test TTS speak service call with rate limit error."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Get the client from runtime_data and make it fail with RateLimitError
    entry = hass.config_entries.async_get_entry(mock_config_entry.entry_id)
    assert entry is not None
    mock_fishaudio_client.tts.stream_websocket.side_effect = lambda *_args, **_kwargs: (
        _failing_stream(RateLimitError(429, "Rate limit exceeded"))
    )

    await hass.services.async_call(
        tts.DOMAIN,
        "speak",
        {
            ATTR_ENTITY_ID: "tts.test_voice_test_voice",
            tts.ATTR_MEDIA_PLAYER_ENTITY_ID: "media_player.something",
            tts.ATTR_MESSAGE: "Test rate limit error",
        },
        blocking=True,
    )

    assert len(calls) == 1
    assert (
        await retrieve_media(hass, hass_client, calls[0].data[ATTR_MEDIA_CONTENT_ID])
        == HTTPStatus.INTERNAL_SERVER_ERROR
    )


@pytest.fixture
async def tts_entity(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
) -> tts.TextToSpeechEntity:
    """Set up the integration and return the first TTS entity."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    entity = hass.data[tts.DOMAIN].get_entity("tts.test_voice_test_voice")
    assert entity is not None
    return entity


async def _abandon_stream(entity: tts.TextToSpeechEntity) -> tts.TTSAudioResponse:
    """Read the first audio chunk, then stop reading the stream."""

    async def message() -> AsyncGenerator[str]:
        yield "Hello"

    response = await entity.async_stream_tts_audio(
        TTSAudioRequest(language="en", options={}, message_gen=message())
    )
    assert await anext(response.data_gen) == b"chunk"
    return response


async def test_tts_stream_closed_when_abandoned(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test the SDK stream is closed in the reading task when reading stops."""
    closed_in: list[asyncio.Task | None] = []

    async def stream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[bytes]:
        try:
            yield b"chunk"
            yield b"chunk"
        finally:
            closed_in.append(asyncio.current_task())

    mock_fishaudio_client.tts.stream_websocket.side_effect = stream

    response = await _abandon_stream(tts_entity)
    await response.data_gen.aclose()

    assert closed_in == [asyncio.current_task()]


async def test_tts_stream_close_ignores_httpx_ws_echo(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test the GeneratorExit group httpx-ws raises on close is not propagated."""

    async def stream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[bytes]:
        try:
            yield b"chunk"
        except GeneratorExit as err:
            raise BaseExceptionGroup("unhandled errors in a TaskGroup", [err]) from None

    mock_fishaudio_client.tts.stream_websocket.side_effect = stream

    response = await _abandon_stream(tts_entity)
    await response.data_gen.aclose()


async def test_tts_stream_close_raises_other_errors(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test an error raised while closing the SDK stream is not swallowed."""

    async def stream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[bytes]:
        try:
            yield b"chunk"
        except GeneratorExit as err:
            raise BaseExceptionGroup(
                "unhandled errors in a TaskGroup", [err, OSError("socket broke")]
            ) from None

    mock_fishaudio_client.tts.stream_websocket.side_effect = stream

    response = await _abandon_stream(tts_entity)
    with pytest.raises(UnexpectedError) as exc_info:
        await response.data_gen.aclose()
    cause = exc_info.value.__cause__
    assert isinstance(cause, ExceptionGroup)
    assert [type(err) for err in cause.exceptions] == [OSError]


async def _stream_refused(entity: tts.TextToSpeechEntity) -> None:
    """Read a stream whose WebSocket upgrade is refused."""

    async def message() -> AsyncGenerator[str]:
        yield "Hello"

    response = await entity.async_stream_tts_audio(
        TTSAudioRequest(language="en", options={}, message_gen=message())
    )
    [chunk async for chunk in response.data_gen]


@pytest.mark.parametrize(
    ("status", "message", "reauth_flows"),
    [
        pytest.param(401, "Invalid Fish Audio API key", 1, id="invalid-key"),
        pytest.param(402, "insufficient credits", 0, id="no-credits"),
        pytest.param(403, "not allowed to use this voice", 0, id="forbidden"),
        pytest.param(429, "Rate limited by Fish Audio", 0, id="rate-limited"),
        pytest.param(500, r"refused the connection \(HTTP 500\)", 0, id="other"),
    ],
)
async def test_tts_stream_upgrade_refused(
    hass: HomeAssistant,
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    status: int,
    message: str,
    reauth_flows: int,
) -> None:
    """Test a refused WebSocket upgrade gives a clear error, and reauth on 401."""
    error = WebSocketUpgradeError(SimpleNamespace(status_code=status))
    mock_fishaudio_client.tts.stream_websocket.side_effect = lambda *_args, **_kwargs: (
        _failing_stream(error)
    )

    with pytest.raises(HomeAssistantError, match=message) as exc_info:
        await _stream_refused(tts_entity)
    assert not isinstance(exc_info.value, UnexpectedError)
    await hass.async_block_till_done()

    flows = mock_config_entry.async_get_active_flows(hass, {SOURCE_REAUTH})
    assert len(list(flows)) == reauth_flows


async def test_tts_rest_invalid_key_starts_reauth(
    hass: HomeAssistant,
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test a rejected API key over REST starts reauthentication."""
    mock_fishaudio_client.tts.convert.side_effect = AuthenticationError(
        401, "Invalid API key"
    )

    with pytest.raises(HomeAssistantError, match="Invalid Fish Audio API key"):
        await tts_entity.async_get_tts_audio(message="Hello", language="en", options={})
    await hass.async_block_till_done()

    flows = mock_config_entry.async_get_active_flows(hass, {SOURCE_REAUTH})
    assert len(list(flows)) == 1


async def test_tts_stream_unknown_error(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test an unknown error after audio started is reported as unexpected."""

    async def stream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[bytes]:
        yield b"chunk"
        raise RuntimeError("boom")

    mock_fishaudio_client.tts.stream_websocket.side_effect = stream

    with pytest.raises(UnexpectedError):
        await _stream_refused(tts_entity)
    mock_fishaudio_client.tts.convert.assert_not_called()


# The model header values in Fish Audio's API reference, written out so a
# typo in BACKEND_MODELS fails here.
FISH_API_MODELS = ["s2.1-pro", "s2.1-pro-free", "s2-pro", "drama-3", "s1"]


def test_backend_models_match_fish_api() -> None:
    """Test the offered models are exactly the ones Fish Audio accepts."""
    assert BACKEND_MODELS == FISH_API_MODELS


@pytest.mark.parametrize("model", FISH_API_MODELS)
async def test_model_header_rest(
    hass: HomeAssistant,
    tts_entity: tts.TextToSpeechEntity,
    model: str,
) -> None:
    """Test the exact model string reaches the SDK's REST request header."""
    # The real SDK client: the integration's AsyncFishAudio is mocked here.
    tts_entity.client = AsyncFishAudio(api_key="test-api-key")

    with (
        patch(
            "fishaudio.core.client_wrapper.AsyncClientWrapper.request",
            side_effect=RuntimeError("request recorded"),
        ) as mock_request,
        pytest.raises(UnexpectedError),
    ):
        await tts_entity.async_get_tts_audio(
            message="Hello", language="en", options={"backend": model}
        )

    assert mock_request.call_args.args[1] == "/v1/tts"
    assert mock_request.call_args.kwargs["headers"]["model"] == model


@pytest.mark.parametrize("model", FISH_API_MODELS)
async def test_model_header_stream(
    hass: HomeAssistant,
    tts_entity: tts.TextToSpeechEntity,
    model: str,
) -> None:
    """Test the exact model string reaches the WebSocket handshake header."""
    # The real SDK client: the integration's AsyncFishAudio is mocked here.
    tts_entity.client = AsyncFishAudio(api_key="test-api-key")

    async def message() -> AsyncGenerator[str]:
        yield "Hello"

    with patch(
        "fishaudio.resources.tts.aconnect_ws",
        side_effect=RuntimeError("handshake recorded"),
    ) as mock_connect:
        response = await tts_entity.async_stream_tts_audio(
            TTSAudioRequest(
                language="en", options={"backend": model}, message_gen=message()
            )
        )
        with pytest.raises(UnexpectedError):
            [chunk async for chunk in response.data_gen]

    assert mock_connect.call_args.args[0] == "/v1/tts/live"
    assert mock_connect.call_args.kwargs["headers"]["model"] == model


@pytest.mark.parametrize("backend", ["s2.1pro", "S2.1-Pro", "drama3", "speech-1.6"])
async def test_tts_unknown_model_override_rejected(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
    backend: str,
) -> None:
    """Test a per-call model Fish Audio doesn't accept is rejected, not sent."""

    async def message() -> AsyncGenerator[str]:
        yield "Hello"

    with pytest.raises(ServiceValidationError, match="Unknown Fish Audio model"):
        await tts_entity.async_stream_tts_audio(
            TTSAudioRequest(
                language="en", options={"backend": backend}, message_gen=message()
            )
        )
    with pytest.raises(ServiceValidationError, match="Unknown Fish Audio model"):
        await tts_entity.async_get_tts_audio(
            message="Hello", language="en", options={"backend": backend}
        )

    mock_fishaudio_client.tts.stream_websocket.assert_not_called()
    mock_fishaudio_client.tts.convert.assert_not_called()


async def test_tts_stored_retired_model_still_used(
    hass: HomeAssistant,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test a voice saved with a model no longer offered keeps working."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_KEY: "test-key"},
        unique_id="test_user",
        subentries_data=[
            ConfigSubentryData(
                data={CONF_VOICE_ID: "voice-123", CONF_BACKEND: "speech-1.6"},
                subentry_type="tts",
                title="Test Voice",
                subentry_id="test-sub-id",
                unique_id="voice-123-speech-1.6",
            )
        ],
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    entity = hass.data[tts.DOMAIN].get_entity("tts.test_voice_test_voice")
    assert entity is not None

    await entity.async_get_tts_audio(message="Hello", language="en", options={})

    assert mock_fishaudio_client.tts.convert.call_args.kwargs["model"] == "speech-1.6"


@pytest.mark.usefixtures("tts_entity")
async def test_tts_device_shows_voice_id(
    mock_config_entry: MockConfigEntry,
    device_registry: dr.DeviceRegistry,
) -> None:
    """Test each voice's device shows its Fish Audio voice ID as the model ID."""
    device = device_registry.async_get_device_by_identifier(
        (DOMAIN, "test-subentry-id"), mock_config_entry.entry_id
    )
    assert device is not None
    assert device.model_id == "voice-123"


FLUSH = FlushEvent()


async def _message(*parts: str) -> AsyncGenerator[str]:
    """Yield message parts like Home Assistant streams text to the entity."""
    for part in parts:
        yield part


def _capture_stream(sent: list[str | FlushEvent]) -> Callable[..., Any]:
    """Return a stream_websocket replacement recording what is sent to Fish."""

    async def stream(
        text_stream: AsyncIterable[str | FlushEvent], **_kwargs: Any
    ) -> AsyncGenerator[bytes]:
        sent.extend([item async for item in text_stream])
        yield b"chunk"

    return stream


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        pytest.param(["Hel", "lo ", "wor", "ld"], ["Hello world"], id="joins-words"),
        pytest.param(
            ["one two ", "three fo", "ur five six se", "ven"],
            ["one two three four five ", "six seven"],
            id="five-words",
        ),
        pytest.param(
            ["Hi there. How ", "are you? Fi", "ne"],
            ["Hi there. ", FLUSH, "How are you? ", FLUSH, "Fine"],
            id="flush-each-sentence",
        ),
        pytest.param(
            ["Line one\nLine ", "two"], ["Line one\n", FLUSH, "Line two"], id="newline"
        ),
        pytest.param(
            ['He said "go." ', "Then"],
            ['He said "go." ', FLUSH, "Then"],
            id="closing-quote",
        ),
        pytest.param(["Pi is 3.", "14 today"], ["Pi is 3.14 today"], id="decimal"),
        pytest.param(
            ["你好。今天", "怎么样？"],
            ["你好。", FLUSH, "今天怎么样？", FLUSH],
            id="cjk-punctuation",
        ),
        pytest.param(["x" * 60, "x" * 60], ["x" * 120], id="no-spaces-capped"),
        pytest.param(
            ["  Hello world. ", "Bye"],
            ["  Hello world. ", FLUSH, "Bye"],
            id="leading-whitespace",
        ),
        pytest.param([], [], id="empty"),
    ],
)
async def test_tts_stream_text_buffering(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
    parts: list[str],
    expected: list[str | FlushEvent],
) -> None:
    """Test text goes out as whole words, with a flush after each sentence."""
    sent: list[str | FlushEvent] = []
    mock_fishaudio_client.tts.stream_websocket.side_effect = _capture_stream(sent)

    response = await tts_entity.async_stream_tts_audio(
        TTSAudioRequest(language="en", options={}, message_gen=_message(*parts))
    )
    [chunk async for chunk in response.data_gen]

    assert sent == expected


async def test_tts_stream_sends_text_before_message_ends(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test a complete sentence is sent while the message is still being written."""
    more_text = asyncio.Event()
    first_sent: asyncio.Future[str | FlushEvent] = (
        asyncio.get_running_loop().create_future()
    )

    async def slow_message() -> AsyncGenerator[str]:
        yield "The first sentence is here. "
        await more_text.wait()
        yield "The rest."

    async def stream(
        text_stream: AsyncIterable[str | FlushEvent], **_kwargs: Any
    ) -> AsyncGenerator[bytes]:
        async for item in text_stream:
            if not first_sent.done():
                first_sent.set_result(item)
        yield b"chunk"

    mock_fishaudio_client.tts.stream_websocket.side_effect = stream
    response = await tts_entity.async_stream_tts_audio(
        TTSAudioRequest(language="en", options={}, message_gen=slow_message())
    )
    reader = asyncio.create_task(anext(response.data_gen))

    assert await asyncio.wait_for(first_sent, 1) == "The first sentence is here. "
    assert not reader.done()
    more_text.set()
    assert await reader == b"chunk"


async def test_tts_stream_no_fallback_after_audio(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test a disconnect after audio started is raised with a clear message."""

    async def stream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[bytes]:
        yield b"partial"
        raise WebSocketError("WebSocket disconnected")

    mock_fishaudio_client.tts.stream_websocket.side_effect = stream
    response = await tts_entity.async_stream_tts_audio(
        TTSAudioRequest(language="en", options={}, message_gen=_message("Hello"))
    )

    assert await anext(response.data_gen) == b"partial"
    with pytest.raises(HomeAssistantError, match="Websocket connection closed"):
        await anext(response.data_gen)
    mock_fishaudio_client.tts.convert.assert_not_called()


async def test_tts_stream_message_error(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test a failing message still lets Fish finish, then raises the error."""
    sent: list[str | FlushEvent] = []
    mock_fishaudio_client.tts.stream_websocket.side_effect = _capture_stream(sent)

    async def failing_message() -> AsyncGenerator[str]:
        yield "Hello there. "
        raise HomeAssistantError("Conversation agent failed")

    response = await tts_entity.async_stream_tts_audio(
        TTSAudioRequest(language="en", options={}, message_gen=failing_message())
    )

    assert await anext(response.data_gen) == b"chunk"
    with pytest.raises(HomeAssistantError, match="Conversation agent failed"):
        await anext(response.data_gen)
    assert sent == ["Hello there. ", FLUSH]


async def test_tts_concurrent_requests_capped(
    hass: HomeAssistant,
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test at most three requests per account run at once, across its voices."""
    release = asyncio.Event()
    open_streams = 0
    most_open = 0

    async def stream(
        text_stream: AsyncIterable[str | FlushEvent], **_kwargs: Any
    ) -> AsyncGenerator[bytes]:
        nonlocal open_streams, most_open
        open_streams += 1
        most_open = max(most_open, open_streams)
        try:
            async for _ in text_stream:
                pass
            await release.wait()
            yield b"chunk"
        finally:
            open_streams -= 1

    mock_fishaudio_client.tts.stream_websocket.side_effect = stream
    second_voice = hass.data[tts.DOMAIN].get_entity("tts.second_voice_second_voice")
    assert second_voice is not None

    async def speak(entity: tts.TextToSpeechEntity) -> list[bytes]:
        response = await entity.async_stream_tts_audio(
            TTSAudioRequest(language="en", options={}, message_gen=_message("Hi"))
        )
        return [chunk async for chunk in response.data_gen]

    streams = [
        asyncio.create_task(speak(entity)) for entity in (tts_entity, second_voice) * 3
    ]
    await hass.async_block_till_done()
    assert mock_fishaudio_client.tts.stream_websocket.call_count == 3

    rest = asyncio.create_task(
        tts_entity.async_get_tts_audio(message="Hi", language="en", options={})
    )
    await hass.async_block_till_done()
    mock_fishaudio_client.tts.convert.assert_not_called()

    release.set()
    results = await asyncio.wait_for(asyncio.gather(*streams, rest), 5)

    assert results == [[b"chunk"]] * 6 + [("mp3", b"fake_audio_data")]
    assert most_open == 3


async def test_tts_slot_released_when_stream_abandoned(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test stopping a stream early frees its slot."""

    async def stream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[bytes]:
        yield b"chunk"
        yield b"chunk"

    mock_fishaudio_client.tts.stream_websocket.side_effect = stream
    for _ in range(4):
        response = await _abandon_stream(tts_entity)
        await response.data_gen.aclose()

    assert await asyncio.wait_for(
        tts_entity.async_get_tts_audio(message="Hi", language="en", options={}), 1
    ) == ("mp3", b"fake_audio_data")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        pytest.param(
            RateLimitError(429, "Too many"), "Rate limited", id="rate-limited"
        ),
        pytest.param(ServerError(500, "Server error"), "TTS request failed", id="api"),
    ],
)
async def test_tts_stream_api_error_after_audio(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
    error: Exception,
    message: str,
) -> None:
    """Test an API error after audio started is raised with a clear message."""

    async def stream(*_args: Any, **_kwargs: Any) -> AsyncGenerator[bytes]:
        yield b"partial"
        raise error

    mock_fishaudio_client.tts.stream_websocket.side_effect = stream
    response = await tts_entity.async_stream_tts_audio(
        TTSAudioRequest(language="en", options={}, message_gen=_message("Hello"))
    )

    assert await anext(response.data_gen) == b"partial"
    with pytest.raises(HomeAssistantError, match=message) as exc_info:
        await anext(response.data_gen)
    assert not isinstance(exc_info.value, UnexpectedError)


async def test_tts_stream_error_before_audio(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test an error before any audio is reported, without a REST request."""
    mock_fishaudio_client.tts.stream_websocket.side_effect = lambda *_args, **_kwargs: (
        _failing_stream(RuntimeError("boom"))
    )

    with pytest.raises(UnexpectedError):
        await _stream_refused(tts_entity)
    mock_fishaudio_client.tts.convert.assert_not_called()


async def test_tts_rest_api_error(
    tts_entity: tts.TextToSpeechEntity,
    mock_fishaudio_client: AsyncMock,
) -> None:
    """Test an API error over REST is reported as a failed request."""
    mock_fishaudio_client.tts.convert.side_effect = ServerError(500, "Server error")

    with pytest.raises(HomeAssistantError, match="TTS request failed"):
        await tts_entity.async_get_tts_audio(message="Hello", language="en", options={})
