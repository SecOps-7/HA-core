"""TTS platform for the Fish Audio integration."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from enum import StrEnum
import logging
import random
import re
from typing import Any, override

from fishaudio import FlushEvent
from fishaudio.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    WebSocketError,
)

from homeassistant.components.tts import (
    TextToSpeechEntity,
    TTSAudioRequest,
    TTSAudioResponse,
    TtsAudioType,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    BACKEND_MODELS,
    CONF_BACKEND,
    CONF_LATENCY,
    CONF_SPEED,
    CONF_VOICE_ID,
    DEFAULT_SPEED,
    DOMAIN,
    MAX_SPEED,
    MIN_SPEED,
    TTS_SUPPORTED_LANGUAGES,
)
from .error import UnexpectedError
from .types import FishAudioConfigEntry

PARALLEL_UPDATES = 1
_LOGGER = logging.getLogger(__name__)

# Fish Audio limits concurrent requests per account (5 on the entry tier),
# shared with its other API calls; an open WebSocket holds a slot.
MAX_CONCURRENT_REQUESTS = 3

# Waits before each REST retry after HTTP 429, plus up to 50% jitter.
RETRY_DELAYS = (0.4, 0.8, 1.6)

# Fish Audio advises sending complete words, 5 to 10 at a time.
WORDS_PER_CHUNK = 5
# Send text without spaces (Chinese, Japanese) once this much is buffered.
MAX_BUFFER_CHARS = 100

# Whitespace, or the point after CJK end punctuation, which has no space.
_BOUNDARY = re.compile(r"\s+|(?<=[。！？])")
# A word ending a sentence, allowing closing quotes or brackets.
_SENTENCE_END = re.compile(r"[.!?。！？][\"'”’)\]]*$")


_UPGRADE_ERRORS = {
    401: "Invalid Fish Audio API key",
    402: "Fish Audio account has insufficient credits",
    403: "Fish Audio API key is not allowed to use this voice or model",
    429: "Rate limited by Fish Audio",
}


def _upgrade_status(err: Exception) -> int | None:
    """Return the HTTP status if Fish Audio refused the WebSocket upgrade."""
    # httpx-ws raises WebSocketUpgradeError with the refused httpx response.
    return getattr(getattr(err, "response", None), "status_code", None)


async def _close_stream(stream: AsyncGenerator[bytes]) -> None:
    """Close the SDK stream in the task that consumed it."""
    # Left to garbage collection, httpx-ws closes the WebSocket from another
    # task and cancels the task that opened it.
    try:
        await stream.aclose()
    except BaseExceptionGroup as group:
        # httpx-ws wraps the GeneratorExit of our own close in a group.
        _, rest = group.split(GeneratorExit)
        if rest is not None:
            raise rest from None


def _next_chunk(buffer: str) -> tuple[str, str, bool] | None:
    """Split the next chunk off the buffer, or return None to wait for more.

    Returns (chunk, rest, ends_sentence). A chunk ends at the first sentence
    end or after WORDS_PER_CHUNK whole words.
    """
    words = 0
    start = 0
    for match in _BOUNDARY.finditer(buffer):
        segment = buffer[start : match.start()]
        start = match.end()
        if not segment.strip():
            continue
        words += 1
        ends_sentence = (
            "\n" in match.group() or _SENTENCE_END.search(segment) is not None
        )
        if ends_sentence or words >= WORDS_PER_CHUNK:
            return buffer[: match.end()], buffer[match.end() :], ends_sentence
    if len(buffer) >= MAX_BUFFER_CHARS:
        return buffer, "", False
    return None


async def _buffered_text(
    source: AsyncIterator[str],
) -> AsyncGenerator[str | FlushEvent]:
    """Regroup streamed text into whole words, flushing at sentence ends."""
    buffer = ""
    async for text in source:
        buffer += text
        while (split := _next_chunk(buffer)) is not None:
            chunk, buffer, ends_sentence = split
            yield chunk
            if ends_sentence:
                yield FlushEvent()
    if buffer.strip():
        yield buffer


async def _end_on_error(
    message_gen: AsyncIterator[str], errors: list[Exception]
) -> AsyncGenerator[str]:
    """Yield the message, ending normally if it fails and recording the error.

    The SDK reads the message in its own task: an error there would never
    reach us, and the SDK would not send "stop", leaving the WebSocket open.
    """
    try:
        async for text in message_gen:
            yield text
    except Exception as err:  # noqa: BLE001
        errors.append(err)


class FishAudioTextToSpeechOption(StrEnum):
    """Store keys for the Fish Audio text-to-speech options."""

    BACKEND = "backend"
    LATENCY = "latency"
    SPEED = "speed"
    VOICE_ID = "voice_id"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: FishAudioConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Fish Audio TTS platform."""
    _LOGGER.debug("Setting up Fish Audio TTS platform")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    _LOGGER.debug("Entry: %s", entry)
    # Iterate over values
    for subentry in entry.subentries.values():
        _LOGGER.debug("Subentry: %s", subentry)
        if subentry.subentry_type != "tts":
            continue
        async_add_entities(
            [FishAudioTTSEntity(entry, subentry, semaphore)],
            config_subentry_id=subentry.subentry_id,
        )


class FishAudioTTSEntity(TextToSpeechEntity):
    """Fish Audio TTS entity."""

    _attr_has_entity_name = True
    _attr_supported_options = [
        FishAudioTextToSpeechOption.VOICE_ID,
        FishAudioTextToSpeechOption.BACKEND,
        FishAudioTextToSpeechOption.LATENCY,
        FishAudioTextToSpeechOption.SPEED,
    ]

    def __init__(
        self,
        entry: FishAudioConfigEntry,
        sub_entry: ConfigSubentry,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Initialize the TTS entity."""
        self._entry = entry
        self._semaphore = semaphore
        self.client = entry.runtime_data
        self.sub_entry = sub_entry
        self._attr_unique_id = sub_entry.subentry_id
        # Configured speed must be a default option so it is part of the TTS cache key.
        self._attr_default_options = {
            FishAudioTextToSpeechOption.SPEED: sub_entry.data.get(
                CONF_SPEED, DEFAULT_SPEED
            )
        }
        title = sub_entry.title
        backend = sub_entry.data[CONF_BACKEND]
        self._attr_name = title

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, sub_entry.subentry_id)},
            manufacturer="Fish Audio",
            model=backend,
            model_id=sub_entry.data.get(CONF_VOICE_ID),
            name=title,
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    @override
    def default_language(self) -> str:
        """Return the default language."""
        return "en"

    @property
    @override
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return TTS_SUPPORTED_LANGUAGES

    def _get_options(self, options: dict[str, Any]) -> tuple[str, str, str, float]:
        voice_id = options.get(
            FishAudioTextToSpeechOption.VOICE_ID, self.sub_entry.data.get(CONF_VOICE_ID)
        )
        backend = options.get(
            FishAudioTextToSpeechOption.BACKEND, self.sub_entry.data.get(CONF_BACKEND)
        )
        latency = options.get(
            FishAudioTextToSpeechOption.LATENCY,
            self.sub_entry.data.get(CONF_LATENCY, "balanced"),
        )
        speed = options.get(
            FishAudioTextToSpeechOption.SPEED,
            self.sub_entry.data.get(CONF_SPEED, DEFAULT_SPEED),
        )

        if voice_id is None:
            raise ServiceValidationError("Voice ID not configured")
        if backend is None:
            raise ServiceValidationError("Backend model not configured")
        # Fish Audio would silently use s2.1-pro for a mistyped model. Stored
        # values are not checked, so existing voices keep working.
        if (
            FishAudioTextToSpeechOption.BACKEND in options
            and backend not in BACKEND_MODELS
        ):
            raise ServiceValidationError(
                f"Unknown Fish Audio model '{backend}'. "
                f"Valid models: {', '.join(BACKEND_MODELS)}"
            )
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise ServiceValidationError(
                f"Speed must be between {MIN_SPEED} and {MAX_SPEED}"
            )

        return voice_id, backend, latency, speed

    @override
    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any],
    ) -> TtsAudioType:
        """Load tts audio file from engine."""

        _LOGGER.debug("Getting TTS audio for %s", message)

        voice_id, backend, latency, speed = self._get_options(options)

        async with self._semaphore:
            audio = await self._convert(message, voice_id, backend, latency, speed)

        return "mp3", audio

    async def _convert(
        self, text: str, voice_id: str, backend: str, latency: str, speed: float
    ) -> bytes:
        """Synthesize over REST, retrying when rate limited."""
        delays = iter(RETRY_DELAYS)
        while True:
            try:
                return await self.client.tts.convert(
                    text=text,
                    reference_id=voice_id,
                    latency=latency,
                    speed=speed,
                    model=backend,
                    format="mp3",
                )
            except AuthenticationError as err:
                self._entry.async_start_reauth(self.hass)
                raise HomeAssistantError(_UPGRADE_ERRORS[401]) from err
            except RateLimitError as err:
                if (delay := next(delays, None)) is None:
                    raise HomeAssistantError(f"Rate limited: {err}") from err
                await asyncio.sleep(delay + random.uniform(0, delay / 2))
            except APIError as err:
                raise HomeAssistantError(f"TTS request failed: {err}") from err
            except Exception as err:
                raise UnexpectedError(err) from err

    def _stream_error(self, err: Exception) -> HomeAssistantError:
        """Translate a streaming error, starting reauth for a rejected key."""
        if isinstance(err, RateLimitError):
            return HomeAssistantError(f"Rate limited: {err}")
        if isinstance(err, APIError):
            return HomeAssistantError(f"TTS request failed: {err}")
        if isinstance(err, WebSocketError):
            return HomeAssistantError(
                f"Websocket connection closed unexpectedly: {err}"
            )
        if (status := _upgrade_status(err)) is None:
            return UnexpectedError(err)
        if status == 401:
            self._entry.async_start_reauth(self.hass)
        return HomeAssistantError(
            _UPGRADE_ERRORS.get(
                status, f"Fish Audio refused the connection (HTTP {status})"
            )
        )

    @override
    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Stream TTS audio from engine."""

        voice_id, backend, latency, speed = self._get_options(request.options)

        return TTSAudioResponse(
            "mp3",
            self._stream_audio(request.message_gen, voice_id, backend, latency, speed),
        )

    async def _stream_audio(
        self,
        message_gen: AsyncIterator[str],
        voice_id: str,
        backend: str,
        latency: str,
        speed: float,
    ) -> AsyncGenerator[bytes]:
        """Stream the message over the WebSocket."""
        async with self._semaphore:
            message_errors: list[Exception] = []
            stream = self.client.tts.stream_websocket(
                _buffered_text(_end_on_error(message_gen, message_errors)),
                reference_id=voice_id,
                latency=latency,
                speed=speed,
                model=backend,
                format="mp3",
            )
            try:
                try:
                    async for chunk in stream:
                        yield chunk
                finally:
                    await _close_stream(stream)
            except Exception as err:
                raise self._stream_error(err) from err
            if message_errors:
                raise message_errors[0]
