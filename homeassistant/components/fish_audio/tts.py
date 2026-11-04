"""TTS platform for the Fish Audio integration."""

from collections.abc import AsyncGenerator
from enum import StrEnum
import logging
from typing import Any, override

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

    _LOGGER.debug("Entry: %s", entry)
    # Iterate over values
    for subentry in entry.subentries.values():
        _LOGGER.debug("Subentry: %s", subentry)
        if subentry.subentry_type != "tts":
            continue
        async_add_entities(
            [FishAudioTTSEntity(entry, subentry)],
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

    def __init__(self, entry: FishAudioConfigEntry, sub_entry: ConfigSubentry) -> None:
        """Initialize the TTS entity."""
        self._entry = entry
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

        try:
            audio = await self.client.tts.convert(
                text=message,
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
            raise HomeAssistantError(f"Rate limited: {err}") from err
        except APIError as err:
            raise HomeAssistantError(f"TTS request failed: {err}") from err
        except Exception as err:
            raise UnexpectedError(err) from err

        return "mp3", audio

    @override
    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Stream TTS audio from engine."""

        voice_id, backend, latency, speed = self._get_options(request.options)

        stream = self.client.tts.stream_websocket(
            request.message_gen,
            reference_id=voice_id,
            latency=latency,
            speed=speed,
            model=backend,
            format="mp3",
        )

        async def gen() -> AsyncGenerator[bytes]:
            try:
                async for chunk in stream:
                    yield chunk

            except RateLimitError as err:
                raise HomeAssistantError(f"Rate limited: {err}") from err
            except APIError as err:
                raise HomeAssistantError(f"TTS request failed: {err}") from err
            except WebSocketError as err:
                raise HomeAssistantError(
                    f"Websocket connection closed unexpectedly: {err}"
                ) from err
            except Exception as err:
                if (status := _upgrade_status(err)) is None:
                    raise UnexpectedError(err) from err
                if status == 401:
                    self._entry.async_start_reauth(self.hass)
                raise HomeAssistantError(
                    _UPGRADE_ERRORS.get(
                        status, f"Fish Audio refused the connection (HTTP {status})"
                    )
                ) from err
            finally:
                await _close_stream(stream)

        return TTSAudioResponse("mp3", gen())
