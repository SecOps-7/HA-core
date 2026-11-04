"""The Fish Audio integration."""

from functools import partial
import logging

from fishaudio import AsyncFishAudio
from fishaudio.exceptions import AuthenticationError, FishAudioError

from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady

from .types import FishAudioConfigEntry

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.TTS]


async def async_create_client(hass: HomeAssistant, api_key: str) -> AsyncFishAudio:
    """Create a Fish Audio client."""
    # The SDK's httpx client loads the CA bundle on creation, which blocks.
    return await hass.async_add_executor_job(partial(AsyncFishAudio, api_key=api_key))


async def async_setup_entry(hass: HomeAssistant, entry: FishAudioConfigEntry) -> None:
    """Set up Fish Audio from a config entry."""
    client = await async_create_client(hass, entry.data[CONF_API_KEY])

    try:
        # Validate API key by getting account credits.
        await client.account.get_credits()
    except AuthenticationError as exc:
        raise ConfigEntryAuthFailed(f"Invalid API key: {exc}") from exc
    except FishAudioError as exc:
        raise ConfigEntryNotReady(f"Error connecting to Fish Audio: {exc}") from exc

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))


async def _async_update_listener(
    hass: HomeAssistant, entry: FishAudioConfigEntry
) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: FishAudioConfigEntry) -> None:
    """Unload a config entry."""
    await hass.config_entries.async_unload_platforms_or_raise(entry, PLATFORMS)
