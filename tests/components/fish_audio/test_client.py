"""Tests for creating the Fish Audio client."""

import asyncio
from collections.abc import Generator
import ssl
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pytest_homeassistant.common import MockConfigEntry

from homeassistant.components.fish_audio.const import DOMAIN
from homeassistant.config_entries import SOURCE_USER, ConfigEntryState
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from .const import MOCK_CREDITS

_load_verify_locations = ssl.SSLContext.load_verify_locations


@pytest.fixture
def ca_bundle_loads() -> Generator[list[ssl.SSLContext]]:
    """Record CA bundle loads, failing any made in the event loop."""
    loads: list[ssl.SSLContext] = []

    def load_outside_event_loop(
        context: ssl.SSLContext, *args: Any, **kwargs: Any
    ) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loads.append(context)
            return _load_verify_locations(context, *args, **kwargs)
        raise AssertionError("CA bundle loaded in the event loop")

    with patch.object(ssl.SSLContext, "load_verify_locations", load_outside_event_loop):
        yield loads


@pytest.fixture
def mock_get_credits() -> Generator[AsyncMock]:
    """Use the real SDK client with only the account request mocked."""
    with patch(
        "fishaudio.resources.account.AsyncAccountClient.get_credits",
        new=AsyncMock(return_value=MOCK_CREDITS),
    ) as get_credits:
        yield get_credits


async def test_setup_creates_client_outside_event_loop(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_get_credits: AsyncMock,
    ca_bundle_loads: list[ssl.SSLContext],
) -> None:
    """Test setup creates the SDK client without blocking the event loop."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.LOADED
    mock_get_credits.assert_awaited_once()
    assert ca_bundle_loads


@pytest.mark.usefixtures("mock_setup_entry")
async def test_config_flow_creates_client_outside_event_loop(
    hass: HomeAssistant,
    mock_get_credits: AsyncMock,
    ca_bundle_loads: list[ssl.SSLContext],
) -> None:
    """Test the config flow creates the SDK client without blocking the event loop."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input={CONF_API_KEY: "test-api-key"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    mock_get_credits.assert_awaited_once()
    assert ca_bundle_loads
