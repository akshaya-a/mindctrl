import logging

# playwright install --with-deps chromium

from playwright.sync_api import sync_playwright, Browser
from playwright._impl._errors import (
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
)

_logger = logging.getLogger(__name__)


def perform_onboarding_and_get_ll_token(hass_url: str) -> str:
    with sync_playwright() as playwright:
        chromium = playwright.chromium  # or "firefox" or "webkit".
        browser = chromium.launch()
        username, password = perform_hass_onboarding(browser, hass_url)
        return perform_long_lived_token_gen(browser, hass_url, username, password)


def perform_hass_onboarding(browser: Browser, hass_url: str) -> tuple[str, str]:
    username = "pytest"
    password = "pytest"
    page = browser.new_page()
    _logger.info(f"Navigating to {hass_url}")
    try:
        page.goto(hass_url)

        _logger.info("Clicking onboarding button")
        page.get_by_role("button", name="Create my smart home").click()

        # Should be in onboarding form
        _logger.info("Filling out account form")
        page.get_by_label("Name", exact=True).fill("pytest")
        page.get_by_label("Username").fill(username)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_label("Confirm password").fill(password)
        _logger.info("Submitting account form")
        page.get_by_role("button", name="CREATE ACCOUNT").click()

        # Should be map/location
        _logger.info("Skipping location")
        page.get_by_role("button", name="Next").click()

        # Should be in country selector
        _logger.info("Selecting country")
        page.get_by_label("Country").click()
        page.get_by_role("option", name="United States").click()
        page.get_by_role("button", name="Next").click()

        # Should be analytics
        _logger.info("Skipping analytics")
        page.get_by_role("button", name="Next").click()

        # Final page
        _logger.info("Finishing onboarding")
        page.get_by_role("button", name="Finish").click()
        _logger.info("Onboarding complete")

        return username, password
    except PlaywrightTimeoutError as e:
        _logger.error(f"Timeout onboarding: {e}")
        page.screenshot(path="timeout.png")
        raise
    except PlaywrightError as e:
        _logger.error(f"Error onboarding: {e}")
        page.screenshot(path="error.png")
        raise


def perform_long_lived_token_gen(
    browser: Browser, hass_url: str, username: str, password: str
) -> str:
    page = browser.new_page()
    try:
        _logger.info(f"Navigating to {hass_url}/profile/security")
        page.goto(f"{hass_url}/profile/security")

        # Should be on page with login form
        _logger.info("Filling out login form")
        page.get_by_label("Username").fill(username)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_role("button", name="Log in").click()

        # Should be on page with token management
        _logger.info("Creating long-lived token")
        page.get_by_role("button", name="Create token").click()

        _logger.info("Filling out token form")
        page.get_by_label("Name", exact=True).fill("pytest-token")
        page.get_by_role("button", name="OK", exact=True).click()

        _logger.info("Trying to fetch the generated token")
        # mdc-text-field__input
        # <input class="mdc-text-field__input" aria-labelledby="label" type="text" placeholder="" readonly="">
        token = page.get_by_label(
            "Copy your access token. It will not be shown again."
        ).input_value()
        _logger.debug(f"Got test token: {token}")

        return token
    except PlaywrightTimeoutError as e:
        _logger.error(f"Timeout onboarding: {e}")
        page.screenshot(path="timeout.png")
        raise
    except PlaywrightError as e:
        _logger.error(f"Error onboarding: {e}")
        page.screenshot(path="error.png")
        raise
