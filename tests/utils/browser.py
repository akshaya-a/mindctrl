import logging
from pathlib import Path

# playwright install --with-deps chromium

from playwright.sync_api import sync_playwright, Browser
from playwright._impl._errors import (
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
)

_logger = logging.getLogger(__name__)


def perform_onboarding_and_get_ll_token(hass_url: str, screenshot_dir: Path) -> str:
    with sync_playwright() as playwright:
        chromium = playwright.chromium  # or "firefox" or "webkit".
        browser = chromium.launch()
        username, password = perform_hass_onboarding(browser, hass_url, screenshot_dir)
        return perform_long_lived_token_gen(
            browser, hass_url, username, password, screenshot_dir
        )


def perform_hass_onboarding(
    browser: Browser, hass_url: str, screenshot_dir: Path
) -> tuple[str, str]:
    username = "pytest"
    password = "pytest"
    page = browser.new_page()
    _logger.info(f"Navigating to {hass_url}")
    try:
        page.goto(hass_url)
        page.wait_for_load_state()
        page.screenshot(path=screenshot_dir / "playwright-step-01-onboarding.png")

        _logger.info("Clicking onboarding button")
        page.get_by_role("button", name="Create my smart home").click()

        page.wait_for_load_state("domcontentloaded")
        page.screenshot(path=screenshot_dir / "playwright-step-02-account.png")

        # Should be in onboarding form
        _logger.info("Filling out account form")
        page.get_by_label("Name", exact=True).fill("pytest")
        page.get_by_label("Username").fill(username)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_label("Confirm password").fill(password)
        _logger.info("Submitting account form")
        page.get_by_role("button", name="CREATE ACCOUNT").click()

        page.wait_for_load_state("domcontentloaded")
        page.screenshot(path=screenshot_dir / "playwright-step-03-location.png")

        # Should be map/location
        _logger.info("Skipping location")
        page.get_by_role("button", name="Next").click()

        page.wait_for_load_state("domcontentloaded")
        page.screenshot(path=screenshot_dir / "playwright-step-04-country.png")

        # Should be in country selector
        _logger.info("Selecting country")
        page.get_by_label("Country").click()
        page.get_by_role("option", name="United States").click()
        page.get_by_role("button", name="Next").click()

        page.wait_for_load_state("domcontentloaded")
        page.screenshot(path=screenshot_dir / "playwright-step-05-analytics.png")

        # Should be analytics
        _logger.info("Skipping analytics")
        page.get_by_role("button", name="Next").click()

        page.wait_for_load_state("domcontentloaded")
        page.screenshot(path=screenshot_dir / "playwright-step-06-finish.png")

        # Final page
        _logger.info("Finishing onboarding")
        page.get_by_role("button", name="Finish").click()
        _logger.info("Onboarding complete")

        page.wait_for_load_state("domcontentloaded")
        page.screenshot(path=screenshot_dir / "playwright-step-07-done.png")

        return username, password
    except (PlaywrightTimeoutError, PlaywrightError) as e:
        _logger.error(f"Error onboarding: {e}")
        page.screenshot(path=screenshot_dir / "playwright-fail-onboarding.png")
        raise


def perform_long_lived_token_gen(
    browser: Browser, hass_url: str, username: str, password: str, screenshot_dir: Path
) -> str:
    page = browser.new_page()
    try:
        _logger.info(f"Navigating to {hass_url}/profile/security")
        page.goto(f"{hass_url}/profile/security")

        page.wait_for_load_state("domcontentloaded")
        page.screenshot(path=screenshot_dir / "playwright-token-01-login.png")

        # Should be on page with login form
        _logger.info("Filling out login form")
        page.get_by_label("Username").fill(username)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_role("button", name="Log in").click()

        page.wait_for_load_state("domcontentloaded")
        page.screenshot(path=screenshot_dir / "playwright-token-02-security.png")

        # Should be on page with token management
        _logger.info("Creating long-lived token")

        page.get_by_role("button", name="Create token").click()

        # Wait for the \"DOMContentLoaded\" event.
        page.wait_for_load_state("domcontentloaded")

        page.screenshot(path=screenshot_dir / "playwright-token-03-token-form.png")

        _logger.info("Filling out token form")
        page.get_by_label("Name", exact=True).fill("pytest-token")
        page.get_by_role("button", name="OK", exact=True).click()

        _logger.info("Trying to fetch the generated token")

        page.wait_for_load_state("domcontentloaded")
        page.screenshot(path=screenshot_dir / "playwright-token-04-token.png")
        # mdc-text-field__input
        # <input class="mdc-text-field__input" aria-labelledby="label" type="text" placeholder="" readonly="">
        token = page.get_by_label(
            "Copy your access token. It will not be shown again."
        ).input_value()
        _logger.debug(f"Got test token: {token}")

        return token
    except (PlaywrightTimeoutError, PlaywrightError) as e:
        _logger.error(f"Error onboarding: {e}")
        page.screenshot(path=screenshot_dir / "playwright-fail-token.png")
        raise
