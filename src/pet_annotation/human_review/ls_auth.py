"""Label Studio authentication helper.

Supports two modes:
- Token auth (legacy API key, disabled by default in LS >= 1.23)
- Session auth (email/password login, works with all LS versions)

Callers should use ``get_ls_session()`` which auto-detects which mode
works and returns an authenticated ``requests.Session``.
"""

from __future__ import annotations

import logging
import re

import requests

logger = logging.getLogger(__name__)


def get_ls_session(
    ls_url: str,
    *,
    api_key: str | None = None,
    email: str | None = None,
    password: str | None = None,
) -> requests.Session:
    """Return an authenticated requests.Session for Label Studio.

    Tries token auth first (if api_key given). If the server rejects it
    (LS >= 1.23 disables legacy tokens), falls back to session login
    with email/password.

    Args:
        ls_url: Label Studio base URL (e.g. http://localhost:8080).
        api_key: API token for token-based auth.
        email: Admin email for session-based auth.
        password: Admin password for session-based auth.

    Returns:
        An authenticated ``requests.Session``.

    Raises:
        RuntimeError: If no auth method succeeds.
    """
    session = requests.Session()

    # Try token auth first
    if api_key:
        session.headers["Authorization"] = f"Token {api_key}"
        try:
            resp = session.get(f"{ls_url}/api/current-user/whoami", timeout=10)
            if resp.status_code == 200:
                logger.info('{"event": "ls_auth", "method": "token"}')
                return session
        except requests.RequestException:
            pass
        # Token failed, clear header and try session login
        session.headers.pop("Authorization", None)
        logger.debug("Token auth failed, falling back to session login")

    # Session login with email/password
    if email and password:
        return _session_login(session, ls_url, email, password)

    raise RuntimeError(
        "Label Studio authentication failed. Provide either a valid API key "
        "or email/password credentials."
    )


def _session_login(
    session: requests.Session,
    ls_url: str,
    email: str,
    password: str,
) -> requests.Session:
    """Log in to Label Studio via the web form and return an authenticated session.

    Args:
        session: Requests session to populate with auth cookies.
        ls_url: Label Studio base URL.
        email: Login email.
        password: Login password.

    Returns:
        The same session, now authenticated with sessionid + csrftoken.

    Raises:
        RuntimeError: If login fails.
    """
    # GET login page to obtain CSRF token
    resp = session.get(f"{ls_url}/user/login", timeout=10)
    resp.raise_for_status()

    csrf_token = session.cookies.get("csrftoken", "")
    if not csrf_token:
        # Try extracting from HTML form
        match = re.search(r'csrfmiddlewaretoken.*?value="([^"]+)"', resp.text)
        if match:
            csrf_token = match.group(1)

    if not csrf_token:
        raise RuntimeError("Could not obtain CSRF token from Label Studio login page")

    # POST login
    resp = session.post(
        f"{ls_url}/user/login",
        data={
            "email": email,
            "password": password,
            "csrfmiddlewaretoken": csrf_token,
        },
        headers={
            "Referer": f"{ls_url}/user/login",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        allow_redirects=False,
        timeout=10,
    )

    if resp.status_code not in (200, 302):
        raise RuntimeError(f"Label Studio login failed with status {resp.status_code}")

    if not session.cookies.get("sessionid"):
        raise RuntimeError("Label Studio login did not return a session cookie")

    # Set CSRF header for all subsequent API calls
    csrf = session.cookies.get("csrftoken", "")
    session.headers.update(
        {
            "X-CSRFToken": csrf,
            "Referer": f"{ls_url}/",
        }
    )

    logger.info('{"event": "ls_auth", "method": "session", "email": "%s"}', email)
    return session
