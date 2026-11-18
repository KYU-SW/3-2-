"""Firebase Authentication REST API와 Streamlit 로그인 상태를 관리합니다."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Final

import requests
import streamlit as st

AUTH_SIGN_UP_URL: Final[str] = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
AUTH_SIGN_IN_URL: Final[str] = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
TOKEN_REFRESH_URL: Final[str] = "https://securetoken.googleapis.com/v1/token"
TOKEN_REFRESH_MARGIN_SECONDS: Final[int] = 60
REQUEST_TIMEOUT_SECONDS: Final[int] = 15


class AuthError(Exception):
    """사용자에게 한국어 안내를 제공할 수 있는 인증 오류입니다."""


@dataclass(frozen=True)
class AuthSession:
    """Firebase 인증 응답에서 앱이 보관할 최소 세션 정보입니다."""

    uid: str
    email: str
    id_token: str
    refresh_token: str
    expires_at: float


def initialize_auth_state() -> None:
    """로그인 세션에 필요한 Streamlit 상태 키를 안전한 기본값으로 초기화합니다."""
    defaults: dict[str, Any] = {
        "uid": None,
        "email": None,
        "idToken": None,
        "refreshToken": None,
        "token_expires_at": 0.0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _get_api_key() -> str:
    """Streamlit Secrets에서 Firebase Web API Key를 읽습니다."""
    try:
        api_key = str(st.secrets["firebase_api_key"]).strip()
    except (KeyError, TypeError, FileNotFoundError) as exc:  # secrets.toml 자체가 없으면 FileNotFoundError 계열
        raise AuthError("Firebase API Key 설정을 찾을 수 없어요. Secrets를 확인해주세요.") from exc
    if not api_key or api_key.startswith("TODO"):
        raise AuthError("Firebase API Key가 설정되지 않았어요. Secrets를 확인해주세요.")
    return api_key


def _error_message(error_code: str) -> str:
    """Firebase 오류 코드를 사용자가 이해하기 쉬운 한국어로 변환합니다."""
    messages: dict[str, str] = {
        "EMAIL_EXISTS": "이미 가입된 이메일이에요. 로그인해주세요.",
        "EMAIL_NOT_FOUND": "가입되지 않은 이메일이에요.",
        "INVALID_PASSWORD": "비밀번호가 올바르지 않아요.",
        "INVALID_LOGIN_CREDENTIALS": "이메일 또는 비밀번호가 올바르지 않아요.",
        "INVALID_EMAIL": "이메일 형식이 올바르지 않아요.",
        "WEAK_PASSWORD": "비밀번호는 6자 이상이어야 해요.",
        "USER_DISABLED": "사용이 중지된 계정이에요.",
        "TOO_MANY_ATTEMPTS_TRY_LATER": "시도가 너무 많아요. 잠시 후 다시 시도해주세요.",
        "OPERATION_NOT_ALLOWED": "이메일/비밀번호 로그인이 Firebase에서 활성화되지 않았어요.",
    }
    return messages.get(error_code, "인증에 실패했어요. 입력 내용을 확인하고 다시 시도해주세요.")


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    params: dict[str, str],
    form_encoded: bool = False,
) -> dict[str, Any]:
    """Firebase REST API에 요청을 보내고 공통 오류를 변환합니다.

    토큰 갱신 API는 공식 문서상 form-urlencoded 형식이라 form_encoded=True로 보낸다.
    """
    try:
        response = requests.post(
            url,
            params=params,
            data=payload if form_encoded else None,
            json=None if form_encoded else payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise AuthError("네트워크 오류가 발생했어요. 잠시 후 다시 시도해주세요.") from exc

    try:
        body: dict[str, Any] = response.json()
    except ValueError as exc:
        raise AuthError("인증 서버 응답을 읽을 수 없어요. 다시 시도해주세요.") from exc

    if not response.ok:
        # Firebase는 "WEAK_PASSWORD : Password should be ..."처럼 설명을 붙여 보내므로
        # 콜론 앞의 코드만 떼어 비교한다.
        raw_message = str(body.get("error", {}).get("message", "UNKNOWN"))
        error_code = raw_message.split(":", 1)[0].strip()
        raise AuthError(_error_message(error_code))
    return body


def _session_from_response(response: dict[str, Any]) -> AuthSession:
    """Firebase 응답을 앱 내부 세션 객체로 변환합니다."""
    try:
        expires_in = int(response["expiresIn"])
        return AuthSession(
            uid=str(response["localId"]),
            email=str(response.get("email", "")),
            id_token=str(response["idToken"]),
            refresh_token=str(response["refreshToken"]),
            expires_at=time.time() + expires_in,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthError("인증 응답이 올바르지 않아요. 다시 시도해주세요.") from exc


def _store_session(session: AuthSession) -> None:
    """인증 세션을 요구된 Streamlit session_state 키에 저장합니다."""
    st.session_state["uid"] = session.uid
    st.session_state["email"] = session.email
    st.session_state["idToken"] = session.id_token
    st.session_state["refreshToken"] = session.refresh_token
    st.session_state["token_expires_at"] = session.expires_at


def sign_up(email: str, password: str) -> AuthSession:
    """Firebase 이메일/비밀번호 계정을 생성하고 로그인 상태를 저장합니다."""
    normalized_email = email.strip()
    if not normalized_email:
        raise AuthError("이메일을 입력해주세요.")
    if not password:
        raise AuthError("비밀번호를 입력해주세요.")

    response = _post_json(
        AUTH_SIGN_UP_URL,
        {"email": normalized_email, "password": password, "returnSecureToken": True},
        params={"key": _get_api_key()},
    )
    session = _session_from_response(response)
    _store_session(session)
    return session


def sign_in(email: str, password: str) -> AuthSession:
    """Firebase 이메일/비밀번호 로그인을 수행하고 로그인 상태를 저장합니다."""
    normalized_email = email.strip()
    if not normalized_email:
        raise AuthError("이메일을 입력해주세요.")
    if not password:
        raise AuthError("비밀번호를 입력해주세요.")

    response = _post_json(
        AUTH_SIGN_IN_URL,
        {"email": normalized_email, "password": password, "returnSecureToken": True},
        params={"key": _get_api_key()},
    )
    session = _session_from_response(response)
    _store_session(session)
    return session


def refresh_id_token() -> str:
    """refreshToken으로 만료된 ID 토큰을 갱신하고 새 토큰을 저장합니다."""
    initialize_auth_state()
    refresh_token = st.session_state.get("refreshToken")
    if not refresh_token:
        raise AuthError("로그인 세션이 없어요. 다시 로그인해주세요.")

    response = _post_json(
        TOKEN_REFRESH_URL,
        {"grant_type": "refresh_token", "refresh_token": str(refresh_token)},
        params={"key": _get_api_key()},
        form_encoded=True,
    )
    try:
        id_token = str(response["id_token"])
        new_refresh_token = str(response.get("refresh_token", refresh_token))
        expires_in = int(response["expires_in"])
        st.session_state["idToken"] = id_token
        st.session_state["refreshToken"] = new_refresh_token
        st.session_state["token_expires_at"] = time.time() + expires_in
        if response.get("user_id"):
            st.session_state["uid"] = str(response["user_id"])
        return id_token
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthError("로그인 세션을 갱신할 수 없어요. 다시 로그인해주세요.") from exc


def get_valid_id_token() -> str:
    """유효한 ID 토큰을 반환하고 만료가 임박하면 자동 갱신합니다."""
    initialize_auth_state()
    id_token = st.session_state.get("idToken")
    expires_at = float(st.session_state.get("token_expires_at", 0.0))
    if id_token and time.time() < expires_at - TOKEN_REFRESH_MARGIN_SECONDS:
        return str(id_token)
    return refresh_id_token()


def is_authenticated() -> bool:
    """현재 Streamlit 세션에 로그인 정보가 있는지 확인합니다."""
    initialize_auth_state()
    return bool(st.session_state.get("uid") and st.session_state.get("refreshToken"))


def logout() -> None:
    """현재 브라우저 세션의 인증 정보를 제거합니다."""
    for key in ("uid", "email", "idToken", "refreshToken", "token_expires_at"):
        st.session_state.pop(key, None)
    initialize_auth_state()
