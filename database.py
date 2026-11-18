"""Firebase Admin SDK를 이용한 사용자별 Firestore 접근을 관리합니다."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Mapping

import streamlit as st
from firebase_admin import credentials, firestore, get_app, initialize_app
from google.cloud.firestore_v1 import Client, DocumentReference

FIREBASE_SERVICE_ACCOUNT_KEY: Final[str] = "firebase_service_account"
USER_COLLECTION: Final[str] = "users"
PROPULSION_COLLECTION: Final[str] = "propulsion_sessions"
POSTURE_COLLECTION: Final[str] = "posture_sessions"
STRETCH_COLLECTION: Final[str] = "stretch_logs"
CVA_CHECK_COLLECTION: Final[str] = "cva_checks"


class DatabaseError(Exception):
    """Firestore 오류를 앱에서 안전하게 안내하기 위한 예외입니다."""


def _service_account_info() -> dict[str, Any]:
    """Streamlit Secrets에서 Firebase 서비스 계정 정보를 읽습니다."""
    try:
        raw_info = st.secrets[FIREBASE_SERVICE_ACCOUNT_KEY]
    except (KeyError, TypeError, FileNotFoundError) as exc:  # secrets.toml 자체가 없으면 FileNotFoundError 계열
        raise DatabaseError("Firebase 서비스 계정 설정을 찾을 수 없어요.") from exc

    info = dict(raw_info)
    if not info or str(info.get("project_id", "")).startswith("TODO"):
        raise DatabaseError("Firebase 서비스 계정이 설정되지 않았어요.")
    return info


def _firestore_client() -> Client:
    """Firebase Admin 앱을 한 번만 초기화하고 Firestore 클라이언트를 반환합니다."""
    try:
        app = get_app()
    except ValueError:
        try:
            app = initialize_app(credentials.Certificate(_service_account_info()))
        except Exception as exc:
            raise DatabaseError("Firebase 초기화에 실패했어요. 설정을 확인해주세요.") from exc

    try:
        return firestore.client(app=app)
    except Exception as exc:
        raise DatabaseError("Firestore에 연결할 수 없어요. 잠시 후 다시 시도해주세요.") from exc


def _require_uid(uid: str) -> str:
    """모든 데이터 접근 함수가 비어 있지 않은 UID를 받도록 강제합니다."""
    normalized_uid = str(uid).strip()
    if not normalized_uid:
        raise DatabaseError("사용자 인증 정보가 없어요. 다시 로그인해주세요.")
    return normalized_uid


def _user_reference(client: Client, uid: str) -> DocumentReference:
    """사용자 UID 아래의 Firestore 문서 참조만 생성합니다."""
    return client.collection(USER_COLLECTION).document(_require_uid(uid))


def _server_timestamp() -> Any:
    # 저장 함수의 created_at 인자를 주면 그 시각으로 저장한다 (데모 데이터를 여러 날짜에 흩뿌릴 때 사용).
    """Firestore 서버 시각을 반환합니다."""
    return firestore.SERVER_TIMESTAMP


def _write_document(reference: DocumentReference, data: Mapping[str, Any], *, merge: bool = True) -> None:
    """Firestore 문서를 저장하고 네트워크 오류를 앱용 예외로 변환합니다."""
    try:
        reference.set(dict(data), merge=merge)
    except Exception as exc:
        raise DatabaseError("저장에 실패했어요. 다시 시도해주세요.") from exc


def _read_documents(reference: Any) -> list[dict[str, Any]]:
    """컬렉션 문서를 읽고 문서 ID를 포함한 사전 목록으로 변환합니다."""
    try:
        documents = reference.stream()
        return [{"id": document.id, **document.to_dict()} for document in documents]
    except Exception as exc:
        raise DatabaseError("기록을 불러오지 못했어요. 다시 시도해주세요.") from exc


def ensure_user_document(uid: str, email: str) -> None:
    """users/{uid} 문서를 만들거나 이메일을 갱신합니다.

    created_at은 문서가 처음 만들어질 때만 기록한다 (로그인마다 덮어쓰지 않음).
    """
    client = _firestore_client()
    user_ref = _user_reference(client, uid)
    data: dict[str, Any] = {"email": email.strip()}
    try:
        if not user_ref.get().exists:
            data["created_at"] = _server_timestamp()
    except Exception as exc:
        raise DatabaseError("사용자 정보를 불러오지 못했어요. 다시 시도해주세요.") from exc
    _write_document(user_ref, data, merge=True)


def update_user_profile(uid: str, profile: Mapping[str, Any]) -> None:
    """users/{uid}/profile 필드에 림 지름과 주 사용 손을 저장합니다."""
    client = _firestore_client()
    _write_document(_user_reference(client, uid), {"profile": dict(profile)}, merge=True)


def set_consent(uid: str, consent_agreed: bool) -> None:
    """민감정보 수집 동의 여부와 동의 시각을 사용자 문서에 저장합니다."""
    client = _firestore_client()
    consent_at: Any = _server_timestamp() if consent_agreed else None
    _write_document(
        _user_reference(client, uid),
        {"consent_agreed": bool(consent_agreed), "consent_at": consent_at},
        merge=True,
    )


def get_user_document(uid: str) -> dict[str, Any] | None:
    """users/{uid} 문서를 조회합니다."""
    client = _firestore_client()
    try:
        snapshot = _user_reference(client, uid).get()
    except Exception as exc:
        raise DatabaseError("사용자 정보를 불러오지 못했어요. 다시 시도해주세요.") from exc
    if not snapshot.exists:
        return None
    return snapshot.to_dict()


def save_propulsion_session(
    uid: str,
    *,
    view: str,
    filmed_side: str,
    metrics: Mapping[str, Any],
    judgments: Mapping[str, str],
    is_demo: bool = False,
    created_at: datetime | None = None,
    answers: Mapping[str, Any] | None = None,
    posture_photo: str | None = None,
) -> str:
    """사용자 하위 추진 분석 세션에 원본 영상 없이 측정값(과 그때의 질문 답변)만 저장합니다.

    posture_photo: 사용자가 고른 경우에만, 얼굴을 흐리고 머리~가슴만 잘라 낸 자세 사진 1장(JPEG base64, 약 30KB).
    """
    client = _firestore_client()
    session_ref = _user_reference(client, uid).collection(PROPULSION_COLLECTION).document()
    _write_document(
        session_ref,
        {
            "created_at": created_at or _server_timestamp(),
            "view": view,
            "filmed_side": filmed_side,
            "metrics": dict(metrics),
            "judgments": dict(judgments),
            "answers": dict(answers or {}),
            "is_demo": bool(is_demo),
            **({"posture_photo": posture_photo} if posture_photo else {}),
        },
        merge=False,
    )
    return session_ref.id


def save_posture_session(
    uid: str,
    *,
    metrics: Mapping[str, Any],
    judgments: Mapping[str, str],
    is_demo: bool = False,
    created_at: datetime | None = None,
) -> str:
    """사용자 하위 정지 자세 세션에 측정값만 저장합니다."""
    client = _firestore_client()
    session_ref = _user_reference(client, uid).collection(POSTURE_COLLECTION).document()
    _write_document(
        session_ref,
        {
            "created_at": created_at or _server_timestamp(),
            "metrics": dict(metrics),
            "judgments": dict(judgments),
            "is_demo": bool(is_demo),
        },
        merge=False,
    )
    return session_ref.id


def save_stretch_log(uid: str, exercise_id: str, created_at: datetime | None = None) -> str:
    """사용자 하위 스트레칭 완료 기록을 저장합니다."""
    client = _firestore_client()
    log_ref = _user_reference(client, uid).collection(STRETCH_COLLECTION).document()
    _write_document(
        log_ref,
        {"created_at": created_at or _server_timestamp(), "exercise_id": exercise_id},
        merge=False,
    )
    return log_ref.id


def save_cva_check(uid: str, *, manual_cva: float, auto_cva: float, created_at: datetime | None = None) -> str:
    """거북목 각도 비교(논문 방식 C7 탭 vs 앱 자동) 한 건을 저장합니다. 사진·좌표는 저장하지 않고 각도만."""
    client = _firestore_client()
    check_ref = _user_reference(client, uid).collection(CVA_CHECK_COLLECTION).document()
    _write_document(
        check_ref,
        {"created_at": created_at or _server_timestamp(), "manual_cva": float(manual_cva), "auto_cva": float(auto_cva)},
        merge=False,
    )
    return check_ref.id


def list_cva_checks(uid: str) -> list[dict[str, Any]]:
    """거북목 각도 비교 기록을 최신순으로 조회합니다."""
    client = _firestore_client()
    reference = _user_reference(client, uid).collection(CVA_CHECK_COLLECTION).order_by(
        "created_at", direction=firestore.Query.DESCENDING
    )
    return _read_documents(reference)


def list_propulsion_sessions(uid: str) -> list[dict[str, Any]]:
    """사용자의 추진 분석 세션만 최신순으로 조회합니다."""
    client = _firestore_client()
    reference = _user_reference(client, uid).collection(PROPULSION_COLLECTION).order_by(
        "created_at", direction=firestore.Query.DESCENDING
    )
    return _read_documents(reference)


def list_posture_sessions(uid: str) -> list[dict[str, Any]]:
    """사용자의 정지 자세 세션만 최신순으로 조회합니다."""
    client = _firestore_client()
    reference = _user_reference(client, uid).collection(POSTURE_COLLECTION).order_by(
        "created_at", direction=firestore.Query.DESCENDING
    )
    return _read_documents(reference)


def list_stretch_logs(uid: str) -> list[dict[str, Any]]:
    """사용자의 스트레칭 기록만 최신순으로 조회합니다."""
    client = _firestore_client()
    reference = _user_reference(client, uid).collection(STRETCH_COLLECTION).order_by(
        "created_at", direction=firestore.Query.DESCENDING
    )
    return _read_documents(reference)


def _delete_collection_documents(collection_reference: Any) -> int:
    """한 하위 컬렉션의 직접 문서를 삭제하고 삭제 개수를 반환합니다."""
    deleted_count = 0
    try:
        for document in collection_reference.stream():
            document.reference.delete()
            deleted_count += 1
    except Exception as exc:
        raise DatabaseError("데이터 삭제에 실패했어요. 다시 시도해주세요.") from exc
    return deleted_count


def delete_demo_data(uid: str) -> int:
    """사용자 UID 아래에서 is_demo=true인 세션만 삭제합니다."""
    client = _firestore_client()
    user_ref = _user_reference(client, uid)
    deleted_count = 0
    for collection_name in (PROPULSION_COLLECTION, POSTURE_COLLECTION):
        try:
            demo_query = user_ref.collection(collection_name).where("is_demo", "==", True)
            for document in demo_query.stream():
                document.reference.delete()
                deleted_count += 1
        except Exception as exc:
            raise DatabaseError("데모 데이터 삭제에 실패했어요. 다시 시도해주세요.") from exc
    return deleted_count


def delete_all_user_data(uid: str) -> int:
    """사용자 UID 아래의 알려진 하위 컬렉션과 사용자 문서를 모두 삭제합니다."""
    client = _firestore_client()
    user_ref = _user_reference(client, uid)
    deleted_count = 0
    for collection_name in (PROPULSION_COLLECTION, POSTURE_COLLECTION, STRETCH_COLLECTION, CVA_CHECK_COLLECTION):
        deleted_count += _delete_collection_documents(user_ref.collection(collection_name))
    try:
        if user_ref.get().exists:
            user_ref.delete()
            deleted_count += 1
    except Exception as exc:
        raise DatabaseError("데이터 삭제에 실패했어요. 다시 시도해주세요.") from exc
    return deleted_count
