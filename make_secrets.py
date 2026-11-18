"""Firebase 서비스 계정 JSON과 웹 API 키로 .streamlit/secrets.toml을 자동으로 만듭니다.

사용법 (wheelance 폴더에서):
    python make_secrets.py 다운로드한파일.json

실행하면 웹 API 키(AIza로 시작)를 물어봅니다. 비밀값은 이 컴퓨터 안에서만 처리됩니다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FIELDS = (
    "type", "project_id", "private_key_id", "private_key", "client_email", "client_id",
    "auth_uri", "token_uri", "auth_provider_x509_cert_url", "client_x509_cert_url", "universe_domain",
)


def toml_string(value: str) -> str:
    """TOML 문자열로 안전하게 감쌉니다 (줄바꿈은 \\n으로)."""
    return json.dumps(value, ensure_ascii=False)


def main() -> None:
    """JSON을 읽고 API 키를 입력받아 secrets.toml을 씁니다."""
    if len(sys.argv) != 2:
        print("사용법: python make_secrets.py 다운로드한파일.json")
        sys.exit(1)
    json_path = Path(sys.argv[1]).expanduser()
    try:
        info = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"JSON 파일을 읽지 못했어요: {error}")
        sys.exit(1)
    if info.get("type") != "service_account" or "private_key" not in info:
        print("서비스 계정 JSON이 아닌 것 같아요. Firebase '새 비공개 키 생성'으로 받은 파일인지 확인해주세요.")
        sys.exit(1)

    api_key = input("웹 API 키를 붙여넣고 Enter (AIza로 시작): ").strip().strip('"')
    if not api_key.startswith("AIza"):
        print("웹 API 키는 보통 AIza로 시작해요. 다시 확인해주세요.")
        sys.exit(1)

    lines = [
        "# make_secrets.py로 자동 생성됨. 절대 GitHub에 올리지 마세요.",
        f"firebase_api_key = {toml_string(api_key)}",
        "",
        "[firebase_service_account]",
    ]
    lines += [f"{field} = {toml_string(str(info[field]))}" for field in FIELDS if field in info]

    target = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    target.parent.mkdir(exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"완료! {target} 를 만들었어요 (프로젝트: {info.get('project_id')}).")
    print("이제 다운로드한 JSON 파일은 안전한 곳에 옮기거나 지워도 돼요.")


if __name__ == "__main__":
    main()
