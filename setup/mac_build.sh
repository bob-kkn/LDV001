#!/usr/bin/env bash
set -euo pipefail

########################
# 환경 변수 (필요에 맞게 수정)
########################
PROJECT_NAME="main"                       # PyInstaller spec의 베이스 이름
SPEC_FILE="${PROJECT_NAME}.spec"          # PyInstaller spec 파일
APP_NAME="LDV001.app"                     # PyInstaller 결과물 .app 이름
APP_BUNDLE_ID="kr.co.knworks.ldv001"      # 앱 번들 ID (Info.plist와 일치)
TEAM_ID="T8WWF3ND26"                      # Apple Developer Team ID
DEVELOPER_ID_APP="Developer ID Application: KnWorks Co., Ltd. (${TEAM_ID})"

DIST_DIR="dist"
BUILD_DIR="build"
DMG_NAME="LDV001Setup.dmg"                # macOS 배포 DMG 파일명
VOL_NAME="LDV001 Installer"               # DMG 볼륨명

# PyArmor: 보호 대상 파이썬 파일들
PYARMOR_SOURCES=("pano_blur.py" "pano_blur_gui.py")

# NotaryTool 프로파일명 (미리 저장해둔 자격 증명 별칭)
NOTARY_PROFILE="KNWORKS_NOTARY"

########################
# 유틸 함수
########################
sign_path() {
  local target="$1"
  echo "[codesign] $target"
  codesign --force --deep --timestamp \
    --options runtime \
    --sign "$DEVELOPER_ID_APP" \
    "$target"
}

########################
# 0) 정리
########################
echo "==> Cleanup"
rm -rf "$DIST_DIR" "$BUILD_DIR"
mkdir -p "$DIST_DIR"

########################
# 1) (선택) PyArmor 보호 + PyInstaller 패킹
########################
# Windows 스크립트의 `pyarmor gen --pack ...`과 유사하게 동작
# PyArmor가 spec 기반으로 PyInstaller를 호출하도록 합니다.
echo "==> PyArmor protect & pack"
pyarmor gen --pack "$SPEC_FILE" -r "${PYARMOR_SOURCES[@]}"

# 위 명령으로 dist/ 하위에 .app 생성이 끝납니다.
# (만약 .app이 아니라 단일 파일을 만들었다면, spec 또는 .spec의 EXE->BUNDLE 설정을 .app으로 조정하세요.)

########################
# 2) 코드 서명 (앱 번들 내부까지 꼼꼼히)
########################
APP_PATH="${DIST_DIR}/${APP_NAME}"
if [[ ! -d "$APP_PATH" ]]; then
  echo "ERROR: ${APP_PATH} 가 존재하지 않습니다. spec 또는 PyArmor 설정을 확인하세요." >&2
  exit 1
fi

echo "==> Signing embedded frameworks / binaries"
# 번들 내부 바이너리/프레임워크/확장 모듈 등을 하위부터 서명
# 필요 시 아래 find 패턴을 프로젝트 구조에 맞춰 보강하세요.
while IFS= read -r f; do
  sign_path "$f"
done < <(find "$APP_PATH" -type f \( -perm +111 -o -name "*.so" -o -name "*.dylib" -o -name "*.bin" \) 2>/dev/null || true)

# 최상위 .app 서명
echo "==> Signing .app"
sign_path "$APP_PATH"

# 서명 검증
codesign --verify --deep --strict --verbose=2 "$APP_PATH"
spctl --assess --type execute --verbose "$APP_PATH" || true

########################
# 3) DMG 생성 (hdiutil 사용)
########################
echo "==> Creating DMG"
# 임시 폴더에 앱 복사 후 DMG 생성
TMP_DMG_DIR="$(mktemp -d)"
cp -R "$APP_PATH" "$TMP_DMG_DIR/"

hdiutil create \
  -volname "$VOL_NAME" \
  -srcfolder "$TMP_DMG_DIR" \
  -ov -format UDZO \
  "${DIST_DIR}/${DMG_NAME}"

rm -rf "$TMP_DMG_DIR"

########################
# 4) DMG 서명 (권장)
########################
echo "==> Signing DMG"
sign_path "${DIST_DIR}/${DMG_NAME}"

########################
# 5) 공증(Notarization) + 스테이플
########################
echo "==> Notarize"
xcrun notarytool submit "${DIST_DIR}/${DMG_NAME}" \
  --keychain-profile "$NOTARY_PROFILE" \
  --wait

echo "==> Staple"
xcrun stapler staple "${DIST_DIR}/${DMG_NAME}"

########################
# 6) SHA256 해시 출력
########################
echo "==> SHA256"
shasum -a 256 "${DIST_DIR}/${DMG_NAME}"
echo "Done: ${DIST_DIR}/${DMG_NAME}"
