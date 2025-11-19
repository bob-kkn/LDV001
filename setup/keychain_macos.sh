#!/usr/bin/env bash
set -euo pipefail

# 전달 받은 파일들 (경로는 적절히 변경)
P12_PATH="/path/to/DeveloperID.p12"      # 전달된 p12
P12_PASS="p12password"                   # p12 비밀번호 (CI secret)
P8_PATH="/path/to/AuthKey_ABCDEF.p8"     # App Store Connect API key (옵션)
KEY_ID="ABCDEF"                          # .p8 Key ID
ISSUER_ID="XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"  # Issuer (Team) ID

# 키체인 이름/비밀번호 (임시 키체인 권장)
KEYCHAIN_NAME="build-buildkey.keychain"
KEYCHAIN_PASS="TempKeychainPass123!"     # CI에서 secret으로 주입

# 1) 임시 키체인 생성 및 기본 키체인으로 추가
security create-keychain -p "$KEYCHAIN_PASS" "$KEYCHAIN_NAME"
security default-keychain -s "$KEYCHAIN_NAME"
security unlock-keychain -p "$KEYCHAIN_PASS" "$KEYCHAIN_NAME"
security set-keychain-settings -t 3600 -l "$KEYCHAIN_NAME"

# 2) p12 임포트 (private key 포함)
security import "$P12_PATH" -k ~/Library/Keychains/"$KEYCHAIN_NAME" -P "$P12_PASS" -T /usr/bin/codesign

# 3) 키 파티션 리스트 설정 (codesign이 사용할 수 있도록)
# (macOS 버전에 따라 필요할 수 있음)
security set-key-partition-list -S apple-tool:,apple: -s -k "$KEYCHAIN_PASS" ~/Library/Keychains/"$KEYCHAIN_NAME"

# 4) codesign 사용 예시 (경로/식별자 변경)
DEVELOPER_ID="Developer ID Application: KnWorks Co., Ltd. (TEAMID)"
APP_PATH="/path/to/dist/LDV001.app"

codesign --verbose --deep --options runtime --timestamp \
  --sign "$DEVELOPER_ID" \
  --keychain ~/Library/Keychains/"$KEYCHAIN_NAME" \
  "$APP_PATH"

# 5) 검증
codesign --verify --deep --strict --verbose=2 "$APP_PATH"
spctl --assess --type execute --verbose "$APP_PATH" || true

# 6) notarize (App Store Connect API key 사용 권장)
if [[ -f "$P8_PATH" ]]; then
  xcrun notarytool submit "$APP_PATH" \
    --key "$P8_PATH" --key-id "$KEY_ID" --issuer "$ISSUER_ID" \
    --wait

  xcrun stapler staple "$APP_PATH"
else
  echo "No .p8 key found: skipping notarization step"
fi

# 7) 정리: 임시 키체인 삭제 (중요)
security delete-keychain "$KEYCHAIN_NAME"
