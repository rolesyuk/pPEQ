if ! command -v pyinstaller 2>&1 >/dev/null
then
    echo "PyInstaller is not found"
    exit 1
fi

rm -f pPEQ pPEQ.zip
pyinstaller --distpath . \
            --noupx \
            --onefile \
            --add-data pulse_effects_config_top.json:. \
            --add-data pulse_effects_config_bottom.json:. \
            pPEQ.py
rm -r build pPEQ.spec

pushd ..
    zip -r pPEQ/pPEQ.zip \
        pPEQ/pPEQ \
        pPEQ/peqs/.gitkeep \
        pPEQ/responses \
        pPEQ/songs/Chirp.wav \
        pPEQ/targets \
        pPEQ/ABX.sha256
popd
