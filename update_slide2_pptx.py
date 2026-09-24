import zipfile, os, shutil
import cv2

PPTX_SRC = r"SIH26166_SunSweepAtlas_FINAL.pptx"
PPTX_BAK = r"SIH26166_SunSweepAtlas_FINAL_backup.pptx"
PPTX_TMP = r"SIH26166_SunSweepAtlas_FINAL_tmp.pptx"

# 1. Create backup if it doesn't already exist
if not os.path.exists(PPTX_BAK):
    shutil.copy2(PPTX_SRC, PPTX_BAK)
    print(f"Created backup at {PPTX_BAK}")
else:
    print(f"Backup already exists at {PPTX_BAK}")

# 2. Prepare high-quality replacement for image3.jpg from slide2_4panel_honest_el10.png
panel_src = "results/slide2_4panel_honest_el10.png"
assert os.path.exists(panel_src), f"Missing {panel_src}!"
im = cv2.imread(panel_src)
# Resize to 1894 x 578 (exact aspect ratio and dimension of original image3.jpg)
# or keep crisp resolution: 1894x578 or 2400x733
im_resized = cv2.resize(im, (1894, 578), interpolation=cv2.INTER_AREA)
jpg_bytes = cv2.imencode('.jpg', im_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
print(f"Encoded replacement image3.jpg: {len(jpg_bytes)} bytes")

# 3. Read PPTX and update slide2.xml and ppt/media/image3.jpg
with zipfile.ZipFile(PPTX_SRC, 'r') as zin, zipfile.ZipFile(PPTX_TMP, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        content = zin.read(item.filename)
        if item.filename == 'ppt/media/image3.jpg':
            print("Replacing ppt/media/image3.jpg with honest 10-deg composite...")
            zout.writestr(item, jpg_bytes)
        elif item.filename == 'ppt/slides/slide2.xml':
            print("Updating text in ppt/slides/slide2.xml...")
            xml_str = content.decode('utf-8')
            # Check replacements
            assert 'NCC −0.74' in xml_str, "Could not find 'NCC −0.74' in slide2.xml!"
            xml_str = xml_str.replace('NCC −0.74', 'NCC −0.54')

            assert '760 × 760 px · 2 m/px' in xml_str, "Could not find '760 × 760 px · 2 m/px' in slide2.xml!"
            xml_str = xml_str.replace('760 × 760 px · 2 m/px', '512 × 512 px · 20 m/px')

            zout.writestr(item, xml_str.encode('utf-8'))
        else:
            zout.writestr(item, content)

# 4. Overwrite original with updated tmp
shutil.move(PPTX_TMP, PPTX_SRC)
print(f"Successfully updated {PPTX_SRC} in place.")

# 5. Verify the updated PPTX
with zipfile.ZipFile(PPTX_SRC, 'r') as zcheck:
    test_res = zcheck.testzip()
    assert test_res is None, f"Zip integrity error: {test_res}"
    s2 = zcheck.read('ppt/slides/slide2.xml').decode('utf-8')
    assert 'NCC −0.54' in s2, "Verification failed: 'NCC −0.54' missing!"
    assert '512 × 512 px · 20 m/px' in s2, "Verification failed: '512 × 512 px · 20 m/px' missing!"
    img3_size = len(zcheck.read('ppt/media/image3.jpg'))
    print(f"Verification PASSED: slide2 text verified, image3 size = {img3_size} bytes.")
