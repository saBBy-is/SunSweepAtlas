import zipfile, os, shutil

PPTX_SRC = r"c:\Users\akoli\OneDrive\Desktop\SIH26166_SunSweepAtlas_FINAL.pptx"
PPTX_BAK = r"c:\Users\akoli\OneDrive\Desktop\SIH26166_SunSweepAtlas_FINAL_backup_slide5.pptx"
PPTX_TMP = r"c:\Users\akoli\OneDrive\Desktop\SIH26166_SunSweepAtlas_FINAL_tmp.pptx"

# Backup
shutil.copy2(PPTX_SRC, PPTX_BAK)
print(f"Created backup at {PPTX_BAK}")

old_stat = "44%"
new_stat = "21%"

old_subtext = "Of our render in cast shadow at 5° Sun elevation, rising to 60% at 3°"
new_subtext = "Of our render in cast shadow at 10° Sun elevation, up from 2.5% at 40°"

with zipfile.ZipFile(PPTX_SRC, 'r') as zin, zipfile.ZipFile(PPTX_TMP, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        content = zin.read(item.filename)
        if item.filename == 'ppt/slides/slide5.xml':
            print("Updating text in ppt/slides/slide5.xml...")
            xml_str = content.decode('utf-8')
            assert old_stat in xml_str, f"Could not find '{old_stat}' in slide5.xml!"
            assert old_subtext in xml_str, f"Could not find '{old_subtext}' in slide5.xml!"

            xml_str = xml_str.replace(old_stat, new_stat)
            xml_str = xml_str.replace(old_subtext, new_subtext)
            zout.writestr(item, xml_str.encode('utf-8'))
        else:
            zout.writestr(item, content)

shutil.move(PPTX_TMP, PPTX_SRC)
print(f"Successfully updated {PPTX_SRC} in place.")

# Verification
with zipfile.ZipFile(PPTX_SRC, 'r') as zcheck:
    assert zcheck.testzip() is None, "Zip integrity error!"
    s5 = zcheck.read('ppt/slides/slide5.xml').decode('utf-8')
    assert new_stat in s5, f"Verification failed: '{new_stat}' missing in slide 5!"
    assert new_subtext in s5, f"Verification failed: '{new_subtext}' missing in slide 5!"
    assert old_stat not in s5, f"Verification failed: old stat '{old_stat}' still present in slide 5!"
    print("Verification PASSED: Slide 5 updated with honest 10-deg/40-deg shadow statistics.")
