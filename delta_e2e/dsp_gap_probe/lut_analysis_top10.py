#!/usr/bin/env python3
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

TOP_CSV = Path('output/train_split_error_analysis/test_design_top10_mape.csv')
OUT_DIR = Path('output/train_split_error_analysis/lut_analysis_top10')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_csynth_lut(csynth_path: Path):
    try:
        root = ET.parse(csynth_path).getroot()
        lut = root.findtext('.//AreaEstimates/Resources/LUT')
        return int(lut) if lut else None
    except Exception:
        return None


def parse_adb_lut(adb_path: Path):
    total = 0
    found = False
    try:
        tree = ET.parse(adb_path)
        root = tree.getroot()
        for dcr in root.iter('dp_component_resource'):
            for item in dcr.findall('item'):
                second = item.find('second')
                if second is None:
                    continue
                for res_item in second.findall('item'):
                    if res_item.findtext('first') == 'LUT':
                        try:
                            total += int(res_item.findtext('second'))
                            found = True
                        except Exception:
                            pass
    except Exception:
        pass
    if not found:
        try:
            text = adb_path.read_text(errors='ignore')
            for m in re.finditer(r'<first>LUT</first>\s*<second>(\d+)</second>', text):
                total += int(m.group(1))
                found = True
        except Exception:
            pass
    return total if found else 0


def main():
    rows = pd.read_csv(TOP_CSV).to_dict('records')
    out_path = OUT_DIR / 'lut_report.txt'
    with out_path.open('w', encoding='utf-8') as out:
        for row in rows:
            design_path = Path(row['design_base_path'])
            csynths = list(design_path.rglob('csynth.xml'))
            lut_total = parse_csynth_lut(csynths[0]) if csynths else None
            lut_adb_sum = 0
            for adb in design_path.rglob('*.adb'):
                lut_adb_sum += parse_adb_lut(adb)
            match = (lut_total is not None and lut_total == lut_adb_sum)
            out.write(f"Design: {row['pair_id']}\n")
            out.write(f"路径: {design_path}\n")
            out.write(f"- 总LUT资源（来自.xml）: {lut_total}\n")
            out.write(f"- ADB LUT 求和: {lut_adb_sum}\n")
            out.write(f"- 是否匹配: {'匹配' if match else '不匹配'}\n\n")
    print(f"done -> {out_path}")

if __name__ == '__main__':
    main()
