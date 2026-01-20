#!/usr/bin/env python3
"""
Quick probe for DSP gap: per design, parse loops/unroll from kernel adb, map dp_component_resource to loops if possible,
and apply naive unroll factor multiplication. Targets: symm_design_1001, syr2k_design_312, trmm_design_892.
"""
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from collections import defaultdict

CASES = [
    ("symm_design_1001", "/home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_500designs/PolyBench/symm/design_1001"),
    ("syr2k_design_312", "/home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_500designs/PolyBench/syr2k/design_312"),
    ("trmm_design_892", "/home/user/zedongpeng/workspace/Huggingface/forgehls_PolyBench_part_500designs/PolyBench/trmm/design_892"),
]


def parse_csynth_dsp(csynth_path: Path):
    try:
        tree = ET.parse(csynth_path)
        root = tree.getroot()
        return int(root.findtext('.//AreaEstimates/Resources/DSP'))
    except Exception:
        return None


def parse_loops(adb_path: Path):
    loops = {}
    try:
        tree = ET.parse(adb_path)
        root = tree.getroot()
        for lp in root.iter('loop'):  # <loop><loop_id> etc.
            loop_id = lp.findtext('loop_id') or lp.findtext('id')
            if not loop_id:
                continue
            info = {
                'name': lp.findtext('name') or '',
                'unroll': lp.findtext('unroll_factor') or '',
                'pipeline': lp.findtext('pipeline') or '',
                'ii': lp.findtext('II') or '',
            }
            loops[loop_id] = info
    except Exception:
        pass
    return loops


def parse_dp_resources(adb_path: Path):
    nodes = {}
    try:
        tree = ET.parse(adb_path)
        root = tree.getroot()
        for dcr in root.iter('dp_component_resource'):
            for item in dcr.findall('item'):
                name = item.findtext('first') or 'unknown'
                dsp = 0
                second = item.find('second')
                if second is not None:
                    for res_item in second.findall('item'):
                        if res_item.findtext('first') == 'DSP':
                            try:
                                dsp = int(res_item.findtext('second'))
                            except Exception:
                                dsp = 0
                if dsp:
                    nodes[name] = {'dsp': dsp}
        # try to get loop binding via regex on text (heuristic: "loop_id=xxx")
        text = ET.tostring(root, encoding='unicode', method='xml')
        for name in nodes.keys():
            m = re.search(rf"{re.escape(name)}.*?loop_id>([^<]+)<", text)
            if m:
                nodes[name]['loop_id'] = m.group(1)
    except Exception:
        pass
    return nodes


def aggregate_design(design_path: Path):
    csynths = list(design_path.rglob('csynth.xml'))
    csynth_dsp = parse_csynth_dsp(csynths[0]) if csynths else None
    adb_files = [p for p in design_path.rglob('*.adb') if p.name.count('.') <= 1]
    kernel_adb = None
    for p in adb_files:
        if p.name.startswith('kernel_') and p.suffix == '.adb':
            kernel_adb = p
            break
    loops = parse_loops(kernel_adb) if kernel_adb else {}

    node_records = []
    total_raw = 0
    for adb in adb_files:
        dps = parse_dp_resources(adb)
        for name, info in dps.items():
            dsp = info['dsp']
            loop_id = info.get('loop_id')
            node_records.append((adb.name, name, dsp, loop_id))
            total_raw += dsp

    # naive unroll adjustment: if loop_id matches, multiply by unroll_factor (int) else 1
    total_adjusted = 0
    adjusted_nodes = []
    for adb_name, name, dsp, loop_id in node_records:
        factor = 1
        if loop_id and loop_id in loops:
            u = loops[loop_id].get('unroll') or ''
            try:
                f = int(u)
                if f > 1:
                    factor = f
            except Exception:
                pass
        adjusted = dsp * factor
        adjusted_nodes.append((adb_name, name, dsp, factor, adjusted))
        total_adjusted += adjusted

    return {
        'csynth_dsp': csynth_dsp,
        'total_raw': total_raw,
        'total_adjusted': total_adjusted,
        'loops': loops,
        'nodes': adjusted_nodes,
    }


def main():
    out = []
    for pid, path_str in CASES:
        path = Path(path_str)
        res = aggregate_design(path)
        out.append((pid, res))
    report = Path('dsp_gap_probe/report.txt')
    with report.open('w', encoding='utf-8') as f:
        for pid, res in out:
            f.write(f"Design {pid}\n")
            f.write(f"  csynth DSP: {res['csynth_dsp']}\n")
            f.write(f"  ADB raw DSP sum: {res['total_raw']}\n")
            f.write(f"  ADB adjusted DSP (naive unroll): {res['total_adjusted']}\n")
            if res['loops']:
                f.write("  Loops (id -> name/unroll/II/pipeline):\n")
                for lid, info in res['loops'].items():
                    f.write(f"    {lid}: name={info['name']} unroll={info['unroll']} II={info['ii']} pipeline={info['pipeline']}\n")
            f.write("  Nodes (adb, name, dsp, factor, adjusted):\n")
            for adb_name, name, dsp, factor, adj in res['nodes']:
                f.write(f"    {adb_name}: {name} dsp={dsp} factor={factor} adj={adj}\n")
            f.write("\n")
    print(f"Done. Report at {report}")

if __name__ == "__main__":
    main()
