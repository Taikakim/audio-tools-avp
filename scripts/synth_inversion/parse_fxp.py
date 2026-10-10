import sys, re
import xml.etree.ElementTree as ET

def get_params_from_fxp(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # Try to find <patch ... > ... </patch>
    start = data.find(b'<patch')
    end = data.find(b'</patch>') + 8
    if start == -1 or end == -1:
        return None
        
    xml_data = data[start:end]
    try:
        root = ET.fromstring(xml_data.decode('utf-8', errors='ignore'))
        params = {}
        for child in root:
            if child.tag == 'parameters':
                for param in child:
                    name = param.tag
                    val = param.attrib.get('value', None)
                    if val is not None:
                        params[name] = float(val)
        return params
    except Exception as e:
        print(e)
        return None

if __name__ == "__main__":
    import glob
    fxps = glob.glob("kim_bass_patches/*.fxp")
    all_params = {}
    for f in fxps:
        p = get_params_from_fxp(f)
        if p:
            for k, v in p.items():
                if k not in all_params: all_params[k] = []
                all_params[k].append(v)
    
    if not all_params:
        print("No params found")
        sys.exit(1)
        
    for k in sorted(all_params.keys()):
        vals = all_params[k]
        mi, ma = min(vals), max(vals)
        if mi != ma and (k.startswith('a_') or k.startswith('fx_')):
            print(f"{k}: min={mi:.4f} max={ma:.4f}")
