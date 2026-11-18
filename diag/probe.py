import json, urllib.request, sys
repos = ['kramp/htdemucs-6s-webgpu-onnx','Andremi/audioweb-models','StemSplitio/htdemucs-6s-onnx','timcsy/demucs-web-onnx']
for r in repos:
    try:
        d = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{r}/tree/main?recursive=1'))
        print('##', r)
        for f in d: print('  ', f.get('path'), f.get('size'))
    except Exception as e: print('##', r, 'ERR', e)
for r, f in [('Andremi/audioweb-models','README.md'),('kramp/htdemucs-6s-webgpu-onnx','README.md')]:
    try:
        t = urllib.request.urlopen(f'https://huggingface.co/{r}/resolve/main/{f}').read().decode()
        print('#### README', r); print(t[:6000])
    except Exception as e: print('README ERR', r, e)
