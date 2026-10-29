import json, re, sys
src = open(sys.argv[1]).read()
parts = re.split(r'^(# %%(?: \[markdown\])?)\s*$', src, flags=re.M)
cells = []
for i in range(1, len(parts), 2):
    kind, body = parts[i], parts[i + 1].strip('\n')
    if 'markdown' in kind:
        text = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines())
        cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True)))
    else:
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                          source=body.splitlines(keepends=True)))
nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
          metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                        language_info=dict(name='python')))
json.dump(nb, open(sys.argv[2], 'w'), indent=1)
print(len(cells), 'cells:', sum(c['cell_type'] == 'code' for c in cells), 'code,',
      sum(c['cell_type'] == 'markdown' for c in cells), 'markdown')
# every code cell preceded by a markdown cell?
bad = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and (i == 0 or cells[i - 1]['cell_type'] != 'markdown')]
print('code cells without a preceding markdown cell:', bad)
