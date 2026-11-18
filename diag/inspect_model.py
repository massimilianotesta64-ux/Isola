import onnx, sys, collections
p = sys.argv[1]
m = onnx.load(p, load_external_data=False)
g = m.graph
dims = lambda t: [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
print('====', p, 'opset', [o.version for o in m.opset_import])
for i in g.input: print('  in ', i.name, dims(i), i.type.tensor_type.elem_type)
for o in g.output: print('  out', o.name, dims(o), o.type.tensor_type.elem_type)
print('  ops', collections.Counter(n.op_type for n in g.node).most_common(40))
print('  meta', [(x.key, x.value[:300]) for x in m.metadata_props])
