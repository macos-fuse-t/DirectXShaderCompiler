#!/usr/bin/env python3
"""Check opt-in flat bit transport, producer linkage, and unaffected interfaces.
Usage: test.py <DXC build directory> [LLVM disassembler]
"""
import ctypes as C
from pathlib import Path
import struct, subprocess, sys, tempfile
build = Path(sys.argv[1]).resolve()
dis = Path(sys.argv[2]) if len(sys.argv)>2 else build/'bin/llvm-dis'
lib = C.CDLL(str(build/'lib/libScorpiDxbcToDxil.dylib'))
class Element(C.Structure):
    _fields_=[('name',C.c_char_p)]+[(key,C.c_uint32) for key in
        ['semantic','register','first','components','type','output','flat']]
inspect = lib.ScorpiInspectShaderLinkage
inspect.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t]
lower=lib.ScorpiLowerTypedBufferViewsWithSamplingPolicy
lower.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_uint32,C.c_uint32,C.c_uint32,C.c_void_p,C.c_size_t,C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t]
retype=lib.ScorpiRetypeShaderOutputs
retype.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t]
lib.ScorpiFreeDxilBinary.argtypes=[C.c_void_p]
fixture=Path(__file__).with_name('interface.hlsl')
def reflect(code):
    records=C.POINTER(Element)();count=C.c_size_t();error=C.create_string_buffer(2048)
    rc=inspect(code,len(code),None,0,C.byref(records),C.byref(count),error,len(error))
    assert rc==0,(rc,error.value)
    try:
        return [dict(name=records[i].name.decode().lower(),**{key:getattr(records[i],key)
            for key,_ in Element._fields_ if key!='name'}) for i in range(count.value)]
    finally:lib.ScorpiFreeDxilBinary(records)
def link(producer,consumer):
    types=(C.c_uint16*32)()
    for output in producer:
        if not output['output']:continue
        for input in consumer:
            if input['output'] or (output['name'],output['semantic'])!=(input['name'],input['semantic']):continue
            for col in range(min(output['components'],input['components'])):
                types[output['register']]|=input['type']<<(4*(output['first']+col))
    return types
def adapt(code,options=128,types=None):
    out,size,error=C.c_void_p(),C.c_size_t(),C.create_string_buffer(2048)
    args=[code,len(code),types] if types is not None else [code,len(code),None,0,0xffffffff,0,options,None,0]
    rc=(retype if types is not None else lower)(*args,C.byref(out),C.byref(size),error,len(error))
    assert rc==0,(rc,error.value)
    if not out.value:
        assert size.value==0
        return None
    try:return C.string_at(out,size.value)
    finally:lib.ScorpiFreeDxilBinary(out)
def ir(code,tmp):
    count=struct.unpack_from('<I',code,28)[0]
    for off in struct.unpack_from('<%dI'%count,code,32):
        if code[off:off+4]==b'DXIL':
            start=off+8;offset,size=struct.unpack_from('<II',code,start+16)
            (tmp/'out.bc').write_bytes(code[start+8+offset:start+8+offset+size])
            return subprocess.check_output([str(dis),str(tmp/'out.bc'),'-o','-'],text=True)
    raise AssertionError('no DXIL')
with tempfile.TemporaryDirectory(prefix='scorpi-flat-linkage-') as directory:
    tmp=Path(directory)
    for mode,extra in [('flat',[]),('no-position',['-DNO_POSITION=1']),('linear',['-DINTERPOLATION=linear']),('centroid',['-DINTERPOLATION=centroid']),('uint',['-DVALUE_TYPE=uint4']),('minimum',['-DVALUE_TYPE=min16float4']),('native16',['-DVALUE_TYPE=float16_t4','-enable-16bit-types'])]:
        shaders=[]
        for stage in ['VS','PS']:
            path=tmp/(stage+'.dxil');profile=stage.lower()+('_6_2' if mode=='native16' else '_6_0')
            subprocess.run([str(build/'bin/dxc'),'-T',profile,'-E',stage+'Main','-Fo',str(path),*extra,str(fixture)],check=True,capture_output=True)
            shaders.append(path.read_bytes())
        consumer,producer=reflect(shaders[1]),reflect(shaders[0]);flat=any(e['flat'] for e in consumer)
        inputs=link(producer,consumer)
        expected=mode in ['flat','minimum','no-position']
        assert flat==expected,(mode,flat)
        converted=adapt(shaders[1])
        if expected:
            assert converted and 0x1111 in inputs and any(e['type']==3 for e in producer)
            if mode=='no-position':
                assert next(e['register'] for e in consumer if not e['output']) != next(e['register'] for e in producer if e['output'])
            text=ir(converted,tmp)
            assert 'call i32 @dx.op.loadInput.i32' in text and 'bitcast i32' in text,(mode,text)
            assert not adapt(converted),mode
            producer=adapt(shaders[0],types=inputs)
            assert producer
            text=ir(producer,tmp)
            assert 'call void @dx.op.storeOutput.i32' in text and 'bitcast float' in text,(mode,text)
            assert not adapt(producer,types=inputs),mode
            # Built-in position stays floating point in both producer and consumer.
            assert 'call void @dx.op.storeOutput.f32' in text
            if mode=='flat':assert adapt(shaders[1],options=0) is None
        else:
            assert converted is None,(mode,ir(converted,tmp) if converted else '')
        print('PASS',mode)
