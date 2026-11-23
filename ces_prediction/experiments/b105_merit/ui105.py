"""Windows current-user DPAPI for local notifier credentials; no plaintext file."""
import base64,ctypes,json
from ctypes import wintypes
class Blob(ctypes.Structure):
    _fields_=[('cbData',wintypes.DWORD),('pbData',ctypes.POINTER(ctypes.c_ubyte))]
def transform(data,decrypt=False):
    buf=ctypes.create_string_buffer(data);src=Blob(len(data),ctypes.cast(buf,ctypes.POINTER(ctypes.c_ubyte)));dst=Blob()
    crypt=ctypes.WinDLL('crypt32',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    fn=crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    fn.argtypes=[ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
    fn.restype=wintypes.BOOL
    if not fn(ctypes.byref(src),None,None,None,None,1,ctypes.byref(dst)):raise ctypes.WinError(ctypes.get_last_error())
    try:return ctypes.string_at(dst.pbData,dst.cbData)
    finally:
        kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p;kernel.LocalFree(dst.pbData)
def protect(value):return base64.b64encode(transform(json.dumps(value).encode())).decode('ascii')
def unprotect(value):return json.loads(transform(base64.b64decode(value),True).decode())
