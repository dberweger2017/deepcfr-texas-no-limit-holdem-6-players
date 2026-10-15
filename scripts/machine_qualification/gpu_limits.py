"""Read-only NVK buffer/allocation-limit query after the rejected 4 GiB buffer."""
import ctypes as c
import json
import os
from pathlib import Path

from gpu_vram import App, InstanceInfo, P, Q, U, V, check, structure

os.environ['VK_DRIVER_FILES'] = '/usr/share/vulkan/icd.d/nouveau_icd.json'
os.environ['VK_ICD_FILENAMES'] = os.environ['VK_DRIVER_FILES']
vk = c.CDLL('libvulkan.so.1')
def fn(name, args, result=c.c_int32):
    f = getattr(vk, name)
    f.argtypes, f.restype = args, result
    return f

Maintenance3 = structure('Maintenance3', [('sType', U), ('pNext', V),
    ('maxPerSetDescriptors', U), ('maxMemoryAllocationSize', Q)])
Maintenance4 = structure('Maintenance4', [('sType', U), ('pNext', V), ('maxBufferSize', Q)])
Properties2 = structure('Properties2', [('sType', U), ('pNext', V), ('properties', c.c_ubyte * 4096)])
app = App(0, None, b'cygnus-read-only-limits', 1, b'qualification', 1, (1 << 22) | (3 << 12))
instance = V()
check(fn('vkCreateInstance', [P(InstanceInfo), V, P(V)])(
    c.byref(InstanceInfo(1, None, 0, c.pointer(app), 0, None, 0, None)),
    None, c.byref(instance)), 'create instance')
count = U()
enumerate_devices = fn('vkEnumeratePhysicalDevices', [V, P(U), P(V)])
check(enumerate_devices(instance, c.byref(count), None), 'enumerate device count')
devices = (V * count.value)()
check(enumerate_devices(instance, c.byref(count), devices), 'enumerate devices')
rows = []
for device in devices:
    p4 = Maintenance4(1000413001, None, 0)
    p3 = Maintenance3(1000168000, c.cast(c.pointer(p4), V), 0, 0)
    props = Properties2(1000059001, c.cast(c.pointer(p3), V))
    fn('vkGetPhysicalDeviceProperties2', [V, P(Properties2)], None)(device, c.byref(props))
    words = c.cast(props.properties, P(U))
    if words[2] != 0x10DE or words[3] != 0x1E84 or words[4] != 2:
        raise SystemExit('Actual discrete RTX required')
    rows.append({'vendor_id': hex(words[2]), 'device_id': hex(words[3]),
        'device_type': words[4], 'maxBufferSize': p4.maxBufferSize,
        'maxMemoryAllocationSize': p3.maxMemoryAllocationSize,
        'requested_failed_buffer_bytes': 4 * 1024**3})
receipt = {'status': 'read-only properties query; no device allocation or queue submission',
           'devices': rows, 'header_source': 'Khronos Vulkan-Headers v1.4.309'}
out = Path(os.environ['QUALIFICATION_ROOT']) / 'gpu-limits.json'
with out.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
fn('vkDestroyInstance', [V, V], None)(instance, None)
print(json.dumps(receipt), flush=True)
