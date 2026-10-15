"""Headless Vulkan VRAM transfer-pattern smoke on the actual RTX 2070 Super.

Uses the installed nouveau ICD, never a CPU Vulkan fallback. This validates a
1/4 GiB device-local allocation and transfers, not CUDA or shader arithmetic.
Run inside the isolated user test session with render access and the watchdog.
"""
import ctypes as c
import json
import os
from pathlib import Path
import time

U = c.c_uint32
Q = c.c_uint64
V = c.c_void_p
F = c.c_float
P = c.POINTER
H = Q  # Non-dispatchable Vulkan handles on this inventoried 64-bit host.


def structure(name, fields):
    return type(name, (c.Structure,), {'_fields_': fields})


App = structure('App', [('sType', U), ('pNext', V), ('name', c.c_char_p), ('version', U),
                        ('engine', c.c_char_p), ('engineVersion', U), ('api', U)])
InstanceInfo = structure('InstanceInfo', [('sType', U), ('pNext', V), ('flags', U),
    ('app', P(App)), ('layerCount', U), ('layers', V), ('extensionCount', U), ('extensions', V)])
QueueInfo = structure('QueueInfo', [('sType', U), ('pNext', V), ('flags', U),
    ('family', U), ('count', U), ('priorities', P(F))])
DeviceInfo = structure('DeviceInfo', [('sType', U), ('pNext', V), ('flags', U),
    ('queueCount', U), ('queues', P(QueueInfo)), ('layerCount', U), ('layers', V),
    ('extensionCount', U), ('extensions', V), ('features', V)])
QueueFamily = structure('QueueFamily', [('flags', U), ('count', U), ('timestampBits', U),
    ('granularity', U * 3)])
MemoryType = structure('MemoryType', [('flags', U), ('heap', U)])
Heap = structure('Heap', [('size', Q), ('flags', U)])
MemoryProperties = structure('MemoryProperties', [('typeCount', U), ('types', MemoryType * 32),
    ('heapCount', U), ('heaps', Heap * 16)])
BufferInfo = structure('BufferInfo', [('sType', U), ('pNext', V), ('flags', U), ('size', Q),
    ('usage', U), ('sharing', U), ('familyCount', U), ('families', V)])
Requirements = structure('Requirements', [('size', Q), ('alignment', Q), ('typeBits', U)])
Allocation = structure('Allocation', [('sType', U), ('pNext', V), ('size', Q), ('type', U)])
PoolInfo = structure('PoolInfo', [('sType', U), ('pNext', V), ('flags', U), ('family', U)])
CommandAllocation = structure('CommandAllocation', [('sType', U), ('pNext', V),
    ('pool', H), ('level', U), ('count', U)])
CommandBegin = structure('CommandBegin', [('sType', U), ('pNext', V), ('flags', U), ('inherit', V)])
Barrier = structure('Barrier', [('sType', U), ('pNext', V), ('source', U), ('destination', U)])
Copy = structure('Copy', [('source', Q), ('destination', Q), ('size', Q)])
FenceInfo = structure('FenceInfo', [('sType', U), ('pNext', V), ('flags', U)])
Submit = structure('Submit', [('sType', U), ('pNext', V), ('waitCount', U), ('waits', V),
    ('waitStages', V), ('commandCount', U), ('commands', P(V)), ('signalCount', U), ('signals', V)])


def check(code, operation):
    if code != 0:
        raise RuntimeError(f'{operation}: VkResult {code}')


def main():
    root = Path(os.environ['QUALIFICATION_ROOT']) / os.environ.get('QUALIFICATION_GPU_LABEL', 'gpu-results')
    root.mkdir(exist_ok=False)
    icd = '/usr/share/vulkan/icd.d/nouveau_icd.json'
    if not Path(icd).is_file():
        raise RuntimeError('Installed nouveau Vulkan ICD is missing')
    os.environ['VK_DRIVER_FILES'] = icd
    os.environ['VK_ICD_FILENAMES'] = icd
    vk = c.CDLL('libvulkan.so.1')

    def fn(name, types, result=c.c_int32):
        f = getattr(vk, name)
        f.argtypes, f.restype = types, result
        return f

    create_instance = fn('vkCreateInstance', [P(InstanceInfo), V, P(V)])
    enumerate_devices = fn('vkEnumeratePhysicalDevices', [V, P(U), P(V)])
    get_properties = fn('vkGetPhysicalDeviceProperties', [V, V], None)
    get_families = fn('vkGetPhysicalDeviceQueueFamilyProperties', [V, P(U), P(QueueFamily)], None)
    get_memory = fn('vkGetPhysicalDeviceMemoryProperties', [V, P(MemoryProperties)], None)
    create_device = fn('vkCreateDevice', [V, P(DeviceInfo), V, P(V)])
    get_queue = fn('vkGetDeviceQueue', [V, U, U, P(V)], None)
    create_buffer = fn('vkCreateBuffer', [V, P(BufferInfo), V, P(H)])
    get_requirements = fn('vkGetBufferMemoryRequirements', [V, H, P(Requirements)], None)
    allocate = fn('vkAllocateMemory', [V, P(Allocation), V, P(H)])
    bind = fn('vkBindBufferMemory', [V, H, H, Q])
    map_memory = fn('vkMapMemory', [V, H, Q, Q, U, P(V)])
    create_pool = fn('vkCreateCommandPool', [V, P(PoolInfo), V, P(H)])
    allocate_commands = fn('vkAllocateCommandBuffers', [V, P(CommandAllocation), P(V)])
    begin = fn('vkBeginCommandBuffer', [V, P(CommandBegin)])
    fill = fn('vkCmdFillBuffer', [V, H, Q, Q, U], None)
    barrier = fn('vkCmdPipelineBarrier', [V, U, U, U, U, P(Barrier), U, V, U, V], None)
    copy = fn('vkCmdCopyBuffer', [V, H, H, U, P(Copy)], None)
    end = fn('vkEndCommandBuffer', [V])
    create_fence = fn('vkCreateFence', [V, P(FenceInfo), V, P(H)])
    submit = fn('vkQueueSubmit', [V, U, P(Submit), H])
    wait = fn('vkWaitForFences', [V, U, P(H), U, Q])
    reset_fence = fn('vkResetFences', [V, U, P(H)])
    reset_pool = fn('vkResetCommandPool', [V, H, U])

    app = App(0, None, b'cygnus-vram-smoke', 1, b'qualification', 1, 1 << 22)
    instance = V()
    check(create_instance(c.byref(InstanceInfo(1, None, 0, c.pointer(app), 0, None, 0, None)),
                          None, c.byref(instance)), 'create instance')
    count = U()
    check(enumerate_devices(instance, c.byref(count), None), 'enumerate device count')
    devices = (V * count.value)()
    check(enumerate_devices(instance, c.byref(count), devices), 'enumerate devices')
    identities, physical = [], None
    for handle in devices:
        blob = c.create_string_buffer(4096)
        get_properties(handle, blob)
        words = c.cast(blob, P(U))
        name = bytes(blob)[20:276].split(b'\0', 1)[0].decode(errors='replace')
        identity = {'name': name, 'vendor_id': hex(words[2]), 'device_id': hex(words[3]),
                    'device_type': words[4], 'api_version': words[0], 'driver_version': words[1]}
        identities.append(identity)
        if words[2] == 0x10DE and words[3] == 0x1E84 and words[4] == 2:
            physical = handle
    receipt = {'status': 'device discovery', 'icd': icd, 'devices': identities,
               'scope': 'Declared 1/4 GiB VRAM fill/copy/readback, not CUDA or shader arithmetic'}
    destination = root / 'receipt.json'
    destination.write_text(json.dumps(receipt, indent=2) + '\n')
    if physical is None:
        raise RuntimeError('Actual discrete RTX 2070 Super not exposed; CPU fallback refused')
    if os.environ.get('QUALIFICATION_GPU_INVENTORY_ONLY') == '1':
        receipt['status'] = 'actual RTX identified; no allocation or load'
        destination.write_text(json.dumps(receipt, indent=2) + '\n')
        fn('vkDestroyInstance', [V, V], None)(instance, None)
        print(json.dumps(receipt), flush=True)
        return
    count = U()
    get_families(physical, c.byref(count), None)
    families = (QueueFamily * count.value)()
    get_families(physical, c.byref(count), families)
    family = next(i for i, f in enumerate(families) if f.count and f.flags & 4)
    priority = F(1)
    queue_info = QueueInfo(2, None, 0, family, 1, c.pointer(priority))
    device_info = DeviceInfo(3, None, 0, 1, c.pointer(queue_info), 0, None, 0, None, None)
    device = V()
    check(create_device(physical, c.byref(device_info), None, c.byref(device)), 'create device')
    queue = V()
    get_queue(device, family, 0, c.byref(queue))
    memory = MemoryProperties()
    get_memory(physical, c.byref(memory))
    receipt['heaps'] = [{'bytes': memory.heaps[i].size, 'flags': memory.heaps[i].flags}
                        for i in range(memory.heapCount)]
    allocations = []

    def buffer(size, required, forbidden):
        handle = H()
        check(create_buffer(device, c.byref(BufferInfo(12, None, 0, size, 3, 0, 0, None)),
                            None, c.byref(handle)), 'create buffer')
        requirements = Requirements()
        get_requirements(device, handle, c.byref(requirements))
        indices = [i for i in range(memory.typeCount) if requirements.typeBits & (1 << i)
                   and memory.types[i].flags & required == required
                   and not memory.types[i].flags & forbidden]
        if not indices:
            raise RuntimeError(f'No required memory type: required={required}, forbidden={forbidden}')
        index, allocation = indices[0], H()
        check(allocate(device, c.byref(Allocation(5, None, requirements.size, index)),
                       None, c.byref(allocation)), 'allocate buffer memory')
        check(bind(device, handle, allocation, 0), 'bind buffer memory')
        allocations.append((handle, allocation))
        return handle, allocation, index

    gib = int(os.environ.get('QUALIFICATION_GPU_GIB', '1'))
    if gib not in (1, 4):
        raise RuntimeError('Only the declared 1 GiB and 4 GiB smoke sizes are allowed')
    segments = int(os.environ.get('QUALIFICATION_GPU_SEGMENTS', '1'))
    if segments not in (1, 4) or (segments == 4 and gib != 4):
        raise RuntimeError('Only one allocation or four 1 GiB allocations are declared')
    size, chunk = gib * 1024**3, 64 * 1024**2
    segment_bytes = size // segments
    if segments > 1:
        limits = json.loads((Path(os.environ['QUALIFICATION_ROOT']) / 'gpu-limits.json').read_text())['devices'][0]
        if segment_bytes > min(limits['maxBufferSize'], limits['maxMemoryAllocationSize']):
            raise RuntimeError('Segment exceeds the queried driver limits')
    # All device-local allocations stay live throughout all four pattern phases.
    local_buffers = [buffer(segment_bytes, 1, 2) for _ in range(segments)]
    host, host_memory, host_type = buffer(chunk, 2 | 4, 1)  # Host-visible/coherent, not VRAM.
    mapped = V()
    check(map_memory(device, host_memory, 0, chunk, 0, c.byref(mapped)), 'map readback')
    pool = H()
    check(create_pool(device, c.byref(PoolInfo(39, None, 0, family)), None, c.byref(pool)), 'create pool')
    command = V()
    check(allocate_commands(device, c.byref(CommandAllocation(40, None, pool, 0, 1)),
                            c.byref(command)), 'allocate command buffer')
    fence = H()
    check(create_fence(device, c.byref(FenceInfo(8, None, 0)), None, c.byref(fence)), 'create fence')
    patterns = [0, 0xFFFFFFFF, 0xA5A5A5A5, 0x5A5A5A5A]
    started = time.monotonic()
    receipt.update(status='testing', tested_vram_bytes=size, readback_chunk_bytes=chunk,
                   allocation_count=segments, allocation_bytes=segment_bytes,
                   device_local_types=[v[2] for v in local_buffers], host_coherent_type=host_type,
                   patterns_completed=[], phase_patterns_by_allocation=[])
    destination.write_text(json.dumps(receipt, indent=2) + '\n')
    for phase in range(len(patterns)):
        # Rotate the four declared patterns: each allocation gets all four,
        # while all allocations contain different data before any readback.
        phase_patterns = [patterns[(phase + i) % len(patterns)] for i in range(segments)]
        for segment, (local, _, _) in enumerate(local_buffers):
            expected = phase_patterns[segment].to_bytes(4, 'little') * (chunk // 4)
            for offset in range(0, segment_bytes, chunk):
                check(reset_pool(device, pool, 0), 'reset command pool')
                check(reset_fence(device, 1, c.byref(fence)), 'reset fence')
                check(begin(command, c.byref(CommandBegin(42, None, 1, None))), 'begin commands')
                if segment == 0 and offset == 0:
                    for i, (target, _, _) in enumerate(local_buffers):
                        fill(command, target, 0, segment_bytes, phase_patterns[i])
                # Transfer writes must become visible before the readback copy.
                memory_barrier = Barrier(46, None, 0x1000, 0x800)
                barrier(command, 0x1000, 0x1000, 0, 1, c.byref(memory_barrier), 0, None, 0, None)
                copy(command, local, host, 1, c.byref(Copy(offset, 0, chunk)))
                host_barrier = Barrier(46, None, 0x1000, 0x2000)
                barrier(command, 0x1000, 0x4000, 0, 1, c.byref(host_barrier), 0, None, 0, None)
                check(end(command), 'end commands')
                submission = Submit(4, None, 0, None, None, 1, c.pointer(command), 0, None)
                check(submit(queue, 1, c.byref(submission), fence), 'queue submit')
                check(wait(device, 1, c.byref(fence), 1, 5_000_000_000), 'five-second fence wait')
                if c.string_at(mapped, chunk) != expected:
                    raise RuntimeError(f'VRAM pattern mismatch: phase={phase}, allocation={segment}, offset={offset}')
        receipt['patterns_completed'].append(hex(patterns[phase]))
        receipt['phase_patterns_by_allocation'].append([hex(v) for v in phase_patterns])
        destination.write_text(json.dumps(receipt, indent=2) + '\n')
        print(f'GPU phase {phase}: {gib} GiB verified across {segments} allocation(s)', flush=True)
    elapsed = time.monotonic() - started
    receipt.update(status='passed', elapsed_seconds=elapsed, verified_readback_bytes=size * len(patterns),
        end_to_end_verified_mib_s=size * len(patterns) / 1024**2 / elapsed,
        rate_scope='Includes GPU fill/copy, synchronization and CPU byte verification; not peak PCIe bandwidth')
    destination.write_text(json.dumps(receipt, indent=2) + '\n')
    fn('vkUnmapMemory', [V, H], None)(device, host_memory)
    fn('vkDestroyFence', [V, H, V], None)(device, fence, None)
    fn('vkDestroyCommandPool', [V, H, V], None)(device, pool, None)
    for handle, allocation in allocations:
        fn('vkDestroyBuffer', [V, H, V], None)(device, handle, None)
        fn('vkFreeMemory', [V, H, V], None)(device, allocation, None)
    fn('vkDestroyDevice', [V, V], None)(device, None)
    fn('vkDestroyInstance', [V, V], None)(instance, None)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
