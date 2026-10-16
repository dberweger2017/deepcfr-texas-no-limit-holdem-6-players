//! Requested allocation bytes, separate from OS residency/allocator overhead.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
pub struct Meter;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
thread_local! {
    static LOCAL_LIVE: std::cell::Cell<i128> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
fn local_delta(n: i128) {
    // libtest's coordinator also allocates. Its work must not enter a test's
    // table/observer accounting window, including during thread teardown.
    let _ = LOCAL_LIVE.try_with(|live| live.set(live.get() + n));
}
fn add(n: usize) {
    let live = LIVE.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(live, Ordering::Relaxed);
    #[cfg(test)]
    local_delta(n as i128);
}
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            add(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            add(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        System.dealloc(p, layout);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        #[cfg(test)]
        local_delta(-(layout.size() as i128));
    }
    unsafe fn realloc(&self, p: *mut u8, old: Layout, size: usize) -> *mut u8 {
        let q = System.realloc(p, old, size);
        if !q.is_null() {
            if size >= old.size() {
                add(size - old.size());
            } else {
                LIVE.fetch_sub(old.size() - size, Ordering::Relaxed);
                #[cfg(test)]
                local_delta(-((old.size() - size) as i128));
            }
        }
        q
    }
}
pub fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}
#[cfg(test)]
pub fn accounted_live() -> i128 {
    LOCAL_LIVE.with(|live| live.get())
}
pub fn peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}
pub fn reset_peak() {
    PEAK.store(live(), Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn accounting_excludes_allocations_on_other_threads() {
        let start = AtomicBool::new(false);
        let allocated = AtomicBool::new(false);
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                while !start.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                let payload = std::hint::black_box(vec![0u8; 144]);
                allocated.store(true, Ordering::Release);
                while !done.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                drop(payload);
            });
            let before = accounted_live();
            start.store(true, Ordering::Release);
            while !allocated.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            let after = accounted_live();
            done.store(true, Ordering::Release);
            worker.join().unwrap();
            assert_eq!(before, after);
        });
        let before = accounted_live();
        let own = std::hint::black_box(vec![0u8; 144]);
        assert_eq!(accounted_live() - before, 144);
        drop(own);
        assert_eq!(accounted_live(), before);
    }
}
