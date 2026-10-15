//! Requested allocation bytes, separate from OS residency/allocator overhead.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
pub struct Meter;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn add(n: usize) {
    let live = LIVE.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(live, Ordering::Relaxed);
}
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() { add(layout.size()); }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() { add(layout.size()); }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        System.dealloc(p, layout);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, old: Layout, size: usize) -> *mut u8 {
        let q = System.realloc(p, old, size);
        if !q.is_null() {
            if size >= old.size() { add(size - old.size()); }
            else { LIVE.fetch_sub(old.size() - size, Ordering::Relaxed); }
        }
        q
    }
}
pub fn live() -> usize { LIVE.load(Ordering::Relaxed) }
pub fn peak() -> usize { PEAK.load(Ordering::Relaxed) }
pub fn reset_peak() { PEAK.store(live(), Ordering::Relaxed); }
