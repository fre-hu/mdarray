use std::alloc::{AllocError, Allocator, Global, Layout};
use std::cmp;
use std::ptr::NonNull;

/// Aligned memory allocator, using the global allocator as default.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlignedAlloc<const N: usize, A: Allocator = Global> {
    alloc: A,
}

impl<const N: usize, A: Allocator> AlignedAlloc<N, A> {
    /// Creates a new aligned allocator based on the specified allocator.
    pub fn new(alloc: A) -> Self {
        assert!(N.is_power_of_two(), "alignment must be power of two");

        Self { alloc }
    }
}

unsafe impl<const N: usize, A: Allocator> Allocator for AlignedAlloc<N, A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc.allocate(aligned_layout::<N>(layout))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.alloc.deallocate(ptr, aligned_layout::<N>(layout))
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc.allocate_zeroed(aligned_layout::<N>(layout))
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc.grow(ptr, aligned_layout::<N>(old_layout), aligned_layout::<N>(new_layout))
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc.grow_zeroed(
            ptr,
            aligned_layout::<N>(old_layout),
            aligned_layout::<N>(new_layout),
        )
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc.shrink(ptr, aligned_layout::<N>(old_layout), aligned_layout::<N>(new_layout))
    }
}

fn aligned_layout<const N: usize>(layout: Layout) -> Layout {
    // Align to the specified value, but not larger than the layout size rounded
    // to the next power of two and not smaller than the layout alignment.
    let align = cmp::min(N, layout.size().next_power_of_two());

    unsafe { Layout::from_size_align_unchecked(layout.size(), cmp::max(layout.align(), align)) }
}
