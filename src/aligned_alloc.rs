use std::alloc::{AllocError, Allocator, Global, Layout};
use std::cmp;
use std::ptr::NonNull;

// Maximum SIMD vector size.
const MIN_ALIGN: usize = if cfg!(target_feature = "avx512f") {
    64
} else if cfg!(target_feature = "avx") {
    32
} else {
    16
};

/// Aligned memory allocator, using the global allocator with SIMD vector alignment as default.
#[derive(Clone, Copy, Debug)]
pub struct AlignedAlloc<A: Allocator = Global, const N: usize = MIN_ALIGN> {
    alloc: A,
}

impl<A: Allocator, const N: usize> AlignedAlloc<A, N> {
    /// Creates a new aligned allocator based on the specified allocator.
    pub fn new(alloc: A) -> Self {
        assert!(N.is_power_of_two());

        Self { alloc }
    }
}

unsafe impl<A: Allocator, const N: usize> Allocator for AlignedAlloc<A, N> {
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
        self.alloc.grow(
            ptr,
            aligned_layout::<N>(old_layout),
            aligned_layout::<N>(new_layout),
        )
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
        self.alloc.shrink(
            ptr,
            aligned_layout::<N>(old_layout),
            aligned_layout::<N>(new_layout),
        )
    }
}

fn aligned_layout<const N: usize>(layout: Layout) -> Layout {
    unsafe { Layout::from_size_align_unchecked(layout.size(), cmp::max(layout.align(), N)) }
}
